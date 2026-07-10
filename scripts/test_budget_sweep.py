"""Fast paired sweep: mmL1 sample count vs MSE (and implied adjusted score).

Also compares whitened-antithetic at matched sample counts.
Uses precomputed layer-1 targets; skips full K=2 each time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def mmL1_final(Ws, seed, n_pairs):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_t = relu_cov_exact(Ws[0])
    h = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h, m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(axis=0)


def whiten_anti_final(Ws, seed, n_pairs):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    # half-cov fold (galfaroi style)
    cov = (u.T @ u) / n_pairs
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-6)
    inv_sqrt = (evecs / np.sqrt(evals)) @ evecs.T
    W0 = inv_sqrt @ Ws[0]
    x = np.concatenate([u, -u], axis=0)
    h = np.maximum(x @ W0, 0.0)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 40
    # 2750 pairs ~ our current; ~4900 ~15% budget-ish; scale up/down
    pair_counts = [1500, 2750, 4000, 4900, 7000, 10000]
    B = 2.72e11
    # rough analytical FLOPs per pair (2 samples * depth * 2 w^2)
    flops_per_pair = 2 * 32 * (2 * 256 * 256)

    results = {p: {"mm": [], "wh": []} for p in pair_counts}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        for p in pair_counts:
            mm = mmL1_final(Ws, row["mlp_seed"], p)
            wh = whiten_anti_final(Ws, row["mlp_seed"], p)
            results[p]["mm"].append(np.mean((mm - truth) ** 2))
            results[p]["wh"].append(np.mean((wh - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}", flush=True)

    print(f"\n{'pairs':>7} {'util%':>7} {'mult':>6} {'mmL1 MSE':>12} {'mm adj':>12} "
          f"{'whiten MSE':>12} {'wh adj':>12} {'mm/wh':>7}")
    for p in pair_counts:
        util = p * flops_per_pair / B
        # add ~1.6e9 for K2 if present; ignore for sampler-only compare
        mult = max(0.1, util)
        mm = np.mean(results[p]["mm"])
        wh = np.mean(results[p]["wh"])
        print(
            f"{p:>7} {100*util:>6.1f}% {mult:>6.3f} {mm:>12.4e} {mm*mult:>12.4e} "
            f"{wh:>12.4e} {wh*mult:>12.4e} {wh/mm:>6.2f}x"
        )


if __name__ == "__main__":
    main()
