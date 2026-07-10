"""Test: pure-numpy Latin Hypercube stratification vs scipy Sobol vs plain MC.

QMC (scrambled Sobol) needs a large embedded direction-number table and
careful bit-manipulation to implement correctly without scipy (which is not
a declared/locked dependency and may not exist on the grader). Latin
Hypercube Sampling (LHS) needs none of that: per dimension, stratify into N
equal-probability bins and randomly permute which sample lands in which bin.
Weaker than Sobol at controlling JOINT discrepancy, but if most of Sobol's
gain here comes from per-dimension stratification of the input (plausible,
since our own post-h1 marginal quantile-match test showed no additional gain
once mean+cov are matched -- so the win must be upstream, in how well the
256 INPUT coordinates are spread), LHS should capture most of it safely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402
from test_qmc import sobol_normal  # noqa: E402


def lhs_normal(n_total, dim, seed):
    """Latin Hypercube in each of `dim` coordinates independently, mapped to
    standard normal via inverse CDF. n_total need not be a power of 2."""
    rng = np.random.default_rng(seed)
    # stratified 1D positions (jittered within each stratum), independently
    # permuted per dimension
    strata = (np.arange(n_total)[:, None] + rng.random((n_total, dim))) / n_total
    idx = np.argsort(rng.random((n_total, dim)), axis=0)  # per-column permutation
    u = np.take_along_axis(strata, np.argsort(idx, axis=0), axis=0)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return ndtri(u)


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward(x, Ws):
    for w in Ws:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def run(Ws, seed, m_t, cov_t, n_pairs, mode):
    n = Ws[0].shape[0]
    n_total_pow2 = 1
    while n_total_pow2 < 2 * n_pairs:
        n_total_pow2 *= 2
    if mode == "mc":
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n_pairs, n))
        x = np.concatenate([u, -u], axis=0)
    elif mode == "sobol":
        x = sobol_normal(n_total_pow2, n, seed)
    elif mode == "lhs":
        x = lhs_normal(2 * n_pairs, n, seed)
    elif mode == "lhs_antithetic":
        half = lhs_normal(n_pairs, n, seed)
        x = np.concatenate([half, -half], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h1 = affine_match(h1, m_t, cov_t)
    return forward(h1, Ws[1:])


MODES = ["mc", "sobol", "lhs", "lhs_antithetic"]


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    errs = {m: [] for m in MODES}
    n_mlps = 100
    n_pairs = 2750
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]
        for mode in MODES:
            est = run(Ws, row["mlp_seed"], m_t, cov_t, n_pairs, mode)
            errs[mode].append(np.mean((est - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{n_mlps}: " +
                  "  ".join(f"{m}={np.mean(errs[m]):.3e}" for m in MODES),
                  flush=True)
    print(f"\nFinal over {n_mlps} MLPs:")
    base = np.mean(errs["mc"])
    for m in MODES:
        v = np.mean(errs[m])
        print(f"{m:16s} {v:.4e}  ({base/v:.2f}x vs plain MC)")


if __name__ == "__main__":
    main()
