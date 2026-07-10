"""Decisive sampler comparison over 200 full-split MLPs.

The 15-MLP mini sweeps produced subset-noise conclusions (mm beat whitening
1.6x on mini ids 0-14, loses on full ids 0-14). This runs all candidate
samplers on the same 200 full-split MLPs with paired seeds.

Variants:
  whiten        : input whitening + antithetic (submission #4/#5 sampler)
  mmL1          : layer-1 exact moment match only
  mm_diag234    : mmL1 + diagonal pinning layers 2-4 (v3 sampler)
  whiten_mmL1   : input whitening + layer-1 match
  whiten_diag234: input whitening + diag pinning 2-4 (no L1 full match)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def run_variant(Ws, seed, variant, targets, cov_l1, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)).astype(np.float64)
    if variant.startswith("whiten"):
        C = (u.T @ u) / n_pairs
        evals, evecs = np.linalg.eigh(C)
        S = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
        u = u @ S
    x = np.concatenate([u, -u], axis=0)

    do_l1_full = variant in ("mmL1", "mm_diag234", "whiten_mmL1")
    do_diag = variant in ("mm_diag234", "whiten_diag234")

    for li, w in enumerate(Ws):
        x = np.maximum(x @ w, 0.0)
        if li == 0 and do_l1_full:
            m_t, _ = targets[0]
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            cov_emp = (xc.T @ xc) / len(x)
            A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_l1)
            x = xc @ A + m_t
        elif 1 <= li <= 3 and do_diag and li < len(targets):
            m_t, var_t = targets[li]
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            var_emp = np.maximum((xc * xc).mean(axis=0), 1e-24)
            x = xc * np.sqrt(np.maximum(var_t, 0.0) / var_emp) + m_t
    return x.mean(axis=0)


VARIANTS = ["whiten", "mmL1", "mm_diag234", "whiten_mmL1", "whiten_diag234"]


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    errs = {v: [] for v in VARIANTS}
    n_mlps = 200
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        for v in VARIANTS:
            est = run_variant(Ws, row["mlp_seed"], v, fin["targets"], fin["cov_l1"])
            errs[v].append(np.mean((est - truth) ** 2))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{n_mlps}: " +
                  "  ".join(f"{v}={np.mean(errs[v]):.3e}" for v in VARIANTS),
                  flush=True)
    print("\nFinal over", n_mlps, "MLPs:")
    for v in VARIANTS:
        a = np.array(errs[v])
        print(f"{v:16s} mean {a.mean():.4e}  median {np.median(a):.4e}")


if __name__ == "__main__":
    main()
