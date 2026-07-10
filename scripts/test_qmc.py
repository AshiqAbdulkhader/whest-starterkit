"""Test: quasi-Monte Carlo (scrambled Sobol) input in place of pseudo-random.

Sobol sequences have discrepancy O((log N)^d / N) vs pseudo-random's
O(1/sqrt(N)) -- for smooth low-effective-dimension integrands this beats
plain MC even in fairly high dimensions. Scrambled (Owen-scrambled) Sobol
retains an unbiased randomized-QMC estimator with a valid (if pessimistic)
variance estimate from independent scrambles.

Compares (200 full-split MLPs, budget-matched ~2750 antithetic pairs):
  mmL1       : current submission sampler (pseudo-random + antithetic + L1 match)
  qmc_mmL1   : scrambled Sobol (base-2, antithetic via first-half/second-half
               digital-shift trick) + same L1 moment matching
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def sobol_normal(n_total, dim, seed):
    """n_total points (power of 2) via scrambled Sobol -> standard normal
    (inverse CDF). n_total should match 2*n_pairs of the MC baseline for a
    fair per-sample-cost comparison."""
    m = int(round(np.log2(n_total)))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random_base2(m=m)  # (2^m, dim) in (0,1)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    from scipy.special import ndtri

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


def run_mmL1(Ws, seed, m_t, cov_t, n_pairs, qmc_mode):
    n = Ws[0].shape[0]
    if qmc_mode:
        n_total = 1
        while n_total < 2 * n_pairs:
            n_total *= 2
        x = sobol_normal(n_total, n, seed)
    else:
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n_pairs, n))
        x = np.concatenate([u, -u], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h1 = affine_match(h1, m_t, cov_t)
    return forward(h1, Ws[1:])


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    errs_mc, errs_qmc = [], []
    n_mlps = 100
    n_pairs = 2750
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t = fin["targets"][0][0]
        cov_t = fin["cov_l1"]
        e_mc = run_mmL1(Ws, row["mlp_seed"], m_t, cov_t, n_pairs, qmc_mode=False)
        e_qmc = run_mmL1(Ws, row["mlp_seed"], m_t, cov_t, n_pairs, qmc_mode=True)
        errs_mc.append(np.mean((e_mc - truth) ** 2))
        errs_qmc.append(np.mean((e_qmc - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{n_mlps}: mc={np.mean(errs_mc):.3e}  "
                  f"qmc={np.mean(errs_qmc):.3e}", flush=True)
    print(f"\nmmL1 (pseudo-random) : {np.mean(errs_mc):.4e}")
    print(f"mmL1 (scrambled Sobol): {np.mean(errs_qmc):.4e}  "
          f"({np.mean(errs_mc)/np.mean(errs_qmc):.2f}x)")


if __name__ == "__main__":
    main()
