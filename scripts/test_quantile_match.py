"""Test: exact per-neuron marginal quantile matching at layer 1.

z1 = u @ W1 is exactly Gaussian per neuron: z1_j ~ N(0, sigma_j^2). So
h1_j = ReLU(z1_j) has an EXACT closed-form marginal CDF (point mass 0.5 at
zero, continuous half-Gaussian above):
    F(t) = 0                       t < 0
    F(t) = Phi(t / sigma_j)        t >= 0   (note F(0)=0.5)

Rank-transforming each neuron's own samples to match this exact marginal
(inverse CDF at the sample's empirical rank) fixes ALL marginal moments for
that neuron, not just mean+variance -- and does so per-neuron independently,
so cross-neuron (joint/covariance) structure from the raw sample is
untouched. This is complementary to (not a replacement for) affine
mean+covariance matching, which fixes the joint 2nd moment but leaves each
marginal's shape as sampled.

Variants (200 full-split MLPs, 2750 antithetic pairs):
  mmL1          : current submission (affine mean+cov match only)
  qmatch        : quantile match only (marginals exact, cov as-sampled)
  qmatch_mmL1   : quantile match, THEN affine mean+cov match
  mmL1_qmatch   : affine mean+cov match, THEN quantile match
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def quantile_match(x, sigma):
    """Per-column rank -> exact ReLU(N(0,sigma^2)) quantile.

    Ranks in [0, 0.5) map to h=0 (the point mass); ranks in [0.5, 1) map to
    sigma * Phi^{-1}(rank). Uses average ranks for ties (all exact zeros tie).
    """
    N, n = x.shape
    out = np.empty_like(x)
    # ranks via argsort of argsort (0-indexed), midpoint rule
    order = np.argsort(x, axis=0)
    ranks = np.empty_like(order)
    ar = np.arange(N)
    for j in range(n):
        ranks[order[:, j], j] = ar
    q = (ranks + 0.5) / N  # in (0,1)
    below = q < 0.5
    target = np.where(below, 0.0, sigma[None, :] * ndtri(np.clip(q, 0.5, 1 - 1e-9)))
    return target


def forward(x, Ws):
    for w in Ws:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def run(Ws, seed, variant, m_t, cov_t, sigma1, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    W1 = Ws[0]
    h1 = np.maximum(x @ W1, 0.0)

    if variant == "mmL1":
        h1 = affine_match(h1, m_t, cov_t)
    elif variant == "qmatch":
        h1 = quantile_match(h1, sigma1)
    elif variant == "qmatch_mmL1":
        h1 = quantile_match(h1, sigma1)
        h1 = affine_match(h1, m_t, cov_t)
    elif variant == "mmL1_qmatch":
        h1 = affine_match(h1, m_t, cov_t)
        h1 = quantile_match(h1, sigma1)
    return forward(h1, Ws[1:])


VARIANTS = ["mmL1", "qmatch", "qmatch_mmL1", "mmL1_qmatch"]


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
        m_t, var_t = fin["targets"][0]
        cov_t = fin["cov_l1"]
        sigma1 = np.sqrt(np.maximum(np.diagonal(cov_t) + m_t * m_t, 1e-24))
        # sigma1 should be the PRE-relu Gaussian std: recover via W1 norms
        sigma1 = np.sqrt((Ws[0] * Ws[0]).sum(axis=0))
        for v in VARIANTS:
            est = run(Ws, row["mlp_seed"], v, m_t, cov_t, sigma1)
            errs[v].append(np.mean((est - truth) ** 2))
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{n_mlps}: " +
                  "  ".join(f"{v}={np.mean(errs[v]):.3e}" for v in VARIANTS),
                  flush=True)
    print("\nFinal over", n_mlps, "MLPs:")
    for v in VARIANTS:
        a = np.array(errs[v])
        print(f"{v:14s} mean {a.mean():.4e}  median {np.median(a):.4e}")


if __name__ == "__main__":
    main()
