"""Test: exact moment-matching of the sample batch at layer 1.

z1 = u @ W1 is exactly Gaussian, so E[h1] and Cov[h1] for h1 = ReLU(z1) have
closed forms (bivariate Gaussian ReLU expectation):

    E[ReLU(z_i)]              = sigma_i / sqrt(2 pi)
    E[ReLU(z_i) ReLU(z_j)]    = (sigma_i sigma_j / 2 pi) *
                                ( sqrt(1-rho^2) + rho (pi/2 + arcsin rho) )

Affinely transform the sampled batch h1 so its empirical mean/cov EXACTLY
match the analytic ones, then continue the forward pass. This annihilates all
sampling noise entering through the first two moments of h1 (any order in u),
at an O(1/N) bias of the same nature as input whitening.

Variants (15 mini MLPs, 2750 antithetic pairs):
  A: input whitening (current submission)
  E: layer-1 moment matching only
  F: input whitening + layer-1 moment matching
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def whiten(u):
    C = (u.T @ u) / len(u)
    evals, evecs = np.linalg.eigh(C)
    S = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
    return u @ S


def mat_sqrt(C, inv=False):
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-12)
    d = 1.0 / np.sqrt(evals) if inv else np.sqrt(evals)
    return (evecs * d) @ evecs.T


def relu_cov_exact(W1):
    """Exact mean vector and covariance matrix of ReLU(u @ W1), u ~ N(0, I)."""
    G = W1.T @ W1                       # gram: cov of z1
    sigma = np.sqrt(np.maximum(np.diagonal(G), 1e-24))
    rho = np.clip(G / np.outer(sigma, sigma), -1.0, 1.0)
    m = sigma / math.sqrt(2 * math.pi)
    second = (np.outer(sigma, sigma) / (2 * math.pi)) * (
        np.sqrt(np.maximum(1 - rho * rho, 0.0)) + rho * (math.pi / 2 + np.arcsin(rho))
    )
    cov = second - np.outer(m, m)
    return m, cov


def forward(x, Ws):
    for w in Ws:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def run(Ws, seed, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u_half = rng.standard_normal((n_pairs, n))
    W1 = Ws[0]
    out = {}

    # A: input whitening (current)
    uw = whiten(u_half)
    x = np.concatenate([uw, -uw], axis=0)
    out["A_whiten"] = forward(x, Ws)

    m_exact, cov_exact = relu_cov_exact(W1)
    cov_exact_sqrt = mat_sqrt(cov_exact)

    def moment_match(h1):
        mu_emp = h1.mean(axis=0)
        hc = h1 - mu_emp
        cov_emp = (hc.T @ hc) / len(h1)
        A = mat_sqrt(cov_emp, inv=True) @ cov_exact_sqrt
        return hc @ A + m_exact

    # E: moment match at layer 1 only (plain antithetic input)
    u = np.concatenate([u_half, -u_half], axis=0)
    h1 = np.maximum(u @ W1, 0.0)
    h1m = moment_match(h1)
    out["E_mm"] = forward(h1m, Ws[1:])

    # F: whiten + moment match
    h1w = np.maximum(x @ W1, 0.0)
    h1wm = moment_match(h1w)
    out["F_whiten_mm"] = forward(h1wm, Ws[1:])

    return out


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    errs = {}
    for i in range(15):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        res = run(Ws, seed=row["mlp_seed"])
        for k, v in res.items():
            errs.setdefault(k, []).append(np.mean((v - truth) ** 2))
        print(f"mlp {i}: " + "  ".join(f"{k}={errs[k][-1]:.2e}" for k in sorted(errs)))
    print()
    base = np.mean(errs["A_whiten"])
    for k in sorted(errs):
        m = np.mean(errs[k])
        print(f"{k:14s} MSE {m:.4e}  ({base/m:.2f}x vs whitened)")


if __name__ == "__main__":
    main()
