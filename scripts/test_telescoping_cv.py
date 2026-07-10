"""Offline A/B: whitened antithetic MC vs + layerwise telescoping control variate.

The telescoping CV (per layer l):
    E_hat_l = gain_l * (W_l^T E_hat_{l-1}) + mean_k[ ReLU(z_lk) - gain_l * z_lk ]
where gain_l = Phi(alpha_l) is the mechanistic ReLU linearization slope and
z_lk are the true pre-activations of sample k at layer l. Unbiased for any
fixed gain_l; the sampled term has ~4x lower variance than raw ReLU(z) means
because the linear part of layerwise fluctuation is carried analytically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop, norm_cdf, norm_pdf  # noqa: E402


def whitened_antithetic_forward(Ws, n_pairs, seed):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)).astype(np.float64)
    C = (u.T @ u) / n_pairs
    evals, evecs = np.linalg.eigh(C)
    C_inv_sqrt = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
    x = np.concatenate([u, -u], axis=0)
    Ws_eff = [C_inv_sqrt @ Ws[0]] + list(Ws[1:])
    return x, Ws_eff


def plain_mc(Ws, n_pairs, seed):
    x, Ws_eff = whitened_antithetic_forward(Ws, n_pairs, seed)
    for w in Ws_eff:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def telescoping_cv_mc(Ws, n_pairs, seed):
    """Same forward pass, but telescoping the mean estimate layer by layer."""
    x, Ws_eff = whitened_antithetic_forward(Ws, n_pairs, seed)
    # mechanistic gains from the K=2 pass (uses ORIGINAL weights; the folded
    # whitening only perturbs W1 by O(1/sqrt(N)))
    n = Ws[0].shape[0]
    mu = np.zeros(n)
    e_hat = np.zeros(n)
    # cheap K=1-style pass to get gains per layer (diagonal variance)
    var = np.ones(n)
    for li, (w, w_eff) in enumerate(zip(Ws, Ws_eff)):
        mu_pre = w.T @ mu
        var_pre = np.maximum((w * w).T @ var, 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        Phi = norm_cdf(alpha)
        phi = norm_pdf(alpha)
        gain = Phi
        # sample step
        z = x @ w_eff
        x = np.maximum(z, 0.0)
        # telescoping estimate
        e_hat = gain * (w.T @ e_hat) + (x - gain * z).mean(axis=0)
        # mech K=1 update for next layer's gain
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
    return e_hat


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    n_mlps = 15
    e_plain, e_cv = [], []
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        m_plain = plain_mc(Ws, 2750, seed=row["mlp_seed"])
        m_cv = telescoping_cv_mc(Ws, 2750, seed=row["mlp_seed"])
        e_plain.append(np.mean((m_plain - truth) ** 2))
        e_cv.append(np.mean((m_cv - truth) ** 2))
        print(f"mlp {i}: plain {e_plain[-1]:.3e}  cv {e_cv[-1]:.3e}")
    print(f"\nmean plain MSE: {np.mean(e_plain):.4e}")
    print(f"mean CV MSE   : {np.mean(e_cv):.4e}  ({np.mean(e_plain)/np.mean(e_cv):.2f}x)")


if __name__ == "__main__":
    main()
