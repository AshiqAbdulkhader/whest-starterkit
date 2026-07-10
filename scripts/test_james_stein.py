"""Test: empirical-Bayes (James-Stein) shrinkage ACROSS THE 256 NEURONS of
one network, not across networks.

Different mechanism from the earlier-failed "Bayesian shrinkage" (round 6):
that used a GLOBALLY fit prior variance tau^2 (from held-out MLPs, ~7.3e-5,
much bigger than the MC noise variance ~4e-6), so blind shrinkage toward the
biased mech prediction hurt everywhere. This instead estimates the effective
prior variance tau^2 FRESH, PER NETWORK, from the observed spread of the 256
neurons' own (mc_estimate - mech_prediction) residuals via method of
moments:

    tau^2_hat = max(0, (sum_i d_i^2 - sum_i sigma_i^2) / k)

where d_i = X_i - m_i (MC minus mech prior), sigma_i^2 = per-neuron MC
sampling variance (already computed as mc_sem^2), k=256. Per-neuron
shrinkage weight w_i = tau^2_hat / (tau^2_hat + sigma_i^2); estimator
mu_hat_i = m_i + w_i * d_i. This is the classical Efron-Morris / XKB
empirical-Bayes estimator, adaptive to how much THIS SPECIFIC network's mech
bias varies (pscamillo's writeup found the K=2 correction factor varies
0.9858-1.0010 across networks -- real per-network variation this could
exploit, unlike a fixed global prior).

Also tests classical (null-shrinkage-toward-zero) JS as a sanity baseline,
and shrinkage toward the K=2 baseline vs the pscamillo-style 0.992-corrected
K=2 baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402

K = 256


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def mmL1_mc(Ws, seed, m_t, cov_t, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h1, m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    mc_mean = h.mean(axis=0)
    mc_var = h.var(axis=0) / len(h)  # per-neuron sampling variance of the mean
    return mc_mean, mc_var


def js_shrink(X, sigma2, m):
    """Empirical-Bayes James-Stein shrinkage of X toward prior mean m."""
    d = X - m
    tau2_hat = max(0.0, (np.sum(d**2) - np.sum(sigma2)) / K)
    w = tau2_hat / (tau2_hat + sigma2 + 1e-30)
    return m + w * d, tau2_hat


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 100
    errs_raw, errs_js_k2, errs_js_zero, errs_js_k2corr = [], [], [], []
    taus = []

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        mu2, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]

        X, sigma2 = mmL1_mc(Ws, row["mlp_seed"], m_t, cov_t)
        errs_raw.append(np.mean((X - truth) ** 2))

        est_k2, tau2 = js_shrink(X, sigma2, mu2)
        errs_js_k2.append(np.mean((est_k2 - truth) ** 2))
        taus.append(tau2)

        est_zero, _ = js_shrink(X, sigma2, np.zeros(K))
        errs_js_zero.append(np.mean((est_zero - truth) ** 2))

        est_k2c, _ = js_shrink(X, sigma2, mu2 * 0.992)
        errs_js_k2corr.append(np.mean((est_k2c - truth) ** 2))

        if (i + 1) % 20 == 0:
            base = np.mean(errs_raw)
            print(f"{i+1}/{n_mlps}: raw={base:.3e}  "
                  f"js_k2={np.mean(errs_js_k2)/base:.3f}x  "
                  f"js_zero={np.mean(errs_js_zero)/base:.3f}x  "
                  f"js_k2corr={np.mean(errs_js_k2corr)/base:.3f}x  "
                  f"mean_tau2={np.mean(taus):.2e}", flush=True)

    base = np.mean(errs_raw)
    print(f"\nraw (mmL1)         : {base:.4e}")
    print(f"js toward K=2       : {np.mean(errs_js_k2):.4e}  ({base/np.mean(errs_js_k2):.3f}x)")
    print(f"js toward zero      : {np.mean(errs_js_zero):.4e}  ({base/np.mean(errs_js_zero):.3f}x)")
    print(f"js toward 0.992*K=2 : {np.mean(errs_js_k2corr):.4e}  ({base/np.mean(errs_js_k2corr):.3f}x)")
    print(f"mean estimated tau^2: {np.mean(taus):.4e}")


if __name__ == "__main__":
    main()
