"""Test: "Monte Carlo sampling with cumulant extrapolation" (paper Appendix G,
baseline #3) applied to OUR estimation task.

Idea: instead of averaging raw ReLU(z_32,k) samples for the final layer,
estimate the pre-activation z_32's mean, variance, and THIRD CENTRAL MOMENT
from the same sample batch, then apply the analytic (Edgeworth/Hermite)
correction for E[ReLU(Z)] given a non-Gaussian Z with known first three
cumulants. This only needs summary statistics of z_32, not raw sample
averaging of the (higher-variance) post-ReLU quantity, and needs zero extra
sampling cost -- it's a different function of the SAME batch we already draw.

Derivation (from the paper's Hermite expansion, Section 3.2): for Z with
mean mu, var sigma^2, and third cumulant kappa3, matching the Hermite
expansion point to (mu, sigma^2) so E[He_0]=E[He_1]=E[He_2]=0 automatically,
the leading non-Gaussian correction comes from E[He_3((Z-mu)/sigma)] =
kappa3/sigma^3 (the standardized skewness):

    E[ReLU(Z)] ~= b_0(mu,sigma) + (1/6) * b_3(mu,sigma) * sigma^3 * (kappa3/sigma^3)
                = E_Gaussian[ReLU(Y)] + (1/6) * b_3(mu,sigma) * kappa3

where b_0 is the standard Gaussian ReLU-expectation formula (our K=1/K=2
formula) and b_3(mu,sigma) = E[d^3/dz^3 ReLU(Z)] for Z~N(mu,sigma^2), which
via the paper's relu_wick_coef formula (k>=2 case) is:
    b_3 = (-1)^{3-2} * sigma^{-(3-1)} * He_1(alpha) * phi(alpha)
        = -alpha * phi(alpha) / sigma^2
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop, norm_cdf, norm_pdf  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def relu_mean_gaussian(mu, sigma):
    alpha = mu / sigma
    return mu * norm_cdf(alpha) + sigma * norm_pdf(alpha)


def b3_gaussian(mu, sigma):
    alpha = mu / sigma
    return -alpha * norm_pdf(alpha) / (sigma * sigma)


def cumulant_extrapolated_relu_mean(z_samples):
    """z_samples: (N, n) pre-activation samples. Returns (n,) corrected
    E[ReLU(z)] estimate using sample mean/var/3rd-central-moment of z."""
    mu = z_samples.mean(axis=0)
    zc = z_samples - mu
    var = np.mean(zc * zc, axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-24))
    kappa3 = np.mean(zc**3, axis=0)  # sample third central moment == kappa3 estimate
    base = relu_mean_gaussian(mu, sigma)
    corr = (1.0 / 6.0) * b3_gaussian(mu, sigma) * kappa3
    return base + corr, base  # corrected, and Gaussian-only (no skew) for comparison


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def run(Ws, seed, m_t, cov_t, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h1 = affine_match(h1, m_t, cov_t)
    h = h1
    for w in Ws[1:-1]:
        h = np.maximum(h @ w, 0.0)
    z_final = h @ Ws[-1]  # pre-activation of the LAST layer (before final ReLU)

    raw_mean = np.maximum(z_final, 0.0).mean(axis=0)  # current mmL1 approach
    extrap_mean, gauss_only_mean = cumulant_extrapolated_relu_mean(z_final)
    return raw_mean, extrap_mean, gauss_only_mean


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    errs_raw, errs_extrap, errs_gauss = [], [], []
    n_mlps = 150
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]
        raw, extrap, gauss = run(Ws, row["mlp_seed"], m_t, cov_t)
        errs_raw.append(np.mean((raw - truth) ** 2))
        errs_extrap.append(np.mean((extrap - truth) ** 2))
        errs_gauss.append(np.mean((gauss - truth) ** 2))
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{n_mlps}: raw={np.mean(errs_raw):.3e}  "
                  f"extrap={np.mean(errs_extrap):.3e}  "
                  f"gauss_only={np.mean(errs_gauss):.3e}", flush=True)
    print(f"\nraw (current mmL1)      : {np.mean(errs_raw):.4e}")
    print(f"cumulant-extrapolated   : {np.mean(errs_extrap):.4e}  "
          f"({np.mean(errs_raw)/np.mean(errs_extrap):.2f}x)")
    print(f"gaussian-only (no skew) : {np.mean(errs_gauss):.4e}  "
          f"({np.mean(errs_raw)/np.mean(errs_gauss):.2f}x)")


if __name__ == "__main__":
    main()
