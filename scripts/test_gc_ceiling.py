"""Gram-Charlier ceiling vs reachable: can better moments beat mmL1?

Reproduces galfaroi's key finding locally on our data, and tests whether
using HIGH-N moments (oracle) vs same-batch moments changes the picture
when the BASE mean is mmL1 rather than plain MC.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward_preacts(Ws, seed, n_pairs, match=True):
    """Return final-layer pre-activations z (N, n) after mmL1 path."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_t = relu_cov_exact(Ws[0])
    h = np.maximum(x @ Ws[0], 0.0)
    if match:
        h = affine_match(h, m_t, cov_t)
    for w in Ws[1:-1]:
        h = np.maximum(h @ w, 0.0)
    z = h @ Ws[-1]
    return z


def gc_relu_mean(mu, sig, skew, kurt):
    """Edgeworth/GC approximation to E[ReLU(Z)] for Z ~ approx with given moments."""
    sig = np.maximum(sig, 1e-12)
    alpha = mu / sig
    Phi = ndtr(alpha)
    phi = pdf(alpha)
    # Gaussian part
    eg = mu * Phi + sig * phi
    # skew correction ~ (skew/6) * sig * He2(alpha) * phi  (standard for E[ReLU])
    # He2(x) = x^2 - 1; contribution to E[(Z)_+] from κ3
    # Using galfaroi/paper style: (κ3/6) * d³/dm³ of Gaussian ReLU moment
    # κ3 = skew * sig^3; κ4 = kurt * sig^4 (excess kurtosis)
    kappa3 = skew * sig**3
    kappa4 = kurt * sig**4
    # derivatives of eg w.r.t mu (holding sig): 
    # deg/dmu = Phi; d2 = phi/sig; d3 = -alpha*phi/sig^2; d4 = (alpha^2-1)*phi/sig^3
    d3 = -alpha * phi / (sig**2)
    d4 = (alpha**2 - 1) * phi / (sig**3)
    return eg + (kappa3 / 6.0) * d3 + (kappa4 / 24.0) * d4


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 30
    n_est = 2750
    n_oracle = 50000  # "true" moments from big independent sample

    errs = {k: [] for k in [
        "mmL1_raw", "gc_batch", "gc_oracle_moments", "gc_oracle_all", "gauss_oracle"
    ]}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)

        z_est = forward_preacts(Ws, row["mlp_seed"], n_est, match=True)
        relu_est = np.maximum(z_est, 0.0)
        mm = relu_est.mean(axis=0)
        errs["mmL1_raw"].append(np.mean((mm - truth) ** 2))

        # batch moments
        mu = z_est.mean(0)
        sig = z_est.std(0)
        xc = z_est - mu
        skew = (xc**3).mean(0) / (sig**3 + 1e-24)
        kurt = (xc**4).mean(0) / (sig**4 + 1e-24) - 3.0
        gc_b = gc_relu_mean(mu, sig, skew, kurt)
        errs["gc_batch"].append(np.mean((gc_b - truth) ** 2))

        # oracle moments from big independent sample (different seed)
        z_or = forward_preacts(Ws, row["mlp_seed"] + 99991, n_oracle, match=True)
        mu_o = z_or.mean(0)
        sig_o = z_or.std(0)
        xc_o = z_or - mu_o
        skew_o = (xc_o**3).mean(0) / (sig_o**3 + 1e-24)
        kurt_o = (xc_o**4).mean(0) / (sig_o**4 + 1e-24) - 3.0

        # GC with oracle moments but... still need a mean source.
        # Use oracle mu/sig/skew/kurt fully:
        gc_o = gc_relu_mean(mu_o, sig_o, skew_o, kurt_o)
        errs["gc_oracle_all"].append(np.mean((gc_o - truth) ** 2))

        # GC with batch mu/sig but oracle skew/kurt (idealized hybrid)
        gc_h = gc_relu_mean(mu, sig, skew_o, kurt_o)
        errs["gc_oracle_moments"].append(np.mean((gc_h - truth) ** 2))

        # Gaussian closure with oracle mu/sig only
        alpha = mu_o / np.maximum(sig_o, 1e-12)
        gauss = mu_o * ndtr(alpha) + sig_o * pdf(alpha)
        errs["gauss_oracle"].append(np.mean((gauss - truth) ** 2))

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}", flush=True)

    base = np.mean(errs["mmL1_raw"])
    print(f"\n{'method':<22} {'MSE':>12} {'vs mmL1':>10}")
    for k, v in errs.items():
        m = np.mean(v)
        print(f"{k:<22} {m:>12.4e} {base/m:>9.3f}x")


if __name__ == "__main__":
    main()
