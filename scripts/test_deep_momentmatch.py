"""Test: moment-matching the sample batch at layers 1..T.

Layer 1 targets are exact (z1 Gaussian). Layers >= 2 use the K=2 mech
propagation's (mean, cov) as targets -- this pins the batch's first two
moments to the mech trajectory (bias = mech moment error, systematic and
corrector-learnable) while the samples retain their true higher-moment
structure (variance reduction without collapsing to the mech estimate).

Two variants per T:
  full: affine map matching mean + full covariance   (N x n^2 matmul / layer)
  diag: elementwise scale+shift matching mean + var  (N x n / layer, ~free)

Sweep T in {1, 2, 4, 8, 16, 32} on 15 mini MLPs, 2750 antithetic pairs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import norm_cdf, norm_pdf  # noqa: E402
from test_h1_momentmatch import mat_sqrt, relu_cov_exact  # noqa: E402


def mech_targets(Ws):
    """K=2 propagation (exact layer-1, gain method after): per-layer
    (mean, cov) of post-ReLU activations."""
    n = Ws[0].shape[0]
    targets = []
    # layer 1 exact
    m, cov = relu_cov_exact(Ws[0])
    targets.append((m, cov))
    mu, C = m, cov
    for w in Ws[1:]:
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", C, w, w, optimize=True)
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        Phi, phi = norm_cdf(alpha), norm_pdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        C = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(C, var_post)
        targets.append((mu.copy(), C.copy()))
    return targets


def run(Ws, seed, T, mode, targets, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u_half = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u_half, -u_half], axis=0)
    for li, w in enumerate(Ws):
        x = np.maximum(x @ w, 0.0)
        if li < T:
            m_t, cov_t = targets[li]
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            if mode == "full":
                cov_emp = (xc.T @ xc) / len(x)
                A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
                x = xc @ A + m_t
            else:  # diag
                var_emp = np.maximum((xc * xc).mean(axis=0), 1e-24)
                s = np.sqrt(np.maximum(np.diagonal(cov_t), 0.0) / var_emp)
                x = xc * s + m_t
    return x.mean(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    n_mlps = 15
    results = {}
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        targets = mech_targets(Ws)
        for T in (1, 2, 4, 8, 16, 32):
            for mode in ("full", "diag"):
                key = f"{mode}_T{T}"
                est = run(Ws, row["mlp_seed"], T, mode, targets)
                results.setdefault(key, []).append(np.mean((est - truth) ** 2))
        print(f"mlp {i} done", flush=True)
    print()
    for key in sorted(results, key=lambda k: (k.split('_')[0], int(k.split('T')[1]))):
        print(f"{key:10s} MSE {np.mean(results[key]):.4e}")


if __name__ == "__main__":
    main()
