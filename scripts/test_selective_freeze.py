"""Test: selective-freeze coarse model correlation for MLMC.

Freeze gates where K=2 predicts |alpha| > threshold (nearly deterministic).
Coarse = same weights/inputs, but frozen gates use majority state from K2
sign(mu_pre). If corr(fine, coarse) is high, MLMC can help.

Also reports bias of the frozen-tail estimator used alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop, norm_cdf, norm_pdf, relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def k2_alphas(Ws):
    """Per-layer pre-activation alpha = mu/sigma from K=2 (layers 2..L)."""
    mu, cov = relu_cov_exact(Ws[0])
    alphas = [None]  # layer 1
    for w in Ws[1:]:
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        alphas.append(alpha)
        Phi = norm_cdf(alpha)
        phi = norm_pdf(alpha)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
    return alphas


def forward_pair(Ws, seed, m_t, cov_t, alphas, thresh, n_pairs=2750):
    """Fine (exact) vs coarse (selective freeze) on shared antithetic+mmL1 samples."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)

    h = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h, m_t, cov_t)
    h_c = h.copy()

    freeze_masks = []
    for li, w in enumerate(Ws[1:], start=1):
        z = h @ w
        h = np.maximum(z, 0.0)

        z_c = h_c @ w
        alpha = alphas[li]
        freeze = np.abs(alpha) > thresh  # (n,)
        freeze_masks.append(freeze.mean())
        majority_on = alpha > 0
        # frozen neurons: force gate to majority; unfrozen: use true gate
        gate_true = z_c > 0
        gate = np.where(freeze, majority_on, gate_true)
        h_c = np.where(gate, z_c, 0.0)

    return h, h_c, float(np.mean(freeze_masks))


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 30
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"{'thresh':>7} {'frac_frz':>9} {'corr':>8} {'var_ratio':>10} "
          f"{'fine_MSE':>10} {'coarse_MSE':>11} {'diff_MSE':>10}")

    for thresh in thresholds:
        corrs, ratios, fine_mses, coarse_mses, diff_mses, fracs = [], [], [], [], [], []
        for i in range(n_mlps):
            row = d[i]
            Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
            truth = np.asarray(row["final_means"], dtype=np.float64)
            _, fin = cov_prop(Ws)
            m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]
            alphas = k2_alphas(Ws)
            fine, coarse, frac = forward_pair(
                Ws, row["mlp_seed"], m_t, cov_t, alphas, thresh
            )
            fc = fine - fine.mean(0)
            cc = coarse - coarse.mean(0)
            num = (fc * cc).mean(0)
            den = np.sqrt((fc**2).mean(0) * (cc**2).mean(0)) + 1e-24
            corr = np.median(num / den)
            diff = fine - coarse
            vr = np.median((diff**2).mean(0) / ((fc**2).mean(0) + 1e-24))
            corrs.append(corr)
            ratios.append(vr)
            fracs.append(frac)
            fine_mses.append(np.mean((fine.mean(0) - truth) ** 2))
            coarse_mses.append(np.mean((coarse.mean(0) - truth) ** 2))
            # MLMC-style: use coarse mean as primary + mean(fine-coarse)
            # (same N here just for bias check of the combo identity)
            mlmc_est = coarse.mean(0) + (fine - coarse).mean(0)  # = fine.mean
            diff_mses.append(np.mean((mlmc_est - truth) ** 2))  # sanity = fine

        print(
            f"{thresh:>7.1f} {np.mean(fracs):>9.3f} {np.median(corrs):>8.3f} "
            f"{np.median(ratios):>10.3f} {np.mean(fine_mses):>10.3e} "
            f"{np.mean(coarse_mses):>11.3e} {np.mean(diff_mses):>10.3e}"
        )


if __name__ == "__main__":
    main()
