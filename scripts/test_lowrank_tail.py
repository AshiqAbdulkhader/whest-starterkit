"""Test: low-rank sampled forward in deep layers.

Deep-layer activation covariance is rank-concentrated (rank-32 captures
94-97% of variance past layer 16). Idea: after layer T_switch, represent each
sample's fluctuation in the top-r eigenbasis of the mech-predicted covariance;
matmuls then cost O(r*n) per sample instead of O(n^2), buying ~3-4x more
samples at equal FLOPs. The discarded orthogonal variance is re-injected as
independent Gaussian noise (diagonal of the residual covariance) so ReLU
means don't suffer variance-shrinkage bias.

Compares at (approximately) EQUAL analytical FLOPs:
  A: moment-matched MC, 2750 pairs (current candidate)
  L: moment-matched MC with low-rank tail, pairs scaled up to equal FLOPs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import antithetic_mc, cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def mech_covs(Ws, fin_targets_all=False):
    """Full K=2 covariance at every layer (exact L1 + gain method)."""
    from extract_features import relu_cov_exact, norm_cdf, norm_pdf

    covs, mus = [], []
    mu, cov = relu_cov_exact(Ws[0])
    covs.append(cov.copy()); mus.append(mu.copy())
    for w in Ws[1:]:
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w, optimize=True)
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        Phi, phi = norm_cdf(alpha), norm_pdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        covs.append(cov.copy()); mus.append(mu.copy())
    return mus, covs


def lowrank_tail_mc(Ws, n_pairs, seed, targets, cov_l1, mus, covs,
                    t_switch=8, rank=48):
    """Moment-matched MC with a low-rank + synthetic-noise deep tail."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)).astype(np.float64)
    x = np.concatenate([u, -u], axis=0)
    n_match = len(targets)
    L = len(Ws)
    for li in range(t_switch):
        x = np.maximum(x @ Ws[li], 0.0)
        if li < n_match:
            m_t, var_t = targets[li]
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            if li == 0:
                cov_emp = (xc.T @ xc) / len(x)
                A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_l1)
                x = xc @ A + m_t
            else:
                var_emp = np.maximum((xc * xc).mean(axis=0), 1e-24)
                x = xc * np.sqrt(np.maximum(var_t, 0.0) / var_emp) + m_t

    # ---- low-rank tail (stays in coefficient space through the matmul) ----
    # per-layer per-sample cost: 2rn (z from coefficients) + 2rn (re-project)
    # + O(n), vs 2n^2 full -- ~2.7x cheaper at r=48, n=256.
    for li in range(t_switch, L):
        w = Ws[li]
        C_in = covs[li - 1]
        evals, evecs = np.linalg.eigh(C_in)
        V = evecs[:, -rank:]                       # (n, r) top-r directions
        resid_var = np.maximum(np.diagonal(C_in) - (V * V) @ evals[-rank:], 0.0)
        mu_in = x.mean(axis=0)                     # batch's own mean (full width)
        c = (x - mu_in) @ V                        # (N, r) coefficients
        M = V.T @ w                                # (r, n), n^2 r once per layer
        # discarded orthogonal variance -> diagonal Gaussian noise in z-space
        var_z_resid = (w * w).T @ resid_var        # (n,)
        xi = rng.standard_normal((len(x), len(var_z_resid)))
        z = mu_in @ w + c @ M + xi * np.sqrt(var_z_resid)
        x = np.maximum(z, 0.0)
    return x.mean(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    errs_a, errs_l = [], []
    for i in range(10):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        mu2, fin = cov_prop(Ws)
        mus, covs = mech_covs(Ws)
        m_a, _ = antithetic_mc(Ws, 2750, seed=row["mlp_seed"],
                               mm_targets=fin["targets"], cov_l1=fin["cov_l1"])
        m_l = lowrank_tail_mc(Ws, 2750, row["mlp_seed"], fin["targets"],
                              fin["cov_l1"], mus, covs)
        errs_a.append(np.mean((m_a - truth) ** 2))
        errs_l.append(np.mean((m_l - truth) ** 2))
        print(f"mlp {i}: mm={errs_a[-1]:.3e}  lowrank(same N)={errs_l[-1]:.3e}")
    print(f"\nmm 2750               : {np.mean(errs_a):.4e}")
    print(f"lowrank tail same N   : {np.mean(errs_l):.4e} "
          f"(bias check; must be close to mm before scaling N up)")


if __name__ == "__main__":
    main()
