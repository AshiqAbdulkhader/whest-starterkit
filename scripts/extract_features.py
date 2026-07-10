"""Extract per-neuron features + targets for the learned-correction model.

Runs K=1 (mean) and K=2 (covariance) propagation in raw NumPy (offline, no
flopscope) over every MLP in a baked dataset split, and saves per-final-layer-
neuron features alongside the N=1e9 Monte-Carlo ground truth.

Usage:
    uv run python scripts/extract_features.py --data <dataset_dir> --split mini --out features_mini.npz
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    from scipy.special import ndtr

    return ndtr(x)


def mean_prop(Ws):
    """K=1: diagonal mean/variance propagation. Returns final (mu, var)."""
    n = Ws[0].shape[0]
    mu = np.zeros(n)
    var = np.ones(n)
    for w in Ws:
        mu_pre = w.T @ mu
        var_pre = np.maximum((w * w).T @ var, 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        phi, Phi = norm_pdf(alpha), norm_cdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
    return mu, var


def relu_cov_exact(W1):
    """Exact mean/cov of ReLU(u @ W1) for u ~ N(0, I): z1 is exactly Gaussian,
    so the bivariate Gaussian ReLU expectation applies in closed form."""
    G = W1.T @ W1
    sigma = np.sqrt(np.maximum(np.diagonal(G), 1e-24))
    rho = np.clip(G / np.outer(sigma, sigma), -1.0, 1.0)
    m = sigma / math.sqrt(2 * math.pi)
    second = (np.outer(sigma, sigma) / (2 * math.pi)) * (
        np.sqrt(np.maximum(1 - rho * rho, 0.0)) + rho * (math.pi / 2 + np.arcsin(rho))
    )
    cov = second - np.outer(m, m)
    return m, cov


def cov_prop(Ws, collect_last_k: int = 2, collect_targets_k: int = 4):
    """K=2 covariance propagation: exact bivariate formula at layer 1,
    gain method for layers >= 2.

    Returns (final_mean, final-layer artifacts dict). The dict also carries
    `targets`: per-layer (mean, var_diagonal) for the first
    `collect_targets_k` layers plus the full layer-1 covariance -- consumed
    by the moment-matched Monte Carlo sampler.
    """
    n = Ws[0].shape[0]
    means_hist = []
    final = {}
    targets = []
    L = len(Ws)
    mu, cov = relu_cov_exact(Ws[0])
    cov_l1 = cov.copy()
    targets.append((mu.copy(), np.diagonal(cov).copy()))
    if L - collect_last_k <= 0:
        means_hist.append(mu.copy())
    for li, w in enumerate(Ws[1:], start=1):
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w, optimize=True)
        var_pre = np.maximum(np.diagonal(cov_pre), 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi, Phi = norm_pdf(alpha), norm_cdf(alpha)
        mu = mu_pre * Phi + sigma_pre * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma_pre * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        if li < collect_targets_k:
            targets.append((mu.copy(), var_post.copy()))
        if li >= L - collect_last_k:
            means_hist.append(mu.copy())
        if li == L - 1:
            offdiag_strength = (np.abs(cov).sum(axis=1) - np.abs(np.diagonal(cov)))
            final = dict(
                mu_pre=mu_pre, sigma_pre=sigma_pre, alpha=alpha,
                phi=phi, Phi=Phi, var_post=var_post,
                offdiag_strength=offdiag_strength,
            )
    final["means_hist"] = means_hist
    final["targets"] = targets
    final["cov_l1"] = cov_l1
    return mu, final


def _mat_sqrt(C, inv=False):
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-12)
    d = 1.0 / np.sqrt(evals) if inv else np.sqrt(evals)
    return (evecs * d) @ evecs.T


def antithetic_mc(Ws, n_pairs, seed, mm_targets=None, cov_l1=None):
    """Moment-matched antithetic Monte Carlo ("fullL1_diag234").

    Antithetic pairs (u, -u) kill all odd-order noise exactly. Then the batch
    is affinely renormalized at layer 1 so its empirical mean and FULL
    covariance exactly equal the analytic ones (z1 is exactly Gaussian, so
    both are closed-form -- no bias beyond O(1/N) from the data-dependent
    affine map). At layers 2..4 the batch's mean and per-neuron variance are
    pinned (diagonal match) to the K=2 mech trajectory; that bias is
    systematic and corrector-learnable, while the retained samples carry the
    true higher-moment structure.

    mm_targets: list of (mean, var_diag) for layers 1..k from cov_prop.
    cov_l1: exact full layer-1 covariance from relu_cov_exact.

    Returns (mc_mean, mc_sem) per final-layer neuron.
    """
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)).astype(np.float64)
    x = np.concatenate([u, -u], axis=0)
    m_t = mm_targets[0][0] if mm_targets else None
    for li, w in enumerate(Ws):
        x = np.maximum(x @ w, 0.0)
        if li == 0 and cov_l1 is not None and m_t is not None:
            # exact layer-1 moment matching only; deeper pinning measured to
            # hurt over 200 MLPs (mech bias amplifies through depth)
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            cov_emp = (xc.T @ xc) / len(x)
            A = _mat_sqrt(cov_emp, inv=True) @ _mat_sqrt(cov_l1)
            x = xc @ A + m_t
    mc_mean = x.mean(axis=0)
    mc_sem = x.std(axis=0) / np.sqrt(x.shape[0])
    return mc_mean, mc_sem


def features_for_mlp(Ws, mlp_seed, n_mc_pairs=2750, collect_last_k=5):
    """Assemble the per-neuron feature matrix (n, F) for the final layer."""
    mu1, var1 = mean_prop(Ws)
    mu2, fin = cov_prop(Ws, collect_last_k=collect_last_k)
    wL = Ws[-1]
    col_norm = np.sqrt((wL * wL).sum(axis=0))
    hist = fin["means_hist"]  # oldest..newest, newest == mu2
    # pad history if net shallower than collect_last_k
    while len(hist) < collect_last_k:
        hist = [hist[0]] + hist

    mc_mean, mc_sem = antithetic_mc(
        Ws, n_mc_pairs, seed=mlp_seed,
        mm_targets=fin["targets"], cov_l1=fin["cov_l1"],
    )

    cols = [
        mu2,                       # 0 K=2 prediction (base estimate)
        mu1,                       # 1 K=1 prediction
        mu2 - mu1,                 # 2 K1->K2 step (extrapolation signal)
        fin["mu_pre"],             # 3
        fin["sigma_pre"],          # 4
        fin["alpha"],              # 5
        fin["Phi"],                # 6
        fin["phi"],                # 7
        np.sqrt(fin["var_post"]),  # 8
        fin["offdiag_strength"],   # 9
        col_norm,                  # 10 final-layer column norm
        mc_mean,                   # 11 antithetic MC estimate (unbiased)
        mc_mean - mu2,             # 12 MC-vs-mech disagreement
        mc_sem,                    # 13 MC standard error per neuron
        np.full_like(mu2, mu2.mean()),  # 14 layer mean level
        np.full_like(mu2, np.sqrt(fin["var_post"]).mean()),  # 15 avg sigma
    ]
    # multi-layer mean trajectory (last collect_last_k layers, incl. final)
    for h in hist[:-1]:
        cols.append(h)             # 16.. trajectory means
    feats = np.stack(cols, axis=1)
    return feats, mu2, mc_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    import whestbench.dataset as wds

    d = wds.load_dataset(args.data, split=args.split)
    n_mlps = len(d) if not args.limit else min(args.limit, len(d))

    all_feats, all_base, all_mc, all_truth, all_ids = [], [], [], [], []
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        feats, mu2, mc_mean = features_for_mlp(Ws, mlp_seed=row["mlp_seed"] + args.seed_offset)
        all_feats.append(feats)
        all_base.append(mu2)
        all_mc.append(mc_mean)
        all_truth.append(truth)
        all_ids.append(np.full(len(truth), row["mlp_id"]))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{n_mlps} MLPs done", flush=True)

    np.savez_compressed(
        args.out,
        feats=np.concatenate(all_feats),      # (n_mlps*width, F)
        base=np.concatenate(all_base),        # K=2 baseline prediction
        mc=np.concatenate(all_mc),            # antithetic MC estimate
        truth=np.concatenate(all_truth),
        mlp_id=np.concatenate(all_ids),
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
