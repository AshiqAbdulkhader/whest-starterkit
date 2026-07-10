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


def cov_prop(Ws, collect_last_k: int = 2):
    """K=2: full covariance propagation (gain method).

    Returns dict with final-layer per-neuron quantities and the mean vectors
    of the last `collect_last_k` layers (for trend features).
    """
    n = Ws[0].shape[0]
    mu = np.zeros(n)
    cov = np.eye(n)
    means_hist = []
    final = {}
    L = len(Ws)
    for li, w in enumerate(Ws):
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
        if li >= L - collect_last_k:
            means_hist.append(mu.copy())
        if li == L - 1:
            # row-sum magnitude of off-diagonal covariance: how non-diagonal
            # the joint distribution is at this neuron
            offdiag_strength = (np.abs(cov).sum(axis=1) - np.abs(np.diagonal(cov)))
            final = dict(
                mu_pre=mu_pre, sigma_pre=sigma_pre, alpha=alpha,
                phi=phi, Phi=Phi, var_post=var_post,
                offdiag_strength=offdiag_strength,
            )
    final["means_hist"] = means_hist
    return mu, final


def antithetic_mc(Ws, n_pairs, seed, whiten=True):
    """Whitened antithetic Monte Carlo through the MLP.

    Antithetic pairs (u, -u) kill all odd-order noise components exactly;
    whitening (folding C^{-1/2} of the empirical input covariance into the
    first weight matrix so the sample covariance is exactly identity) kills
    all quadratic noise components. Residual noise is higher-order only.

    Returns (mc_mean, mc_sem) per final-layer neuron. Exact draw equality
    with the grader is not required -- the combiner only relies on the
    statistical relationship (near-unbiased estimate with ~Var/N noise).
    """
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)).astype(np.float64)
    if whiten:
        # empirical covariance of the antithetic batch [u; -u] is u^T u * 2/N
        C = (u.T @ u) / n_pairs
        evals, evecs = np.linalg.eigh(C)
        C_inv_sqrt = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
        W1 = C_inv_sqrt @ Ws[0]  # fold whitening into the first layer
    else:
        W1 = Ws[0]
    x = np.concatenate([u, -u], axis=0)
    x = np.maximum(x @ W1, 0.0)
    for w in Ws[1:]:
        x = np.maximum(x @ w, 0.0)
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

    mc_mean, mc_sem = antithetic_mc(Ws, n_mc_pairs, seed=mlp_seed)

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
    args = ap.parse_args()

    import whestbench.dataset as wds

    d = wds.load_dataset(args.data, split=args.split)
    n_mlps = len(d) if not args.limit else min(args.limit, len(d))

    all_feats, all_base, all_mc, all_truth, all_ids = [], [], [], [], []
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        feats, mu2, mc_mean = features_for_mlp(Ws, mlp_seed=row["mlp_seed"])
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
