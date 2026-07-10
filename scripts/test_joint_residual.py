"""Joint residual denoising: exploit cross-neuron structure in (truth - mmL1).

pscamillo found K=2 residual effective rank ~29. If mmL1 residual is also
low-rank across neurons, a joint (vector-valued) corrector or PCA shrinkage
can beat our per-neuron 1.2x corrector.

Tests on features_full_v4:
  1) Empirical residual covariance rank spectrum
  2) Per-MLP PCA truncate of (mc - k2) toward k2 — diagnostic only
  3) Train joint linear map: residual_vec = W @ feats_summary (low-rank)
  4) Compare to per-neuron ridge on same holdout MLPs
"""

from __future__ import annotations

import numpy as np


def main():
    data = np.load(r"C:\Users\MUKHADE\Workspace\whest-data\features_full_v4.npz")
    feats = data["feats"].astype(np.float64)  # (N*256, F)
    mc = data["mc"].astype(np.float64)
    k2 = data["base"].astype(np.float64)
    truth = data["truth"].astype(np.float64)
    mlp_id = data["mlp_id"]

    ids = np.unique(mlp_id)
    width = 256
    # reshape to (n_mlps, width)
    # assume contiguous blocks of width per mlp
    n_mlps = len(ids)
    assert len(mc) == n_mlps * width

    # reorder by sorted unique appearance order
    order = []
    for i in ids:
        order.append(np.where(mlp_id == i)[0])
    idx = np.concatenate(order)
    mc_m = mc[idx].reshape(n_mlps, width)
    k2_m = k2[idx].reshape(n_mlps, width)
    truth_m = truth[idx].reshape(n_mlps, width)
    resid = truth_m - mc_m  # (n_mlps, width)

    # 1) residual covariance spectrum
    r0 = resid - resid.mean(0)
    cov = (r0.T @ r0) / n_mlps
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = evals.sum()
    cum = np.cumsum(evals) / total
    print("Residual (truth-mc) covariance spectrum:")
    for k in [1, 5, 10, 20, 29, 50, 100]:
        print(f"  top-{k}: {100*cum[k-1]:.1f}% energy")

    # holdout split by MLP
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_mlps)
    n_va = max(1, int(0.15 * n_mlps))
    va, tr = perm[:n_va], perm[n_va:]

    mse_mc = np.mean((mc_m[va] - truth_m[va]) ** 2)
    print(f"\nholdout mmL1 MSE: {mse_mc:.4e}")

    # 2) per-neuron ridge (reproduce current corrector scale)
    # features: use mc, k2, mc-k2 per neuron
    X = np.stack([mc, k2, mc - k2], axis=1)  # (N*w, 3)
    y = truth - mc
    tr_mask = np.isin(mlp_id, ids[tr])
    va_mask = np.isin(mlp_id, ids[va])
    mu = X[tr_mask].mean(0)
    sd = X[tr_mask].std(0) + 1e-12
    Xn = (X - mu) / sd
    A = Xn[tr_mask]
    coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(3), A.T @ y[tr_mask])
    pred = mc[va_mask] + Xn[va_mask] @ coef
    mse_ridge = np.mean((pred - truth[va_mask]) ** 2)
    print(f"per-neuron ridge (3 feats): {mse_ridge:.4e} ({mse_mc/mse_ridge:.2f}x)")

    # 3) joint low-rank: residual ≈ sum_{k=1}^r (a_k · global_feats) * v_k
    # Estimate top-r residual PCs on train, regress PC scores from MLP-level features
    Rtr = resid[tr]
    Rtr0 = Rtr - Rtr.mean(0)
    # SVD of residuals
    U, S, Vt = np.linalg.svd(Rtr0, full_matrices=False)
    for r in [4, 8, 16, 29, 50]:
        V = Vt[:r].T  # (width, r) principal directions
        # scores on train: (n_tr, r)
        scores = Rtr0 @ V
        # MLP-level features: mean/std of mc, k2, and top singular-ish stats
        def mlp_feats(M_mc, M_k2):
            return np.stack([
                M_mc.mean(1), M_mc.std(1),
                M_k2.mean(1), M_k2.std(1),
                (M_mc - M_k2).mean(1), (M_mc - M_k2).std(1),
                np.linalg.norm(M_mc, axis=1),
                np.linalg.norm(M_k2, axis=1),
            ], axis=1)  # (n, 8)

        Ftr = mlp_feats(mc_m[tr], k2_m[tr])
        Fva = mlp_feats(mc_m[va], k2_m[va])
        # standardize
        fmu, fsd = Ftr.mean(0), Ftr.std(0) + 1e-12
        Ftr_n = (Ftr - fmu) / fsd
        Fva_n = (Fva - fmu) / fsd
        # ridge: scores ≈ F @ B
        B = np.linalg.solve(Ftr_n.T @ Ftr_n + 1e-2 * np.eye(Ftr_n.shape[1]), Ftr_n.T @ scores)
        scores_hat = Fva_n @ B
        resid_hat = scores_hat @ V.T + Rtr.mean(0)
        pred = mc_m[va] + resid_hat
        mse = np.mean((pred - truth_m[va]) ** 2)
        print(f"joint PC-r{r} from MLP globals: {mse:.4e} ({mse_mc/mse:.2f}x)")

    # 4) stronger joint: per-neuron feats -> PC scores via pooling
    # For each MLP, average the 20-d features, predict r scores
    Ffull = feats[idx].reshape(n_mlps, width, -1)  # (n, w, F)
    Fmlp = Ffull.mean(1)  # (n, F)
    Ftr = Fmlp[tr]
    Fva = Fmlp[va]
    fmu, fsd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ftr_n = (Ftr - fmu) / fsd
    Fva_n = (Fva - fmu) / fsd
    for r in [8, 16, 29]:
        V = Vt[:r].T
        scores = Rtr0 @ V
        B = np.linalg.solve(Ftr_n.T @ Ftr_n + 1e-2 * np.eye(Ftr_n.shape[1]), Ftr_n.T @ scores)
        resid_hat = (Fva_n @ B) @ V.T + Rtr.mean(0)
        pred = mc_m[va] + resid_hat
        mse = np.mean((pred - truth_m[va]) ** 2)
        print(f"joint PC-r{r} from mean-pooled 20 feats: {mse:.4e} ({mse_mc/mse:.2f}x)")

    # 5) ORACLE joint: project holdout residual onto train PCs (ceiling)
    for r in [8, 16, 29, 50, 100]:
        V = Vt[:r].T
        Rva = resid[va] - Rtr.mean(0)
        proj = (Rva @ V) @ V.T + Rtr.mean(0)
        pred = mc_m[va] + proj
        # this uses oracle residual — only a ceiling on PC denoising
        # actually projecting the TRUE residual onto PCs measures how much
        # energy lives in those PCs (should match spectrum)
        mse = np.mean((pred - truth_m[va]) ** 2)
        print(f"ORACLE PC-r{r} projection ceiling: {mse:.4e} ({mse_mc/mse:.2f}x)")


if __name__ == "__main__":
    main()
