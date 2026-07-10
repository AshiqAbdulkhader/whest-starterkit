"""Low-rank regression CV: hL ~ B @ h1, with E[h1] known exactly.

Unlike 256-basis h1_cv (K/N noise), use rank-r PLS/ridge on the
cross-covariance Cov(h1, hL). E[B @ h1] = B @ E[h1] is exact.
Must run WITHOUT mmL1 (mmL1 pins mean(h1)=E[h1) → correction is 0).
Compare antithetic+CV vs mmL1 vs whitened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward(Ws, seed, n_pairs, mode):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_l1 = relu_cov_exact(Ws[0])
    if mode == "whitened":
        cov = (u.T @ u) / n_pairs
        evals, evecs = np.linalg.eigh(cov)
        inv_sqrt = (evecs / np.sqrt(np.maximum(evals, 1e-6))) @ evecs.T
        h = np.maximum(x @ (inv_sqrt @ Ws[0]), 0.0)
    else:
        h = np.maximum(x @ Ws[0], 0.0)
        if mode == "mmL1":
            h = affine_match(h, m_t, cov_l1)
    h1 = h.copy()
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h1, h, m_t


def lowrank_cv(h1, hL, m_t, rank, n_folds=2):
    """Cross-fitted rank-r regression CV."""
    N, n = h1.shape
    n_pairs = N // 2
    mid = n_pairs // 2
    est = np.zeros(n)
    for tr_p, te_p in [(np.arange(mid), np.arange(mid, n_pairs)),
                       (np.arange(mid, n_pairs), np.arange(mid))]:
        tr = np.concatenate([tr_p, tr_p + n_pairs])
        te = np.concatenate([te_p, te_p + n_pairs])
        X = h1[tr] - h1[tr].mean(0)
        Y = hL[tr] - hL[tr].mean(0)
        # cross-cov C = X.T @ Y / N  -> (n, n); top-r via SVD of C
        C = (X.T @ Y) / len(tr)
        # economy SVD
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        r = min(rank, len(S))
        # B such that Y ≈ X @ B, B = U_r @ diag(S_r) @ Vt_r  / something
        # Actually C = Cov(x,y); beta for y = x @ B: B = Cov(x,x)^{-1} C
        # ridge-stabilize:
        Cx = (X.T @ X) / len(tr) + 1e-4 * np.eye(n)
        # low-rank approx of C
        C_r = (U[:, :r] * S[:r]) @ Vt[:r]
        # solve Cx @ B = C_r
        B = np.linalg.solve(Cx, C_r)
        # apply on te: mean(hL - h1@B) + m_t@B
        pred_c = h1[te] @ B
        e_c = m_t @ B
        fold_est = (hL[te] - pred_c).mean(0) + e_c
        est += fold_est
    return est / n_folds


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 40
    n_pairs = 2750
    ranks = [4, 8, 16, 32]

    methods = ["mmL1", "antithetic", "whitened"] + [f"anti+cv_r{r}" for r in ranks] + [
        f"wh+cv_r{r}" for r in ranks
    ]
    errs = {m: [] for m in methods}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)

        h1, hL, m_t = forward(Ws, row["mlp_seed"], n_pairs, "mmL1")
        errs["mmL1"].append(np.mean((hL.mean(0) - truth) ** 2))

        h1, hL, m_t = forward(Ws, row["mlp_seed"], n_pairs, "antithetic")
        errs["antithetic"].append(np.mean((hL.mean(0) - truth) ** 2))
        for r in ranks:
            est = lowrank_cv(h1, hL, m_t, r)
            errs[f"anti+cv_r{r}"].append(np.mean((est - truth) ** 2))

        h1, hL, m_t = forward(Ws, row["mlp_seed"], n_pairs, "whitened")
        errs["whitened"].append(np.mean((hL.mean(0) - truth) ** 2))
        for r in ranks:
            est = lowrank_cv(h1, hL, m_t, r)
            errs[f"wh+cv_r{r}"].append(np.mean((est - truth) ** 2))

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}", flush=True)

    base = np.mean(errs["mmL1"])
    print(f"\n{'method':<16} {'MSE':>12} {'vs mmL1':>10}")
    for m in methods:
        arr = np.array(errs[m])
        print(f"{m:<16} {arr.mean():>12.4e} {base/arr.mean():>9.3f}x")


if __name__ == "__main__":
    main()
