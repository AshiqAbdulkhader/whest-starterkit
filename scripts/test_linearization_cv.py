"""Test: per-neuron linearization control variate from h1 -> hL.

Why this is different from failed h1_cv (256-basis regression) and from
mmL1-incompatible linear CVs:

1. One scalar control per output neuron: c_j = (A @ h1)_j, where A is the
   gain-linearized map h1 -> hL from the K=2 trajectory. Cross-fitted beta
   has tiny estimation noise (1 param, not 256).
2. E[c] = A @ E[h1] is EXACT (layer-1 closed form) -- real CV, not shrinkage.
3. Must NOT combine with mmL1: after mmL1, mean(h1)=E[h1] as a batch
   constraint, so mean(A@(h1-E[h1]))=0 identically and the CV is a no-op
   (same lesson as telescoping). Compare against antithetic-only and mmL1.

Also tests: whitened + lin-CV, and whether beta=1 (pure delta method) works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop, norm_cdf, relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def build_linearized_map(Ws):
    """Gain-linearized map A such that hL ≈ A @ h1 (plus bias we drop for CV).

    Propagates the diagonal gain Phi(alpha) from K=2 at each layer:
      h_{l+1} ≈ diag(gain_{l+1}) @ W_{l+1}.T @ h_l
    so A = G_L W_L.T G_{L-1} W_{L-1}.T ... G_2 W_2.T
    (h1 already post-ReLU; first map is layer-2 weights).
    """
    _, fin = cov_prop(Ws)
    # Recompute gains layer by layer (cov_prop doesn't return them all)
    n = Ws[0].shape[0]
    mu, cov = relu_cov_exact(Ws[0])
    # A starts as identity on h1
    A = np.eye(n)
    for w in Ws[1:]:
        mu_pre = w.T @ mu
        cov_pre = np.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        Phi = norm_cdf(alpha)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        # h_next ≈ diag(gain) @ W.T @ h_prev  (W is (in,out), forward is h@W)
        # In row-vector form: h_next = h_prev @ W @ diag(gain)
        A = A @ w * gain  # (n,n) @ (n,n) then broadcast-scale columns
        # update mu, cov for next layer (same as cov_prop)
        from extract_features import norm_pdf

        phi = norm_pdf(alpha)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var_post = np.maximum(ez2 - mu * mu, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
    return A, fin


def forward_collect(Ws, seed, n_pairs, mode="antithetic"):
    """Return (h1, hL) for antithetic / whitened / mmL1 modes."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_l1 = relu_cov_exact(Ws[0])

    if mode == "whitened":
        # fold C^{-1/2} into first layer: whiten empirical input cov
        xc = x - x.mean(axis=0)
        C = (xc.T @ xc) / len(x)
        W1 = mat_sqrt(C, inv=True) @ Ws[0]
        h = np.maximum(x @ W1, 0.0)
    else:
        h = np.maximum(x @ Ws[0], 0.0)
        if mode == "mmL1":
            h = affine_match(h, m_t, cov_l1)

    h1 = h.copy()
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h1, h, m_t


def crossfit_cv(hL, control, e_control, n_folds=2):
    """Per-neuron scalar beta, cross-fitted. Returns CV estimate (n,)."""
    N, n = hL.shape
    idx = np.arange(N)
    # simple 2-fold split preserving antithetic pairs: split by pair index
    n_pairs = N // 2
    pair_idx = np.arange(n_pairs)
    mid = n_pairs // 2
    est = np.zeros(n)
    for fold, (tr, te) in enumerate(
        [(pair_idx[:mid], pair_idx[mid:]), (pair_idx[mid:], pair_idx[:mid])]
    ):
        # expand pairs to sample indices (u and -u)
        tr_s = np.concatenate([tr, tr + n_pairs])
        te_s = np.concatenate([te, te + n_pairs])
        # beta_j = Cov(hL_j, c_j) / Var(c_j)
        c_tr = control[tr_s]
        y_tr = hL[tr_s]
        c0 = c_tr - c_tr.mean(axis=0)
        y0 = y_tr - y_tr.mean(axis=0)
        var_c = (c0**2).mean(axis=0) + 1e-24
        cov_yc = (y0 * c0).mean(axis=0)
        beta = cov_yc / var_c
        # apply on te
        est += (hL[te_s] - beta * (control[te_s] - e_control)).mean(axis=0) * (
            len(te_s) / N
        )
    # above weighting is approximate; cleaner: accumulate then done
    # Actually redo more cleanly:
    est = np.zeros(n)
    counts = 0
    for tr, te in [(pair_idx[:mid], pair_idx[mid:]), (pair_idx[mid:], pair_idx[:mid])]:
        tr_s = np.concatenate([tr, tr + n_pairs])
        te_s = np.concatenate([te, te + n_pairs])
        c_tr = control[tr_s]
        y_tr = hL[tr_s]
        c0 = c_tr - c_tr.mean(axis=0)
        y0 = y_tr - y_tr.mean(axis=0)
        beta = (y0 * c0).mean(axis=0) / ((c0**2).mean(axis=0) + 1e-24)
        fold_est = (hL[te_s] - beta * (control[te_s] - e_control)).mean(axis=0)
        est += fold_est
        counts += 1
    return est / counts


def beta1_cv(hL, control, e_control):
    return (hL - (control - e_control)).mean(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 50
    n_pairs = 2750

    methods = [
        "mmL1",
        "antithetic",
        "whitened",
        "anti+linCV_beta1",
        "anti+linCV_xfit",
        "whitened+linCV_xfit",
        "mmL1+linCV_xfit",  # expected no-op / same as mmL1
    ]
    errors = {m: [] for m in methods}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        A, _ = build_linearized_map(Ws)

        # mmL1 baseline
        h1, hL, m_t = forward_collect(Ws, row["mlp_seed"], n_pairs, "mmL1")
        errors["mmL1"].append(np.mean((hL.mean(axis=0) - truth) ** 2))
        # mmL1 + lin CV (should be ~no-op)
        control = h1 @ A
        e_c = m_t @ A
        est = crossfit_cv(hL, control, e_c)
        errors["mmL1+linCV_xfit"].append(np.mean((est - truth) ** 2))

        # antithetic only
        h1, hL, m_t = forward_collect(Ws, row["mlp_seed"], n_pairs, "antithetic")
        errors["antithetic"].append(np.mean((hL.mean(axis=0) - truth) ** 2))
        control = h1 @ A
        e_c = m_t @ A
        errors["anti+linCV_beta1"].append(
            np.mean((beta1_cv(hL, control, e_c) - truth) ** 2)
        )
        errors["anti+linCV_xfit"].append(
            np.mean((crossfit_cv(hL, control, e_c) - truth) ** 2)
        )

        # whitened
        h1, hL, m_t = forward_collect(Ws, row["mlp_seed"], n_pairs, "whitened")
        errors["whitened"].append(np.mean((hL.mean(axis=0) - truth) ** 2))
        control = h1 @ A
        e_c = m_t @ A
        errors["whitened+linCV_xfit"].append(
            np.mean((crossfit_cv(hL, control, e_c) - truth) ** 2)
        )

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}")

    print(f"\nPaired comparison over {n_mlps} full-split MLPs, n_pairs={n_pairs}:")
    base = np.array(errors["mmL1"])
    print(f"{'method':<24} {'mean MSE':>12} {'vs mmL1':>10}")
    for m in methods:
        arr = np.array(errors[m])
        ratio = base.mean() / arr.mean()
        print(f"{m:<24} {arr.mean():>12.4e} {ratio:>9.3f}x")


if __name__ == "__main__":
    main()
