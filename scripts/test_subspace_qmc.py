"""QMC on the effective 8-D subspace of the linearized h1->hL map.

Discovery: A(K1) from h1 to hL has 99.7% energy in top-8 singular vectors.
Prior full-256 Sobol failed because effective dimension looked high; it isn't
for the dominant linear path. Strategy:

  1. Build linearized A (K1 gains, cheap).
  2. Take top-r RIGHT singular vectors V_r of A (directions in h1-space).
  3. Push those back to INPUT space via W1: dirs ~ W1 @ V_r, orthonormalize.
  4. Sample the r-D subspace with scrambled Sobol (power-of-2), remaining
     dims with antithetic Gaussian; combine, run mmL1 forward.
  5. Compare paired vs plain mmL1 at matched total sample count.

Also tests: stratification (not Sobol) on the same subspace as a scipy-free
fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact, norm_cdf, norm_pdf  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def build_A_k1(Ws):
    n = Ws[0].shape[0]
    mu = np.zeros(n)
    var = np.ones(n)
    A = np.eye(n)
    for w in Ws:
        mu_pre = w.T @ mu
        var_pre = np.maximum((w * w).T @ var, 1e-12)
        sigma = np.sqrt(var_pre)
        alpha = mu_pre / sigma
        Phi = norm_cdf(alpha)
        phi = norm_pdf(alpha)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        A = A @ w * gain
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
    return A


def input_subspace(Ws, r):
    """Orthonormal input-space basis for top-r h1-directions of A."""
    A = build_A_k1(Ws)
    # A maps h1 -> hL; right singular vectors are h1-directions
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:r].T  # (n, r) in h1-space
    # h1 ≈ ReLU(x @ W0); for small signals linearize as x @ W0 * gain0
    # push V back: want input dirs D such that (D @ W0) spans V
    # least squares: W0.T @ D.T ≈ V  => D.T ≈ pinv(W0.T) @ V = W0 @ pinv(W0.T@W0) @ V
    # simpler: D = V directions in input via W0 @ V (if W0 square invertible-ish)
    W0 = Ws[0]
    D = W0 @ V  # (n, r) — input directions that excite V in pre-activations
    # orthonormalize
    Q, _ = np.linalg.qr(D)
    return Q[:, :r]


def sample_hybrid(Ws, seed, n_pairs, r, method="sobol"):
    """Hybrid: QMC/strat on r-D subspace + antithetic Gaussian on complement."""
    n = Ws[0].shape[0]
    Q = input_subspace(Ws, r)  # (n, r)
    # complement projector
    # sample in Q-coords and orthogonal coords
    n_tot = 2 * n_pairs
    # use power-of-2 for sobol balance
    n_q = 1 << int(np.ceil(np.log2(n_tot)))
    rng = np.random.default_rng(seed)

    from scipy.stats import norm

    if method == "sobol":
        from scipy.stats import qmc
        eng = qmc.Sobol(d=r, scramble=True, seed=int(seed) % (2**31 - 1))
        u = np.clip(eng.random(n_q), 1e-10, 1 - 1e-10)
        z_sub = norm.ppf(u)[:n_tot]
    elif method == "strat":
        z_sub = np.zeros((n_tot, r))
        for j in range(r):
            centers = (np.arange(n_tot) + rng.random(n_tot)) / n_tot
            rng.shuffle(centers)
            z_sub[:, j] = norm.ppf(np.clip(centers, 1e-10, 1 - 1e-10))
    else:
        z_sub = rng.standard_normal((n_tot, r))

    # orthogonal complement: antithetic Gaussian in full space, then remove Q component
    m = n_tot // 2
    u_full = rng.standard_normal((m, n))
    x_orth = np.concatenate([u_full, -u_full], axis=0)
    # remove projection onto Q
    x_orth = x_orth - (x_orth @ Q) @ Q.T

    x = z_sub @ Q.T + x_orth
    # mmL1 forward
    m_t, cov_t = relu_cov_exact(Ws[0])
    h = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h, m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(0), n_tot


def mmL1(Ws, seed, n_pairs):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_t = relu_cov_exact(Ws[0])
    h = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h, m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 30
    # match sample counts carefully: sobol rounds up to pow2
    # use n_pairs such that 2*n_pairs is power of 2: 2048 samples = 1024 pairs
    n_pairs = 1024
    ranks = [4, 8, 16]

    errs = {"mmL1": []}
    for r in ranks:
        errs[f"sobol_r{r}"] = []
        errs[f"strat_r{r}"] = []

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        # mmL1 at matched N: sobol uses n_q = next pow2 of 2*n_pairs = 2048
        pred = mmL1(Ws, row["mlp_seed"], n_pairs)
        errs["mmL1"].append(np.mean((pred - truth) ** 2))
        for r in ranks:
            for method, key in [("sobol", f"sobol_r{r}"), ("strat", f"strat_r{r}")]:
                pred, n_used = sample_hybrid(Ws, row["mlp_seed"], n_pairs, r, method)
                # if sobol used more samples, also run mmL1 at that count for fair...
                # we forced n_q=2048=2*1024 so matched
                errs[key].append(np.mean((pred - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}", flush=True)

    base = np.mean(errs["mmL1"])
    print(f"\nMatched N={2*n_pairs} samples, {n_mlps} MLPs:")
    print(f"{'method':<12} {'MSE':>12} {'vs mmL1':>10}")
    for k, v in errs.items():
        m = np.mean(v)
        print(f"{k:<12} {m:>12.4e} {base/m:>9.3f}x")


if __name__ == "__main__":
    main()
