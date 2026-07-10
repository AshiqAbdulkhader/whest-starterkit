"""Test: exact zero-mean quartic Hermite control variates (idea sourced from
a Codex consult follow-up -- see EXPERIMENTS.md research round 7).

Different in kind from the earlier failed scripts/test_h1_cv.py: that used a
LARGE basis (all 256 post-ReLU h1 coordinates, degree-1 linear features) and
got eaten by K/N cross-fitted regression noise (K=256 vs N~5500, ~5%
relative noise). This uses a SMALL number (M=8..48) of exact-zero-mean
FOURTH-ORDER Hermite polynomials of a few chosen INPUT-space projections --
targeting the noise order that survives antithetic pairing (kills odd
orders) and exact layer-1 mean/covariance matching (kills up to 2nd order
exactly), i.e. even-order noise beyond the 2nd moment.

For a fixed direction a_m (unit or arbitrary norm) in INPUT space, t_m = u.a_m
is EXACTLY N(0, s_m^2) for u ~ N(0,I) regardless of a_m (linear combination of
iid Gaussians). The standardized 4th probabilist's Hermite polynomial
He_4(t_m/s_m) = (t_m/s_m)^4 - 6(t_m/s_m)^2 + 3 has EXACT zero mean under this
distribution (by Hermite orthogonality to the constant function) -- a valid,
free (no extra sampling) control variate, no matter how deep/nonlinear the
rest of the network is, since it's a fact about the INPUT distribution only.

  q_m = He_4(t_m/s_m) / sqrt(24)   (unit-variance normalization)

Cross-fitted (split-batch) linear regression of the final-layer output onto
q (to avoid in-sample overfitting bias) gives an unbiased, variance-reduced
estimator: est_j = mean(y_j) - beta[:,j] . mean(q).

Direction choices tested: random orthonormal, top singular vectors of W1,
output-aware sensitivity directions (top singular vectors of the full
mech Jacobian-like sensitivity from input to output), and mixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def mmL1_forward(Ws, seed, m_t, cov_t, n_pairs=2750, return_u=False):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u_half = rng.standard_normal((n_pairs, n))
    u = np.concatenate([u_half, -u_half], axis=0)
    h1 = np.maximum(u @ Ws[0], 0.0)
    h = affine_match(h1, m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    if return_u:
        return h, u
    return h


def quartic_cv_basis(u, directions):
    """u: (N, n) input samples. directions: (n, M). Returns (N, M) exact
    zero-mean quartic Hermite control variates."""
    t = u @ directions                      # (N, M)
    s2 = np.sum(directions * directions, axis=0)  # (M,) = ||a_m||^2
    s = np.sqrt(s2)
    x = t / s[None, :]
    He4 = x**4 - 6 * x**2 + 3
    return He4 / np.sqrt(24.0)


def cross_fitted_cv(y, q, lam=1e-6):
    """y: (N, n_out), q: (N, M) exact zero-mean. Cross-fitted CV correction."""
    N = len(y)
    half = N // 2
    idx1, idx2 = np.arange(half), np.arange(half, N)
    est = np.zeros(y.shape[1])
    for tr, te in ((idx1, idx2), (idx2, idx1)):
        qt, yt = q[tr], y[tr]
        G = qt.T @ qt / len(tr) + lam * np.eye(q.shape[1])
        beta = np.linalg.solve(G, qt.T @ yt / len(tr))  # (M, n_out)
        est += y[te].mean(axis=0) - q[te].mean(axis=0) @ beta
    return est / 2


def make_directions(Ws, n, M, kind, rng):
    W1 = Ws[0]
    if kind == "random":
        A = rng.standard_normal((n, M))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        return A
    if kind == "w1_svd":
        U, S, Vt = np.linalg.svd(W1, full_matrices=False)
        return U[:, :M]  # top-M left singular vectors of W1 (input-space directions)
    if kind == "output_aware":
        # crude output-sensitivity: propagate an approximate linear gain
        # through the mech (K=1-style) trajectory, using diag(gain) at each
        # layer as a cheap proxy for the true Jacobian.
        from extract_features import norm_cdf, norm_pdf

        mu = np.zeros(n)
        var = np.ones(n)
        J = np.eye(n)  # accumulate an approximate linear sensitivity input->layer
        for w in Ws:
            mu_pre = w.T @ mu
            var_pre = np.maximum((w * w).T @ var, 1e-12)
            sigma = np.sqrt(var_pre)
            alpha = mu_pre / sigma
            Phi, phi = norm_cdf(alpha), norm_pdf(alpha)
            gain = Phi
            J = J @ w * gain[None, :]
            mu = mu_pre * Phi + sigma * phi
            ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
            var = np.maximum(ez2 - mu * mu, 0.0)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        return U[:, :M]
    raise ValueError(kind)


def run(Ws, seed, m_t, cov_t, kind, M):
    n = Ws[0].shape[0]
    h, u = mmL1_forward(Ws, seed, m_t, cov_t, return_u=True)
    rng = np.random.default_rng(seed + 999999)
    directions = make_directions(Ws, n, M, kind, rng)
    q = quartic_cv_basis(u, directions)
    return cross_fitted_cv(h, q)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 100
    configs = [("random", M) for M in (8, 16, 32)] + \
              [("w1_svd", M) for M in (8, 16, 32)] + \
              [("output_aware", M) for M in (8, 16, 32)]
    results = {"raw": []}
    for kind, M in configs:
        results[f"{kind}_M{M}"] = []

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]

        h, u = mmL1_forward(Ws, row["mlp_seed"], m_t, cov_t, return_u=True)
        raw = h.mean(axis=0)
        results["raw"].append(np.mean((raw - truth) ** 2))

        for kind, M in configs:
            rng = np.random.default_rng(row["mlp_seed"] + 999999)
            directions = make_directions(Ws, Ws[0].shape[0], M, kind, rng)
            q = quartic_cv_basis(u, directions)
            est = cross_fitted_cv(h, q)
            results[f"{kind}_M{M}"].append(np.mean((est - truth) ** 2))

        if (i + 1) % 20 == 0:
            base = np.mean(results["raw"])
            print(f"{i+1}/{n_mlps}: raw={base:.3e}  " +
                  "  ".join(f"{k}={np.mean(v)/base:.2f}x" for k, v in results.items() if k != "raw"),
                  flush=True)

    base = np.mean(results["raw"])
    print(f"\nraw (current mmL1): {base:.4e}")
    for k, v in sorted(results.items(), key=lambda kv: np.mean(kv[1]) if kv[0] != "raw" else 1e9):
        if k == "raw":
            continue
        m = np.mean(v)
        print(f"{k:20s} {m:.4e}  ({base/m:.2f}x)")


if __name__ == "__main__":
    main()
