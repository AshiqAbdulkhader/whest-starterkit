"""Oracle active-subspace QMC ceiling.

Estimate input->final_mean Jacobian via finite differences on a pilot,
take top-r right singular vectors, QMC on that subspace. If this beats
the linearized-A subspace by a lot, the gap is 'better subspace estimate';
if not, low-dim QMC is near its ceiling for this problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402
from test_subspace_qmc import build_A_k1, affine_match, mmL1  # noqa: E402


def forward_mean_from_x(Ws, x):
    h = np.maximum(x @ Ws[0], 0.0)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h  # (N, n) final activations


def oracle_active_subspace(Ws, seed, r, n_pilot=64, eps=1e-3):
    """FD Jacobian of mean-final-activation scalar sum w.r.t. input, top-r."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed + 7)
    # pilot inputs
    X = rng.standard_normal((n_pilot, n))
    # Jacobian of sum_j h_L,j  w.r.t. x — shape (n_pilot, n) gradients, then average
    # For vector output, we want directions affecting the whole output:
    # use Jacobian of the vector mean: J shape (n_out, n_in), SVD.
    # Approximate J by averaging per-sample Jacobians of h_L w.r.t x.
    # Cost: n_in forwards per pilot — too expensive (256*64).
    # Instead: sketch — random projection probe
    # J_sketch = [f(x+eps*e) - f(x)] / eps for random e's, stack
    n_probe = max(r * 4, 32)
    probes = rng.standard_normal((n_probe, n))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    # evaluate at center batch
    f0 = forward_mean_from_x(Ws, X).mean(0)  # (n,) — not used
    # per-probe directional derivatives averaged over pilots
    # M[k] = mean over pilots of (f(x+eps*p_k) - f(x-eps*p_k))/(2eps)  -> (n_out,)
    M = np.zeros((n_probe, n))
    for k in range(n_probe):
        p = probes[k]
        fp = forward_mean_from_x(Ws, X + eps * p).mean(0)
        fm = forward_mean_from_x(Ws, X - eps * p).mean(0)
        M[k] = (fp - fm) / (2 * eps)
    # M ≈ probes @ J.T, so J.T ≈ pinv(probes) @ M
    # Better: SVD of M to get output space; for input space use probes weighted
    # Actually M[k] = J @ p_k approximately (if J is n_out x n_in and we want
    # directional deriv of vector). So M.T ≈ J @ probes.T, J ≈ M.T @ pinv(probes.T)
    J_est = M.T @ np.linalg.pinv(probes.T)  # (n_out, n_in)
    _, _, Vt = np.linalg.svd(J_est, full_matrices=False)
    return Vt[:r].T  # (n_in, r)


def hybrid_with_Q(Ws, seed, n_pairs, Q, method="strat"):
    n = Ws[0].shape[0]
    n_tot = 2 * n_pairs
    n_q = 1 << int(np.ceil(np.log2(n_tot)))
    r = Q.shape[1]
    rng = np.random.default_rng(seed)
    if method == "sobol":
        eng = qmc.Sobol(d=r, scramble=True, seed=int(seed) % (2**31 - 1))
        u = np.clip(eng.random(n_q), 1e-10, 1 - 1e-10)
        z_sub = norm.ppf(u)[:n_tot]
    else:
        z_sub = np.zeros((n_tot, r))
        for j in range(r):
            centers = (np.arange(n_tot) + rng.random(n_tot)) / n_tot
            rng.shuffle(centers)
            z_sub[:, j] = norm.ppf(np.clip(centers, 1e-10, 1 - 1e-10))
    m = n_tot // 2
    u_full = rng.standard_normal((m, n))
    x_orth = np.concatenate([u_full, -u_full], axis=0)
    x_orth = x_orth - (x_orth @ Q) @ Q.T
    x = z_sub @ Q.T + x_orth
    m_t, cov_t = relu_cov_exact(Ws[0])
    h = affine_match(np.maximum(x @ Ws[0], 0.0), m_t, cov_t)
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 20
    n_pairs = 1024
    r = 8

    errs = {"mmL1": [], "linA_strat": [], "oracle_strat": [], "oracle_sobol": []}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)

        errs["mmL1"].append(np.mean((mmL1(Ws, row["mlp_seed"], n_pairs) - truth) ** 2))

        A = build_A_k1(Ws)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        # push to input
        D = Ws[0] @ Vt[:r].T
        Q_lin, _ = np.linalg.qr(D)
        Q_lin = Q_lin[:, :r]
        errs["linA_strat"].append(
            np.mean((hybrid_with_Q(Ws, row["mlp_seed"], n_pairs, Q_lin, "strat") - truth) ** 2)
        )

        Q_or = oracle_active_subspace(Ws, row["mlp_seed"], r)
        Q_or, _ = np.linalg.qr(Q_or)
        Q_or = Q_or[:, :r]
        errs["oracle_strat"].append(
            np.mean((hybrid_with_Q(Ws, row["mlp_seed"], n_pairs, Q_or, "strat") - truth) ** 2)
        )
        errs["oracle_sobol"].append(
            np.mean((hybrid_with_Q(Ws, row["mlp_seed"], n_pairs, Q_or, "sobol") - truth) ** 2)
        )
        print(f"  ... {i+1}/{n_mlps}", flush=True)

    base = np.mean(errs["mmL1"])
    print(f"\n{'method':<14} {'MSE':>12} {'vs mmL1':>10}")
    for k, v in errs.items():
        m = np.mean(v)
        print(f"{k:<14} {m:>12.4e} {base/m:>9.3f}x")


if __name__ == "__main__":
    main()
