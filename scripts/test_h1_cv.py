"""Test: first-layer sufficient-statistic control variates.

z1 = u @ W1 is EXACTLY Gaussian (u ~ N(0,I)), so these have closed-form
expectations per neuron j (sigma_j = ||W1[:,j]||):
    B1_j = ReLU(z1_j)      E = sigma_j / sqrt(2*pi)
    B2_j = ReLU(z1_j)^2    E = sigma_j^2 / 2
    B3_j = z1_j^2          E = sigma_j^2

Everything after layer 1 is a deterministic function of h1 = ReLU(z1), so the
final output's sampling fluctuation is g(h1) fluctuation; regressing it on
centered basis functions of z1 (cross-fitted across half-batches so the
estimate stays exactly unbiased) removes the basis-explained variance.

Compares (15 mini MLPs, 2750 antithetic pairs):
  A: whitened antithetic (current submission)
  B: antithetic + h1-CV (basis B1)
  C: antithetic + h1-CV (basis B1+B2+B3)
  D: whitened antithetic + h1-CV basis B1 (approx expectations)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def forward_from_h1(h, Ws_rest):
    x = h
    for w in Ws_rest:
        x = np.maximum(x @ w, 0.0)
    return x  # (N, n) final activations


def whiten(u):
    C = (u.T @ u) / len(u)
    evals, evecs = np.linalg.eigh(C)
    S = (evecs / np.sqrt(np.maximum(evals, 1e-12))) @ evecs.T
    return u @ S


def cv_estimate(F, B, lam=1e-6):
    """Cross-fitted CV: F (N, n_out) samples, B (N, K) centered basis with
    E[B]=0 exactly. Returns corrected mean estimate of F (n_out,)."""
    N = len(F)
    half = N // 2
    idx1, idx2 = np.arange(half), np.arange(half, N)
    est = np.zeros(F.shape[1])
    for tr, te in ((idx1, idx2), (idx2, idx1)):
        Bt, Ft = B[tr], F[tr]
        Bt_c = Bt - Bt.mean(axis=0)
        Ft_c = Ft - Ft.mean(axis=0)
        G = Bt_c.T @ Bt_c / len(tr) + lam * np.eye(B.shape[1])
        C = np.linalg.solve(G, Bt_c.T @ Ft_c / len(tr))  # (K, n_out)
        # apply on held-out half: E[B]=0 exactly, so correction is unbiased
        est += F[te].mean(axis=0) - B[te].mean(axis=0) @ C
    return est / 2


def run_variants(Ws, seed, n_pairs=2750):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u_half = rng.standard_normal((n_pairs, n))

    W1 = Ws[0]
    sigma1 = np.sqrt((W1 * W1).sum(axis=0))

    out = {}

    # --- A: whitened antithetic (current) ---
    uw = whiten(u_half)
    x = np.concatenate([uw, -uw], axis=0)
    h = np.maximum(x @ W1, 0.0)
    F = forward_from_h1(h, Ws[1:])
    out["A_whiten"] = F.mean(axis=0)

    # --- B/C: plain antithetic + z1-basis CV ---
    u = np.concatenate([u_half, -u_half], axis=0)
    z1 = u @ W1
    h1 = np.maximum(z1, 0.0)
    F2 = forward_from_h1(h1, Ws[1:])

    B1 = h1 - sigma1 / math.sqrt(2 * math.pi)          # E[ReLU] exact
    out["B_h1cv"] = cv_estimate(F2, B1)

    B2 = h1 * h1 - sigma1**2 / 2                        # E[ReLU^2] exact
    B3 = z1 * z1 - sigma1**2                            # E[z^2] exact
    out["C_h1cv_full"] = cv_estimate(F2, np.concatenate([B1, B2, B3], axis=1))

    # --- D: whitened + h1-CV with approximate expectations ---
    z1w = x @ W1
    h1w = np.maximum(z1w, 0.0)
    B1w = h1w - sigma1 / math.sqrt(2 * math.pi)
    out["D_whiten_h1cv"] = cv_estimate(F, B1w)

    return out


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    errs = {}
    n_mlps = 15
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        res = run_variants(Ws, seed=row["mlp_seed"])
        for k, v in res.items():
            errs.setdefault(k, []).append(np.mean((v - truth) ** 2))
        print(f"mlp {i}: " + "  ".join(f"{k}={errs[k][-1]:.2e}" for k in sorted(errs)))
    print()
    base = np.mean(errs["A_whiten"])
    for k in sorted(errs):
        m = np.mean(errs[k])
        print(f"{k:16s} MSE {m:.4e}  ({base/m:.2f}x vs whitened)")


if __name__ == "__main__":
    main()
