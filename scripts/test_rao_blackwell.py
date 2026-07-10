"""Test: output-side Rao-Blackwellization of the final layer (idea sourced
from a Codex consult -- see EXPERIMENTS.md research round 7).

Unlike the rejected "low-rank sampled tail" (which approximated a MID-DEPTH
layer's covariance and let 16-24 remaining ReLU layers amplify the bias) and
"cumulant extrapolation" (which compressed the final pre-activation to 2-3
scalar moments, discarding real sampled structure), this touches ONLY the
very last operation and keeps the dominant sampled structure exact:

1. Run the current mmL1 sampler through layer 31 (unchanged, exact).
2. Let H = post-ReLU activations entering the final layer (N samples, n dims).
   Decompose H's sample covariance C into a top-r eigenspace U (rank r) and
   an orthogonal residual.
3. For each output neuron j (column w_j of the final weight matrix W32),
   split w_j = U @ a_j + w_res_j (exact orthogonal decomposition).
4. The low-rank part of the pre-activation, m_ij = mean(H)@w_j + (Hc_i @ U) @ a_j,
   is kept EXACT per sample (no approximation -- these are the directions
   that carry the most of H's real, non-Gaussian sampled structure).
5. The orthogonal residual (H_res_i @ w_res_j) is treated as independent
   Gaussian noise with variance tau_j^2 = w_res_j . C . w_res_j (exact,
   computed from the same sample covariance -- only the DISTRIBUTIONAL SHAPE
   of the residual is approximated, not its magnitude).
6. Per-sample-and-neuron, replace the noisy ReLU(m_ij + residual_ij) with its
   analytic conditional expectation over the residual:
       E[ReLU(z) | m_ij] = m_ij * Phi(m_ij/tau_j) + tau_j * phi(m_ij/tau_j)
   This is a genuine Rao-Blackwellization: same information used, provably
   lower variance IF the conditional-Gaussian assumption holds even
   approximately, and any residual bias is LOCAL to the last operation (no
   downstream chaos to amplify it, unlike the mid-depth attempts).

Sweeps rank r and two choices of subspace (plain PCA(C) vs output-aware
PCA(C @ W @ W.T @ C), which favors directions that matter most for the
actual output weights) against the current mmL1 raw baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop, norm_cdf, norm_pdf  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def mmL1_penultimate(Ws, seed, m_t, cov_t, n_pairs=2750):
    """Returns H: (N, n) post-ReLU activations entering the final layer."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h1, m_t, cov_t)
    for w in Ws[1:-1]:
        h = np.maximum(h @ w, 0.0)
    return h


def raw_final_mean(H, W_final):
    z = H @ W_final
    return np.maximum(z, 0.0).mean(axis=0)


def rao_blackwell_final_mean(H, W_final, rank, subspace="pca"):
    N, n = H.shape
    mu_H = H.mean(axis=0)
    Hc = H - mu_H
    C = (Hc.T @ Hc) / N

    if subspace == "pca":
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        U = evecs[:, order[:rank]]
    elif subspace == "output_aware":
        M = C @ W_final @ W_final.T @ C
        evals, evecs = np.linalg.eigh(M)
        order = np.argsort(evals)[::-1]
        U = evecs[:, order[:rank]]
    else:
        raise ValueError(subspace)

    A = U.T @ W_final          # (rank, n_out) -- a_j columns
    W_res = W_final - U @ A    # (n, n_out) -- residual weight columns

    low_rank = mu_H @ W_final + (Hc @ U) @ A     # (N, n_out): m_ij, exact per sample
    tau2 = np.einsum("ij,jk,ik->i", W_res.T, C, W_res.T)  # (n_out,) per-neuron residual var
    tau = np.sqrt(np.maximum(tau2, 1e-24))

    alpha = low_rank / tau[None, :]
    Phi = ndtr(alpha)
    phi = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    per_sample = low_rank * Phi + tau[None, :] * phi
    return per_sample.mean(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 100
    ranks = [16, 32, 48, 64, 96, 128]
    results = {"raw": []}
    for r in ranks:
        results[f"pca_r{r}"] = []
        results[f"oaw_r{r}"] = []

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]

        H = mmL1_penultimate(Ws, row["mlp_seed"], m_t, cov_t)
        W_final = Ws[-1]

        raw = raw_final_mean(H, W_final)
        results["raw"].append(np.mean((raw - truth) ** 2))

        for r in ranks:
            pca = rao_blackwell_final_mean(H, W_final, r, subspace="pca")
            oaw = rao_blackwell_final_mean(H, W_final, r, subspace="output_aware")
            results[f"pca_r{r}"].append(np.mean((pca - truth) ** 2))
            results[f"oaw_r{r}"].append(np.mean((oaw - truth) ** 2))

        if (i + 1) % 10 == 0:
            base = np.mean(results["raw"])
            print(f"{i+1}/{n_mlps}: raw={base:.3e}  " +
                  "  ".join(f"pca_r{r}={np.mean(results[f'pca_r{r}'])/base:.2f}x "
                            f"oaw_r{r}={np.mean(results[f'oaw_r{r}'])/base:.2f}x"
                            for r in ranks),
                  flush=True)

    base = np.mean(results["raw"])
    print(f"\nraw (current mmL1): {base:.4e}")
    for r in ranks:
        pca_v = np.mean(results[f"pca_r{r}"])
        oaw_v = np.mean(results[f"oaw_r{r}"])
        print(f"r={r:4d}  pca={pca_v:.4e} ({base/pca_v:.2f}x)   "
              f"output_aware={oaw_v:.4e} ({base/oaw_v:.2f}x)")


if __name__ == "__main__":
    main()
