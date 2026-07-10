"""Test: multilevel Monte Carlo (MLMC) with a width-subsampled coarse network.

Neither our own research nor the independent pscamillo writeup (see
EXPERIMENTS.md round 8) tested this. Idea: build a cheaper "coarse" network
from the SAME weight matrices (bottleneck the internal width from 256 to
n_c), sharing input randomness with the true "fine" network. If fine and
coarse outputs are correlated, the two-level MLMC estimator

    E[fine] ~= mean(coarse samples, CHEAP, many)
             + mean(fine - coarse, on a SHARED-randomness subset, FEW)

has lower variance than plain fine-only MC at the same total cost, provided
Var(fine - coarse) << Var(fine).

Coarse network construction: full input (256) -> layer 1 uses W1[:, :n_c]
(256 -> n_c) -> layers 2..31 use W_l[:n_c, :n_c] (n_c -> n_c, cheap) ->
layer 32 uses W32[:n_c, :] (n_c -> 256, matches the scored output dim).
This keeps the true input and output dimensions; only the INTERNAL width is
bottlenecked, so the FLOP cost of the bottlenecked layers drops by
(n_c/256)^2.

Step 1 (this script): just measure the fine/coarse correlation and the
variance ratio Var(fine-coarse)/Var(fine) using the SAME mmL1 sampler and
shared randomness -- if this ratio isn't well below 1, MLMC can't help and
there's no point building the full two-level estimator.
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


def fine_coarse_forward(Ws, seed, m_t, cov_t, n_c, n_pairs=2750):
    """Runs fine and coarse networks on the SAME antithetic samples (shared
    randomness / common random numbers -- essential for MLMC correlation).
    Returns (fine_out, coarse_out): each (N, 256)."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)

    # fine network, exact layer-1 moment matching (current mmL1)
    h1_fine = np.maximum(x @ Ws[0], 0.0)
    h_fine = affine_match(h1_fine, m_t, cov_t)
    for w in Ws[1:-1]:
        h_fine = np.maximum(h_fine @ w, 0.0)
    fine_out = np.maximum(h_fine @ Ws[-1], 0.0)

    # coarse network: same shared input x, bottlenecked internal width.
    # NOTE: layer 1 of the coarse net does NOT get the moment-matching
    # treatment (that's specific to the fine network's full-width Gram
    # matrix); it shares the same raw x for correlation.
    W1_c = Ws[0][:, :n_c]
    h_coarse = np.maximum(x @ W1_c, 0.0)
    for w in Ws[1:-1]:
        Wc = w[:n_c, :n_c]
        h_coarse = np.maximum(h_coarse @ Wc, 0.0)
    W_last_c = Ws[-1][:n_c, :]
    coarse_out = np.maximum(h_coarse @ W_last_c, 0.0)

    return fine_out, coarse_out


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 20
    n_cs = [32, 64, 128, 192]
    corr_results = {nc: [] for nc in n_cs}
    var_ratio_results = {nc: [] for nc in n_cs}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]

        for n_c in n_cs:
            fine_out, coarse_out = fine_coarse_forward(Ws, row["mlp_seed"], m_t, cov_t, n_c)
            # per-neuron correlation between fine and coarse across samples
            fc = fine_out - fine_out.mean(axis=0)
            cc = coarse_out - coarse_out.mean(axis=0)
            num = (fc * cc).mean(axis=0)
            denom = np.sqrt((fc**2).mean(axis=0) * (cc**2).mean(axis=0)) + 1e-24
            corr = num / denom
            var_fine = (fc**2).mean(axis=0)
            var_diff = ((fc - cc) ** 2).mean(axis=0)
            ratio = var_diff / (var_fine + 1e-24)
            corr_results[n_c].append(np.median(corr))
            var_ratio_results[n_c].append(np.median(ratio))

        if (i + 1) % 5 == 0:
            print(f"{i+1}/{n_mlps} done", flush=True)

    print("\nFine/coarse correlation and Var(fine-coarse)/Var(fine) by n_c "
          f"(median over 256 final neurons, {n_mlps} MLPs):")
    for n_c in n_cs:
        print(f"n_c={n_c:4d}  median corr={np.mean(corr_results[n_c]):.3f}  "
              f"median var_ratio={np.mean(var_ratio_results[n_c]):.3f}  "
              f"(want << 1 for MLMC to help)")


if __name__ == "__main__":
    main()
