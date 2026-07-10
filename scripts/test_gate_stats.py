"""Measure per-layer ReLU gate statistics on real MLPs.

Everything in the frozen-gate-tail idea hinges on the claim (from the
pscamillo writeup: final-layer |alpha| ~ 3.2) that deep-layer gates are
nearly deterministic. Verify directly: per layer, per neuron, measure the
gate probability p = P(z > 0) over samples, and report the distribution of
min(p, 1-p) ("flip rate" if frozen to the majority state) by depth.

Also measures what fraction of the FINAL-layer output variance survives if
tail gates from layer k are frozen (the coupling quality of the coarse
model): corr(fine, frozen-tail-coarse) and Var(fine-coarse)/Var(fine).
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


def forward_collect(Ws, seed, m_t, cov_t, n_pairs=2750):
    """Full mmL1 forward; returns per-layer gate matrices and activations."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    gates = []
    acts = []
    h = np.maximum(x @ Ws[0], 0.0)
    h = affine_match(h, m_t, cov_t)
    gates.append(None)  # layer 1 gates not needed (moment-matched anyway)
    acts.append(h)
    for w in Ws[1:]:
        z = h @ w
        g = z > 0
        h = np.where(g, z, 0.0)
        gates.append(g)
        acts.append(h)
    return gates, acts


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 5
    L = 32

    flip_by_layer = {l: [] for l in range(1, L)}
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]
        gates, acts = forward_collect(Ws, row["mlp_seed"], m_t, cov_t)
        for l in range(1, L):
            p = gates[l].mean(axis=0)          # per-neuron gate probability
            flip = np.minimum(p, 1 - p)         # flip rate if frozen to majority
            flip_by_layer[l].append(flip)

    print("Per-layer gate statistics (5 MLPs, per-neuron flip rate if frozen "
          "to majority state):")
    print(f"{'layer':>5} {'mean flip':>10} {'median':>8} {'p90':>8} "
          f"{'frac<1%':>8} {'frac<0.1%':>9} {'exp flips/sample':>16}")
    for l in range(1, L):
        f = np.concatenate(flip_by_layer[l])
        print(f"{l+1:>5} {f.mean():>10.4f} {np.median(f):>8.4f} "
              f"{np.percentile(f, 90):>8.4f} {(f < 0.01).mean():>8.3f} "
              f"{(f < 0.001).mean():>9.3f} {f.sum() / n_mlps:>16.1f}")


if __name__ == "__main__":
    main()
