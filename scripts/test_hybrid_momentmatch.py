"""Hybrid moment-matching sweeps: full/diag mixes and partial (shrunk) pinning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_deep_momentmatch import mech_targets  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def run(Ws, seed, plan, targets, n_pairs=2750):
    """plan: list of (mode, lam) per layer index; mode in {none, full, diag}."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u_half = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u_half, -u_half], axis=0)
    for li, w in enumerate(Ws):
        x = np.maximum(x @ w, 0.0)
        mode, lam = plan[li] if li < len(plan) else ("none", 0.0)
        if mode == "none":
            continue
        m_t, cov_t = targets[li]
        mu_emp = x.mean(axis=0)
        xc = x - mu_emp
        m_eff = (1 - lam) * mu_emp + lam * m_t
        if mode == "full":
            cov_emp = (xc.T @ xc) / len(x)
            cov_eff = (1 - lam) * cov_emp + lam * cov_t
            A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_eff)
            x = xc @ A + m_eff
        else:
            var_emp = np.maximum((xc * xc).mean(axis=0), 1e-24)
            var_eff = (1 - lam) * var_emp + lam * np.maximum(np.diagonal(cov_t), 0.0)
            x = xc * np.sqrt(var_eff / var_emp) + m_eff
    return x.mean(axis=0)


def make_plans(L):
    plans = {}
    def p(spec):
        plan = [("none", 0.0)] * L
        for li, mode, lam in spec:
            plan[li] = (mode, lam)
        return plan

    plans["full2"] = p([(0, "full", 1.0), (1, "full", 1.0)])
    plans["diag4"] = p([(i, "diag", 1.0) for i in range(4)])
    plans["fullL1_diag234"] = p([(0, "full", 1.0)] + [(i, "diag", 1.0) for i in (1, 2, 3)])
    plans["full2_diag34"] = p([(0, "full", 1.0), (1, "full", 1.0),
                               (2, "diag", 1.0), (3, "diag", 1.0)])
    plans["full2_diag3to8_half"] = p([(0, "full", 1.0), (1, "full", 1.0)]
                                     + [(i, "diag", 0.5) for i in range(2, 8)])
    plans["full2_diag3to16_quarter"] = p([(0, "full", 1.0), (1, "full", 1.0)]
                                         + [(i, "diag", 0.25) for i in range(2, 16)])
    plans["full3"] = p([(i, "full", 1.0) for i in range(3)])
    plans["full2_full34_half"] = p([(0, "full", 1.0), (1, "full", 1.0),
                                    (2, "full", 0.5), (3, "full", 0.5)])
    plans["diag_all_decay"] = p([(0, "full", 1.0), (1, "full", 1.0)]
                                + [(i, "diag", 1.0 / (i - 0.5)) for i in range(2, 32)])
    return plans


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-mini", split="mini")
    results = {}
    for i in range(15):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        targets = mech_targets(Ws)
        plans = make_plans(len(Ws))
        for name, plan in plans.items():
            est = run(Ws, row["mlp_seed"], plan, targets)
            results.setdefault(name, []).append(np.mean((est - truth) ** 2))
        print(f"mlp {i} done", flush=True)
    print()
    for name, v in sorted(results.items(), key=lambda kv: np.mean(kv[1])):
        print(f"{name:26s} MSE {np.mean(v):.4e}")


if __name__ == "__main__":
    main()
