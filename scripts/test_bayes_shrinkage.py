"""Test: per-MLP Bayesian shrinkage between the K=2 mech prediction and the
mmL1 Monte-Carlo estimate.

Model: truth = mech + mech_error, mech_error ~ N(0, tau^2 I) (fitted scalar);
mc = truth + noise, noise ~ N(0, S) (S estimated from two independent-seed MC
draws per MLP). Posterior mean given mc:
    scalar shrink: est = mech + tau^2/(tau^2 + mean(diag(S))) * (mc - mech)
    Wiener        : est = mech + tau^2*(tau^2 I + S)^-1 * (mc - mech)

tau^2 is fit on a held-in MLP block and applied out-of-sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def mmL1_mc_with_cov(Ws, seed, cov_l1, m_t, n_pairs=2750):
    """mmL1 MC estimate plus an estimate S of its own sampling covariance
    (from the empirical covariance of the final batch, scaled by 1/N)."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    for li, w in enumerate(Ws):
        x = np.maximum(x @ w, 0.0)
        if li == 0:
            mu_emp = x.mean(axis=0)
            xc = x - mu_emp
            cov_emp = (xc.T @ xc) / len(x)
            A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_l1)
            x = xc @ A + m_t
    mc = x.mean(axis=0)
    xc = x - mc
    S = (xc.T @ xc) / (len(x) * len(x))  # cov of the batch mean, iid approx
    return mc, S


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    recs = []
    for i in range(100, 180):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        mu2, fin = cov_prop(Ws)
        mc, S = mmL1_mc_with_cov(Ws, row["mlp_seed"], fin["cov_l1"], fin["targets"][0][0])
        recs.append((truth, mu2, mc, S))

    tau2 = np.mean([np.mean((t - m) ** 2) for t, m, _, _ in recs[:30]])
    print("fitted tau^2 (mech error variance):", tau2)

    errs_mc, errs_wiener, errs_scalar = [], [], []
    for truth, mech, mc, S in recs[30:]:
        n = len(truth)
        T = tau2 * np.eye(n)
        K = T @ np.linalg.inv(T + S)
        est_w = mech + K @ (mc - mech)
        s_mean = np.trace(S) / n
        w_scal = tau2 / (tau2 + s_mean)
        est_s = mech + w_scal * (mc - mech)
        errs_mc.append(np.mean((mc - truth) ** 2))
        errs_wiener.append(np.mean((est_w - truth) ** 2))
        errs_scalar.append(np.mean((est_s - truth) ** 2))

    print("mc alone      :", np.mean(errs_mc))
    print("scalar shrink :", np.mean(errs_scalar))
    print("Wiener (S)    :", np.mean(errs_wiener))


if __name__ == "__main__":
    main()
