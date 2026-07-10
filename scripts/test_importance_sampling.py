"""Importance sampling pilots for mmL1.

Trials:
  A) scale-IS: sample x ~ N(0, s^2 I), reweight by (s^{-n} exp(...)) — for
     Gaussian target N(0,I). Tests whether a different input scale reduces
     variance of the ReLU-path mean.
  B) tilt-IS: sample x ~ N(μ_dir, I) with μ_dir along the top sensitivity
     direction (linearized final-layer gain backprop approx), reweight.
  C) mixture: half antithetic N(0,I) + half tilted, self-normalized IS.

All compared paired against mmL1 at equal n_pairs (IS pays reweight cost
only — FLOPs dominated by forwards, so equal N is fair).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import relu_cov_exact  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward_from_h1(Ws, h1):
    h = h1
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h


def mmL1(Ws, seed, n_pairs):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)
    m_t, cov_t = relu_cov_exact(Ws[0])
    h1 = affine_match(np.maximum(x @ Ws[0], 0.0), m_t, cov_t)
    return forward_from_h1(Ws, h1).mean(0)


def scale_is(Ws, seed, n_pairs, s):
    """Sample N(0,s^2 I), antithetic, reweight to N(0,I). Skip mmL1 (biased under IS)."""
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n)) * s
    x = np.concatenate([u, -u], axis=0)
    # log w = log p0 - log q = -0.5||x||^2 + 0.5||x||^2/s^2 - n*log(s)
    # for antithetic, weights are equal for u and -u
    r2 = (x * x).sum(axis=1)
    log_w = -0.5 * r2 + 0.5 * r2 / (s * s) - n * np.log(s)
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    h = np.maximum(x @ Ws[0], 0.0)
    for W in Ws[1:]:
        h = np.maximum(h @ W, 0.0)
    return (w[:, None] * h).sum(axis=0)


def sensitivity_dir(Ws):
    """Rough top input sensitivity: backprop ones through gain-linearized net."""
    # forward gains via diagonal K1
    n = Ws[0].shape[0]
    mu = np.zeros(n)
    var = np.ones(n)
    gains = []
    mus = [mu]
    for w in Ws:
        mu_pre = w.T @ mu
        var_pre = np.maximum((w * w).T @ var, 1e-12)
        sigma = np.sqrt(var_pre)
        from extract_features import norm_cdf, norm_pdf
        alpha = mu_pre / sigma
        Phi = norm_cdf(alpha)
        phi = norm_pdf(alpha)
        gain = np.where(sigma > 1e-12, Phi, 0.0)
        gains.append(gain)
        mu = mu_pre * Phi + sigma * phi
        ez2 = (mu_pre**2 + var_pre) * Phi + mu_pre * sigma * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        mus.append(mu)
    # backprop sensitivity of sum of final means: start with ones
    s = np.ones(n)
    for w, g in zip(reversed(Ws), reversed(gains)):
        s = w @ (g * s)  # dh_prev = W @ (gain * dh_next)
    # s is now sensitivity w.r.t. input (pre first ReLU is identity for input)
    return s / (np.linalg.norm(s) + 1e-12)


def tilt_is(Ws, seed, n_pairs, tilt):
    """Sample N(tilt*dir, I), antithetic around the tilt mean, reweight to N(0,I)."""
    n = Ws[0].shape[0]
    d = sensitivity_dir(Ws)
    mu_q = tilt * d
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    # antithetic around mu_q: mu_q+u and mu_q-u
    x = np.concatenate([mu_q + u, mu_q - u], axis=0)
    # w ∝ p0/q ; q = N(mu_q, I), p0 = N(0,I)
    # log w = -0.5||x||^2 + 0.5||x-mu_q||^2 = -x·mu_q + 0.5||mu_q||^2
    log_w = -(x @ mu_q) + 0.5 * float(mu_q @ mu_q)
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    h = np.maximum(x @ Ws[0], 0.0)
    for W in Ws[1:]:
        h = np.maximum(h @ W, 0.0)
    return (w[:, None] * h).sum(axis=0)


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 30
    n_pairs = 2750

    configs = {
        "mmL1": lambda Ws, seed: mmL1(Ws, seed, n_pairs),
        "scale_s1.1": lambda Ws, seed: scale_is(Ws, seed, n_pairs, 1.1),
        "scale_s0.9": lambda Ws, seed: scale_is(Ws, seed, n_pairs, 0.9),
        "scale_s1.25": lambda Ws, seed: scale_is(Ws, seed, n_pairs, 1.25),
        "tilt_0.5": lambda Ws, seed: tilt_is(Ws, seed, n_pairs, 0.5),
        "tilt_1.0": lambda Ws, seed: tilt_is(Ws, seed, n_pairs, 1.0),
        "tilt_2.0": lambda Ws, seed: tilt_is(Ws, seed, n_pairs, 2.0),
    }
    errs = {k: [] for k in configs}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"], dtype=np.float64)
        for name, fn in configs.items():
            pred = fn(Ws, row["mlp_seed"])
            errs[name].append(np.mean((pred - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n_mlps}", flush=True)

    base = np.mean(errs["mmL1"])
    print(f"\n{'method':<14} {'MSE':>12} {'vs mmL1':>10}")
    for k in configs:
        m = np.mean(errs[k])
        print(f"{k:<14} {m:>12.4e} {base/m:>9.3f}x")


if __name__ == "__main__":
    main()
