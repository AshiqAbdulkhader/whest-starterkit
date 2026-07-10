"""Test: exact radius/direction decomposition from positive homogeneity.

Mathematical basis: the network has NO bias terms anywhere, so
F(t*u) = t*F(u) for t>0 (ReLU(t*z)=t*ReLU(z) composes through every layer).
For u~N(0,I_n), r=||u|| and theta=u/||u|| are exactly independent (isotropic
Gaussian fact), and F(u) = r*f(theta) where f(theta):=F(theta). So:

    E[F(u)] = E[r]*E[f(theta)]   (independence)

E[r] for chi_n is an exact closed-form constant with ZERO variance. Replacing
each sample's own random r with this constant (keeping only its direction)
is a provably variance-reducing, EXACTLY unbiased operation (a special case
of Rao-Blackwellization that's exact, not approximate).

Derived expected effect size: Var(r)/n ~ 0.195% for n=256 (chi_256 is
extremely concentrated in high dimensions) -- small, but free and exact.
Tests whether it's additive on top of our existing mmL1 (layer-1 covariance
matching), since that's a different mechanism (full covariance correction on
h1, not radius correction on u).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402

N = 256
E_R = np.sqrt(2) * np.exp(gammaln((N + 1) / 2) - gammaln(N / 2))


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward(x, Ws):
    for w in Ws:
        x = np.maximum(x @ w, 0.0)
    return x


def run(Ws, seed, m_t, cov_t, n_pairs, radius_fix, l1_match):
    n = Ws[0].shape[0]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_pairs, n))
    x = np.concatenate([u, -u], axis=0)

    if radius_fix:
        r = np.linalg.norm(x, axis=1, keepdims=True)
        x = x * (E_R / r)

    h1 = np.maximum(x @ Ws[0], 0.0)
    if l1_match:
        h1 = affine_match(h1, m_t, cov_t)
    h = h1
    for w in Ws[1:]:
        h = np.maximum(h @ w, 0.0)
    return h.mean(axis=0)


CONFIGS = [
    ("plain_mc", dict(radius_fix=False, l1_match=False)),
    ("radius_only", dict(radius_fix=True, l1_match=False)),
    ("mmL1 (current)", dict(radius_fix=False, l1_match=True)),
    ("mmL1 + radius", dict(radius_fix=True, l1_match=True)),
]


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    n_mlps = 100
    n_pairs = 2750
    errs = {name: [] for name, _ in CONFIGS}

    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]

        for name, kwargs in CONFIGS:
            est = run(Ws, row["mlp_seed"], m_t, cov_t, n_pairs, **kwargs)
            errs[name].append(np.mean((est - truth) ** 2))

        if (i + 1) % 20 == 0:
            print(f"{i+1}/{n_mlps}: " +
                  "  ".join(f"{k}={np.mean(v):.3e}" for k, v in errs.items()),
                  flush=True)

    base = np.mean(errs["mmL1 (current)"])
    print(f"\nFinal over {n_mlps} MLPs:")
    for name, v in errs.items():
        m = np.mean(v)
        print(f"{name:18s} {m:.4e}  ({base/m:.3f}x vs mmL1)")


if __name__ == "__main__":
    main()
