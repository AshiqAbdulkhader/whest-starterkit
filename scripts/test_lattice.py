"""Test: randomized Korobov lattice rule -- a low-discrepancy QMC method that
needs NO external tables, only modular arithmetic (safe to implement in pure
flopscope.numpy, unlike Sobol which needs a large direction-number table and
unlike scipy which isn't a declared/locked dependency and won't survive the
grader's restricted flopscope-client proxy).

Construction: pick a generating vector z (dim,) of integers coprime to N,
points are frac(i*z/N) for i=0..N-1, i.e. x_i[j] = ((i*z[j]) mod N) / N.
Randomly shifted (add a uniform random offset mod 1, standard randomized-QMC
practice) for unbiasedness -- a single random shift per sampled draw.

z is generated via a Korobov construction: z[j] = a^j mod N for a chosen
primitive-ish root a. This is a well-known simple lattice family; not as
good as an optimized Sobol/lattice table but needs zero external data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri  # offline test only; production uses flops.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import cov_prop  # noqa: E402
from test_h1_momentmatch import mat_sqrt  # noqa: E402
from test_qmc import sobol_normal  # noqa: E402


def korobov_lattice_normal(n_total, dim, seed, a=None):
    """n_total points via a randomly-shifted Korobov lattice, standard normal."""
    rng = np.random.default_rng(seed)
    if a is None:
        # a well-behaved multiplier for Korobov lattices (must be coprime to N)
        a = 76621  # a commonly-cited good Korobov constant (Fibonacci-lattice-like)
        while np.gcd(a, n_total) != 1:
            a += 1
    j = np.arange(dim)
    z = pow(int(a), 1, n_total)  # placeholder, real formula below
    # z_j = a^j mod N grows exponentially and overflows fast; use modpow per j
    z = np.array([pow(int(a), int(jj), n_total) for jj in j], dtype=np.int64)
    i = np.arange(n_total).reshape(-1, 1)
    shift = rng.random(dim)
    frac = (i * z.reshape(1, -1)) % n_total
    u = (frac / n_total + shift) % 1.0
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return ndtri(u)


def affine_match(x, m_t, cov_t):
    mu_emp = x.mean(axis=0)
    xc = x - mu_emp
    cov_emp = (xc.T @ xc) / len(x)
    A = mat_sqrt(cov_emp, inv=True) @ mat_sqrt(cov_t)
    return xc @ A + m_t


def forward(x, Ws):
    for w in Ws:
        x = np.maximum(x @ w, 0.0)
    return x.mean(axis=0)


def run(Ws, seed, m_t, cov_t, n_pairs, mode):
    n = Ws[0].shape[0]
    n_total_pow2 = 1
    while n_total_pow2 < 2 * n_pairs:
        n_total_pow2 *= 2
    if mode == "mc":
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n_pairs, n))
        x = np.concatenate([u, -u], axis=0)
    elif mode == "sobol":
        x = sobol_normal(n_total_pow2, n, seed)
    elif mode == "lattice":
        x = korobov_lattice_normal(2 * n_pairs, n, seed)
    elif mode == "lattice_antithetic":
        half = korobov_lattice_normal(n_pairs, n, seed)
        x = np.concatenate([half, -half], axis=0)
    h1 = np.maximum(x @ Ws[0], 0.0)
    h1 = affine_match(h1, m_t, cov_t)
    return forward(h1, Ws[1:])


MODES = ["mc", "sobol", "lattice", "lattice_antithetic"]


def main():
    import whestbench.dataset as wds

    d = wds.load_dataset(r"C:\Users\MUKHADE\Workspace\whest-data\phase1-full", split="full")
    errs = {m: [] for m in MODES}
    n_mlps = 100
    n_pairs = 2750
    for i in range(n_mlps):
        row = d[i]
        Ws = [np.asarray(w, dtype=np.float64) for w in row["weights"]]
        truth = np.asarray(row["final_means"])
        _, fin = cov_prop(Ws)
        m_t, cov_t = fin["targets"][0][0], fin["cov_l1"]
        for mode in MODES:
            est = run(Ws, row["mlp_seed"], m_t, cov_t, n_pairs, mode)
            errs[mode].append(np.mean((est - truth) ** 2))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{n_mlps}: " +
                  "  ".join(f"{m}={np.mean(errs[m]):.3e}" for m in MODES),
                  flush=True)
    print(f"\nFinal over {n_mlps} MLPs:")
    base = np.mean(errs["mc"])
    for m in MODES:
        v = np.mean(errs[m])
        print(f"{m:20s} {v:.4e}  ({base/v:.2f}x vs plain MC)")


if __name__ == "__main__":
    main()
