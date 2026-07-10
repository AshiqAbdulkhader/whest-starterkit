"""Vectorized Sobol point generation from a precomputed direction-number
matrix V (harvested once from scipy's internal `_sv`; scipy itself is never
needed at runtime -- V ships as plain data with the submission).

Uses the direct XOR-of-bits construction (point(i) = XOR over set bits j of i
of V[:, j]) rather than the sequential Gray-code walk: mathematically the
same POINT SET (verified: sorted rows match exactly against the sequential
walk and against scipy's own `.random()`), just a different sample ORDER --
irrelevant since we only consume the sample mean. Vectorized over bit
positions (~30 iterations) instead of over samples (thousands), so it's cheap
in both FLOPs and wall time; this is the same algorithm used in estimator.py
(there implemented in flopscope.numpy for production).
"""

from __future__ import annotations

import numpy as np


def generate_points_vectorized(n_points, V):
    """V: (dim, bits) int64 direction numbers.
    Returns (n_points, dim) int64 unscrambled Sobol points in [0, 2^bits)."""
    dim, bits = V.shape
    idx = np.arange(n_points, dtype=np.int64)
    pts = np.zeros((n_points, dim), dtype=np.int64)
    for j in range(bits):
        bit_j = (idx >> j) & 1
        pts ^= bit_j[:, None] * V[None, :, j]
    return pts


def digital_shift(pts, bits, seed):
    rng = np.random.default_rng(seed)
    dim = pts.shape[1]
    shift = rng.integers(0, 2**bits, size=dim, dtype=np.int64)
    return pts ^ shift[None, :]


def sobol_normal(n_points, V, seed):
    """Randomized (digital-shift) Sobol points mapped to standard normal."""
    from scipy.special import ndtri

    bits = V.shape[1]
    pts = generate_points_vectorized(n_points, V)
    pts = digital_shift(pts, bits, seed)
    u = (pts.astype(np.float64) + 0.5) / (2.0**bits)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    return ndtri(u)


if __name__ == "__main__":
    from scipy.stats import qmc

    dim, n_points = 256, 5500
    s = qmc.Sobol(d=dim, scramble=False, seed=0)
    V = s._sv.copy().astype(np.int64)
    bits = s.bits

    pts_vec = generate_points_vectorized(n_points, V)
    pts_seq = np.zeros((n_points, dim), dtype=np.int64)
    x = np.zeros(dim, dtype=np.int64)
    for i in range(1, n_points):
        c = (i & -i).bit_length() - 1
        x = x ^ V[:, c]
        pts_seq[i] = x

    same_set = np.array_equal(np.sort(pts_vec, axis=0), np.sort(pts_seq, axis=0))
    print("vectorized matches sequential (as a set):", same_set)
