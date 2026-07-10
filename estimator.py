"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just `flopscope` and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator


class Estimator(BaseEstimator):
    """Covariance propagation: track per-neuron mean and the full covariance
    matrix through each ReLU layer. Off-diagonal covariance uses the "gain"
    approximation (cov_post[i,j] ~= gain[i]*gain[j]*cov_pre[i,j]); the
    diagonal uses the exact ReLU marginal variance. ~1.6B FLOPs at
    width=256/depth=32 (<1% of budget) -- see EXPERIMENTS.md for how this
    compares to mean propagation and to the K=3 cumulant-propagation
    research in progress.
    """

    # If any diagonal entry of the covariance exceeds this value we rescale
    # to keep the arithmetic well-behaved in float32.
    _COV_RESCALE_THRESHOLD = 1e100

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        log_scale = 0.0

        rows = []
        for w in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var = float(fnp.max(cov_diag))
            if max_var > self._COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            mu_pre = w.T @ mu
            # einsum (not chained w.T @ cov @ w) so flopscope tags cov_pre
            # as symmetric, avoiding a SymmetryLossWarning downstream.
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

            phi_alpha = flops.stats.norm.pdf(alpha)
            Phi_alpha = flops.stats.norm.cdf(alpha)

            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu * mu, 0.0)

            sigma_np = fnp.asarray(sigma_pre, dtype=fnp.float64)
            Phi_np = fnp.asarray(Phi_alpha, dtype=fnp.float64)
            gain_np = fnp.where(sigma_np > 1e-12, Phi_np, 0.0)
            gain = fnp.array(gain_np.astype(fnp.float32))

            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
