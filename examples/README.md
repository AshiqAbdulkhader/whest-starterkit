# Examples — A Curriculum

Read in order. Each file is a complete, runnable Stage 1 estimator.

| File | Difficulty | Expected MSE (default MLP `width=256, depth=8`, n=100k) | What it teaches |
|---|---|---|---|
| [01_random.py](01_random.py) | introductory | ~0.48 (random baseline) | The `BaseEstimator` interface and the contract: `predict(mlp, budget) -> fnp.ndarray of shape (depth, width)`. Also shows the canonical RNG seeding pattern: per-MLP randomness uses `fnp.random.default_rng(mlp.seed)`, setup-time randomness uses `fnp.random.default_rng(ctx.seed)`. |
| [02_mean_propagation.py](02_mean_propagation.py) | easy | ~5.2e-4 | First-order analytical: propagate per-neuron mean and diagonal variance through ReLU layers |
| [03_covariance_propagation.py](03_covariance_propagation.py) | medium | ~2.4e-5 | Track full covariance, not just diagonal variance — costlier but more accurate. Computes `cov_pre = einsum("ij,ia,jb->ab", cov, w, w)` so flopscope detects the symmetric result. |
| [04_combined.py](04_combined.py) | advanced | ~2.4e-5 | Use the FLOP budget: switch strategy based on budget remaining (at `width=256, depth=8` the budget rule `30 * width^2 = 1.97M` is well below any reasonable budget, so it always picks covariance propagation — hence the matching MSE) |

## Run any example

```bash
uv run python examples/02_mean_propagation.py
```

## Compare against your estimator

```bash
uv run python estimator.py --baseline mean_propagation
```
