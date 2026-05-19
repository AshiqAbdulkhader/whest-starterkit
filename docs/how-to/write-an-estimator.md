# Write an Estimator

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when implementing your custom participant estimator.

## 🚀 Do this now

Start from [`examples/01_random.py`](../../examples/01_random.py), then replace the prediction logic.

Minimal structure:

```python
from __future__ import annotations

import flopscope.numpy as fnp

from whestbench import BaseEstimator, MLP


class Estimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))
```

## ✅ Expected outcome

Your estimator implements `predict(mlp, budget)` and returns a `(depth, width)` array of predicted neuron values.

## MLP traversal starter

If you need exact `MLP` field semantics or weight matrices, use:

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)

## Contract checklist

- return a `(mlp.depth, mlp.width)` array,
- all values must be finite.

## ⚠️ Common first failure

Symptom: estimator returns wrong shape.

Fix: ensure `predict` returns a 2D array with shape `(mlp.depth, mlp.width)` and all finite values.

---

## Building your first estimator

### Step 1: The zeros baseline

The template estimator returns all zeros. Run it to see what a bad score looks like:

```bash
uv run whest run --estimator estimator.py --n-mlps 3
```

Look at `adjusted_final_layer_score` — the zeros template burns negligible compute, so the multiplier hits the 0.1 floor and the score is `0.1 × final_layer_mse`. The raw `final_layer_mse` reflects the natural scale of the ReLU activations. This is your floor.

### Step 2: Mean propagation

Copy the mean propagation example — it uses the ReLU expectation formula:

```bash
cp examples/02_mean_propagation.py estimator.py
uv run whest run --estimator estimator.py --n-mlps 3
```

Compare `adjusted_final_layer_score` to the zeros baseline. Mean propagation uses the network's weights to make informed predictions, so it should score significantly better.

### Step 3: Understand the score report

The report shows per-MLP results:
- `final_layer_mse`: raw accuracy on the final layer (no budget multiplier — diagnostic).
- `adjusted_final_layer_score`: per-MLP score `final_layer_mse × max(0.1, effective_compute / flop_budget)`. The suite mean is the leaderboard metric.
- `effective_compute`: `flops_used + λ · residual_wall_time_s` with `λ = 1e11` FLOPs/sec.
- `flops_used`: analytical FLOP count from flopscope.
- `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`: failure flags. Any `true` means predictions zeroed and multiplier forced to 1.0 (no compute discount).

### Step 4: Try the combined estimator

The combined estimator routes between cheap and expensive algorithms based on budget:

```bash
cp examples/04_combined.py estimator.py
uv run whest run --estimator estimator.py --n-mlps 3
```

This demonstrates the budget-aware routing pattern — a common design for production estimators.

### Step 5: Seed randomness from the grader-supplied seeds

The grader supplies two independent seeds; use the right one for the right scope.

**Predict-time randomness** (Monte Carlo sampling, random projections, randomized hashing inside `predict()`) — seed from `mlp.seed`. The grader supplies a fixed per-MLP seed identical across all submissions for a given MLP. **Submissions that use unseeded randomness or their own per-MLP seeds are not guaranteed to reproduce under regrade and may be disqualified for prize eligibility.**

```python
import flopscope.numpy as fnp

def predict(self, mlp, budget):
    rng = fnp.random.default_rng(mlp.seed)
    samples = rng.standard_normal((100, mlp.width))
    # ... use rng for any further internal randomness
```

**Setup-time randomness** (sampling a random projection basis, jittering initial weights, choosing random hyperparameters in `setup()`) — seed from `ctx.seed`. The grader passes `--seed` and the same value reaches every estimator's `setup(ctx)` via `ctx.seed`. Local runs can pass `--seed` to reproduce.

```python
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext

class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        self.setup_rng = fnp.random.default_rng(ctx.seed)
        # one-time random precompute — e.g. a (width, k) random projection
        self.projection = self.setup_rng.standard_normal((ctx.width, 64))
```

Do **not** call `fnp.random.seed(ctx.seed)` (or `np.random.seed(ctx.seed)`) — that mutates the process-global RNG and breaks composability with other libraries. Use `fnp.random.default_rng(ctx.seed)` to get an isolated `Generator`.

For deterministic estimators (mean propagation, covariance propagation, the zeros baseline), both `mlp.seed` and `ctx.seed` are irrelevant — you can ignore them. The `examples/01_random.py` walkthrough demonstrates the seeded pattern actively; `examples/02_*`, `03_*`, and `04_*` carry the scaffold without consuming it, so the pattern is visible whichever example you copy.

`ctx.seed` and `mlp.seed` are independent: with `--dataset`, the dataset supplies `mlp.seed` values (baked at the dataset's own seed) while `--seed` controls `ctx.seed` only. See [Estimator Contract: Reproducibility](../reference/estimator-contract.md#reproducibility-under-the-grader-seed) for the full contract requirement.

---

## Recommended learning path

1. [`examples/01_random.py`](../../examples/01_random.py) — the interface
2. [`examples/02_mean_propagation.py`](../../examples/02_mean_propagation.py) — simplest real algorithm
3. [`examples/03_covariance_propagation.py`](../../examples/03_covariance_propagation.py) — more accurate, more expensive
4. [`examples/04_combined.py`](../../examples/04_combined.py) — budget-aware routing
5. [`estimator.py`](../../estimator.py) — the repo-root template, runnable two ways: `uv run python estimator.py` for the pure-local pedagogical loop (see [Stage 1](../getting-started/stage-1-standalone.md)) and `uv run whest run --estimator estimator.py` for the harness path. Copy when you want a minimal iteration loop.
6. [Algorithm Ideas](./algorithm-ideas.md) — full survey of strategies
7. [Performance Tips](./performance-tips.md) — FLOP optimization patterns

## ➡️ Next step

- [Inspect and Traverse MLP Structure](./inspect-mlp-structure.md)
- [Algorithm Ideas](./algorithm-ideas.md)
- [Manage FLOP Budget](./manage-flop-budget.md)
- [Estimator Contract](../reference/estimator-contract.md)
- [Validate, Run, and Package](./validate-run-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
