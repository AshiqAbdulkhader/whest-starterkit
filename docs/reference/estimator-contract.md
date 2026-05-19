# Estimator Contract

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### Lifecycle

```
  Estimator()           ──▶  __init__         (cheap; no I/O, no compute)
       │
       ▼
  setup(context)        ──▶  one call before any predict()
       │                     • runs OUTSIDE any BudgetContext (off-budget)
       │                     • bounded by setup_timeout_s (default ~5s)
       │                     • good for: lookup tables, config loads,
       │                                  shape-independent precompute
       ▼
  predict(mlp_1, b)     ──▶  one call per MLP
  predict(mlp_2, b)            • runs INSIDE a BudgetContext
  ...                          • bounded by --flop-budget and (optionally)
  predict(mlp_M, b)              --wall-time-limit / --residual-wall-time-limit
       │
       ▼
  teardown()            ──▶  one call after all predict() calls
                             • cleanup of resources opened in setup()
```

`setup()` and `teardown()` are entirely optional. The four bundled examples all
override `setup()` to demonstrate the [setup-time seeding contract](#setup-time-reproducibility) — for a purely deterministic estimator without setup-time
precompute the body can be a no-op. Define `setup()` when you have shape-agnostic
precompute that's expensive enough to be worth doing once, or when you need
seeded randomness via `ctx.seed`. See
[FAQ: Can I precompute things in setup()?](../troubleshooting/faq.md#can-i-precompute-things-in-setup)
for budget rules.

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `depth` | `int` | Number of layers per MLP |
| `flop_budget` | `int` | FLOP cap for the estimator |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching across calls (subprocess and Docker runners; otherwise typically `None`) |
| `seed` | `int` | Run-level seed for setup-time randomness, defaults to `0`. See [Setup-time reproducibility](#setup-time-reproducibility) below. |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |
| `MLP` | `seed` | Per-MLP grader-supplied seed; use this to seed estimator-internal randomness for reproducibility under regrade. See [Reproducibility under the grader seed](#reproducibility-under-the-grader-seed) below. |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops` and `import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

## Failure semantics

The harness never crashes on a bad estimator. Every failure mode is
surfaced as report data so that one bad MLP doesn't take down the run.

| Failure | Behavior | Report field(s) surfacing it | Stage that catches it first |
|---|---|---|---|
| Wrong return shape (not `(mlp.depth, mlp.width)`) | predictions for this MLP zeroed | `per_mlp[i].error.details.{expected_shape, got_shape}` | Stage 2 (`whest validate`) |
| Wrong dtype (not a `flopscope.numpy.ndarray`) | predictions for this MLP zeroed | `per_mlp[i].error` with hint | Stage 2 |
| Non-finite values (NaN, Inf) | predictions for this MLP zeroed | `per_mlp[i].error.details.cause_hints` | Stage 2 |
| `predict()` raised an exception | predictions for this MLP zeroed; harness continues to the next MLP; CLI exits `1` and prints an "Estimator Errors" panel | `per_mlp[i].{error, error_code, traceback}`; `error_code` is the Python exception class name | Stage 3 (`whest run`) |
| Exceeded `flop_budget` | flopscope raises `BudgetExhaustedError` *before* the over-budget op runs; predictions zeroed | `per_mlp[i].budget_exhausted: true` | Stage 3 |
| Exceeded `--wall-time-limit` (`wall_time_limit_s`) | flopscope raises `TimeExhaustedError`; predictions zeroed | `per_mlp[i].time_exhausted: true` | Stage 3 (with `--wall-time-limit`) |
| Exceeded `--residual-wall-time-limit` | scoring layer (not flopscope) zeroes the predictions after `predict()` returns | `per_mlp[i].residual_wall_time_exhausted: true` | Stage 3 (with `--residual-wall-time-limit`) |

When `predict()` raises, the runner captures the exception, records the
class name in `error_code`, and forwards a formatted `traceback` (subprocess
runs forward it across the worker boundary). Use `--debug` to see
tracebacks inline; `--fail-fast` to halt at the first failure.

Predictions for the failed MLP are scored against zeros AND the per-MLP multiplier is forced to **1.0** (no compute discount), so the per-MLP `adjusted_final_layer_score_m = MSE(0, Y_m) × 1.0`. This is strictly worse than a trivial-zero submission that succeeds, which receives the 0.1 multiplier floor — a factor-of-ten cap on the discount. The suite mean stays finite either way; the `failure_breakdown` and `n_failed_mlps` aggregates surface how many MLPs hit which failure path. If you want the run to stop at the first problem rather than score-against-zeros, use `--fail-fast`.

For the structured `error.details` schema, see [score-report-fields.md](score-report-fields.md#per-mlp-fields).

## Reproducibility under the grader seed

The grader supplies two independent seeds that you must use for any randomness in your estimator: `mlp.seed` for per-MLP randomness inside `predict()`, and `ctx.seed` for one-time randomness inside `setup()`. Both default to `0` when `--seed` is not passed, and both are recorded in the JSON output for audit (`run_config.seed`). All four bundled examples (`examples/0[1-4]_*.py`) carry the scaffold side-by-side so the pattern is visible whether you start from the random baseline or one of the deterministic propagators.

### Predict-time reproducibility

If your estimator uses randomness — Monte Carlo sampling, randomized hashing, random projections, etc. — seed it from `mlp.seed`. The grader supplies a fixed per-MLP seed that is identical across all submissions for a given MLP, derived deterministically from the suite seed. **Submissions that use unseeded randomness or their own seeds are NOT guaranteed to reproduce under regrade and may be disqualified for prize eligibility.**

```python
import flopscope.numpy as fnp

def predict(self, mlp, budget):
    rng = fnp.random.default_rng(mlp.seed)
    # ... use rng for any internal randomness
```

If your estimator is deterministic (no internal randomness), you can ignore `mlp.seed`.

### Setup-time reproducibility

If your estimator does randomized one-time setup (e.g., sampling a random projection basis, jittering initial weights, choosing random hyperparameters), seed it from `ctx.seed` inside `setup()`. When the grader passes `--seed`, the same value is forwarded to `ctx.seed` for every MLP in the run; participants running locally can pass `--seed` themselves to reproduce a given setup.

```python
import flopscope.numpy as fnp

def setup(self, ctx: SetupContext) -> None:
    self.setup_rng = fnp.random.default_rng(ctx.seed)
    # ... use self.setup_rng for any one-time random work
```

Do **not** call `fnp.random.seed(ctx.seed)` (or `np.random.seed(ctx.seed)`) — that mutates the process-global RNG and breaks composability with other libraries. Use `fnp.random.default_rng(ctx.seed)` to get an isolated `Generator`.

`ctx.seed` defaults to `0` when no `--seed` was passed; estimators that don't read it are unaffected. The seed is recorded in the run output under `run_config.seed` for audit-trail purposes — a reviewer can read it from a participant's JSON output and re-run with `--seed N` to reproduce the participant's setup state. See [score-report-fields.md](score-report-fields.md) for the `run_config.seed` field.

`ctx.seed` and `mlp.seed` are independent: `mlp.seed` controls per-MLP randomness inside `predict()`, `ctx.seed` controls one-time setup. With `--dataset`, the dataset supplies `mlp.seed` values (baked at the dataset's own seed) while `--seed` controls `ctx.seed` only. See [CLI reference](https://github.com/AIcrowd/whestbench/blob/main/docs/reference/cli-reference.md) for the `--seed` flag semantics.

## ➡️ Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
