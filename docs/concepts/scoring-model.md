# Scoring Model

> [← Documentation](../README.md)

## 🎯 When to use this page

Use this page to understand how the leaderboard score is computed from your estimator's predictions.

## Pipeline at a glance

```
   ┌─────────────────────────┐
   │  random MLP_m           │   one of M MLPs (default M=10)
   │  flop_budget B          │   default B = 6.8e10
   │  mlp.seed (per-MLP RNG) │   grader-supplied; same value under regrade
   └────────────┬────────────┘
                │
                ▼
   ┌──────────────────────────────────────┐
   │  your predict(mlp_m, budget)         │   runs inside flopscope.BudgetContext
   │  (flopscope counts FLOPs analytically)│
   └────────────┬─────────────────────────┘
                │
                ▼
        failure path triggered?
        (budget/time exhausted, raised,
         wrong shape, non-finite, …)
         /                       \
     yes /                         \ no
        ▼                           ▼
   ┌──────────────┐          ┌────────────────────────┐
   │ pred_m :=    │          │ pred_m := your         │
   │   zeros      │          │   returned array       │
   │ mult_m := 1.0│          │ mult_m := max(0.1,     │
   │ (no discount)│          │   C_m / B)             │
   └──────┬───────┘          └──────────┬─────────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
         ┌─────────────────────────────────────────────┐
         │ final_layer_mse_m  =  MSE(pred_m, truth_m)  │
         │   (over the final layer's `width` cells)    │
         │ s_m  =  final_layer_mse_m  ×  mult_m        │
         └────────────────────────┬────────────────────┘
                                  │
                       (repeat for every MLP)
                                  │
                                  ▼
         ┌──────────────────────────────────────────────────┐
         │ adjusted_final_layer_score = mean over m of s_m  │
         │   (the leaderboard metric — lower is better)     │
         │                                                  │
         │ final_layer_mse  = mean over m of                │
         │                    final_layer_mse_m             │
         │ all_layers_mse   = mean over m, all              │
         │                    (depth × width) cells, of     │
         │                    (pred − truth)²               │
         │   (diagnostic aggregates — no budget multiplier) │
         └──────────────────────────────────────────────────┘
```

## 📌 TL;DR

- **Leaderboard metric**: `adjusted_final_layer_score = mean_m( final_layer_mse_m × max(0.1, C_m / B) )`. Lower is better.
- **Effective compute** `C_m = F_m + λ·R_m` (analytical FLOPs plus a residual-wall-time penalty at λ = 1e11 FLOPs/sec).
- **Discount floor**: the `max(0.1, …)` floor caps the budget discount at 10× so an arbitrarily cheap-but-wrong submission cannot dominate the ranking.
- **Failures** (budget bust, time bust, raised, wrong shape, non-finite) → predictions zeroed, multiplier forced to **1.0** (no compute discount). The suite mean stays finite; one bad MLP no longer poisons the run.
- **Diagnostics**: `final_layer_mse` and `all_layers_mse` are the raw MSEs without budget adjustment — read them to tell *why* a score is where it is.

## The core idea

The scoring model answers a specific question: **how accurately can your estimator predict expected neuron values, weighted by how much of the compute budget it actually used?**

Two ingredients combine into the leaderboard score:

1. **Accuracy** — mean squared error (MSE) between your predictions and Monte Carlo ground truth on the final layer.
2. **Compute weight** — a multiplier in `[0.1, 1.0]` reflecting how much of the FLOP budget your estimator burned. Cheap-but-accurate methods get a discount, capped at 10× by the 0.1 floor.

Failures (budget bust, time bust, exceptions, bad output) skip the discount entirely — you score against zeros at multiplier 1.0.

## How scoring works

For each of the M MLPs in the suite:

1. **Your estimator runs.** `predict(mlp_m, budget)` is called inside a `flopscope.BudgetContext`. flopscope tracks all FLOP usage analytically.
2. **Effective compute is computed.** `C_m = F_m + λ·R_m` where `F_m` is the analytical FLOP count (`flops_used`) and `R_m` is the residual wall-time bucket (`residual_wall_time_s` — time NOT inside flopscope kernels). `λ = 1e11` FLOPs/sec.
3. **Multiplier is set.** If `C_m > B` (any of the failure paths fires), predictions are zeroed and `mult_m = 1.0`. Otherwise `mult_m = max(0.1, C_m / B)`.
4. **Per-MLP score.** `s_m = final_layer_mse_m × mult_m`.
5. **Suite mean.** `adjusted_final_layer_score = mean over m of s_m` — the leaderboard metric.

`final_layer_mse` and `all_layers_mse` are reported alongside the leaderboard metric as **diagnostic aggregates** (no budget multiplier applied) so you can see whether a high `adjusted_final_layer_score` is driven by accuracy, by compute use, or by failures.

## The formula

The leaderboard metric is the **suite mean** of per-MLP budget-adjusted scores:

```
                              1   M
adjusted_final_layer_score = ─── ∑   s_m          ← lower is better
                              M  m=1


s_m = final_layer_mse_m × max(0.1, C_m / B)        (valid runs)
s_m = final_layer_mse_m × 1.0                      (failures — no compute discount)

C_m = F_m + λ·R_m                                  (effective compute)
λ   = 1e11 FLOPs/sec                               (residual-wall-time conversion rate)

final_layer_mse_m = mean over the `width` cells of MLP m's FINAL layer
                    of (pred − truth)²

  M  = number of MLPs in the suite (default 10; --n-mlps overrides)
  B  = flop_budget (default 6.8e10; --flop-budget overrides)
  F_m = flops_used by your predict() for MLP m (analytical, via flopscope)
  R_m = residual_wall_time_s for MLP m (time NOT in flopscope kernels)
```

`adjusted_final_layer_score` is what the leaderboard ranks on. Two diagnostic aggregates accompany it without budget adjustment: `final_layer_mse` (suite mean of `final_layer_mse_m`) and `all_layers_mse` (suite mean over **all** `(depth × width)` cells, not just the final layer). The per-MLP range is reported as `best_mlp_adjusted_final_layer_score` / `worst_mlp_adjusted_final_layer_score` (the min and max of `s_m` across the suite). Full schema in [score report fields](../reference/score-report-fields.md).

## Budget behavior

Your estimator receives a `budget` argument (the FLOP budget). You may use it to route between cheap and expensive algorithms — the combined estimator example does this. But you are not required to. Fixed-strategy estimators that always use the same approach work fine, as long as they stay within budget.

## Budget enforcement and failure handling

flopscope enforces the FLOP budget analytically; the scoring layer applies the multiplier and routes failures:

- **Within budget, normal run.** Predictions are scored as-is, multiplied by `max(0.1, C_m / B)`. If `C_m / B ≤ 0.1` (you used ≤10% of effective compute), the multiplier is pinned at the **0.1 floor** — a factor-of-ten discount and no more. If `C_m / B = 1.0` (you used the full budget), the multiplier is 1.0 (no discount).
- **Exceeded FLOP budget.** flopscope raises `BudgetExhaustedError`; predictions are replaced with zeros, the multiplier is forced to **1.0** (no compute discount), and `budget_exhausted: true` is recorded.
- **Combined-budget post-check.** Even if flopscope didn't fire, the scoring layer checks `C_m > B` post-hoc using `effective_compute`. Same outcome: zeros, multiplier 1.0, `combined_budget_exhausted: true`.
- **Exceeded wall-time cap** (`wall_time_limit_s`, default 60 s) → same.
- **Exceeded residual-wall-time cap** (`residual_wall_time_limit_s`, optional) → same.
- **`predict()` raised** (any exception, including `MemoryError`, `ValueError`, …) → same; `error`, `error_code`, `traceback` recorded.
- **Wrong shape** (not `(depth, width)`) or **non-finite values** → same.

The "no compute discount on failure" rule means a **failed** MLP always scores `final_layer_mse_m × 1.0`, strictly worse than a trivial-zero submission that succeeds (which gets the 0.1 floor multiplier). The suite mean stays finite — one failed MLP no longer poisons the whole submission.

See [failure breakdown in the score report](../reference/score-report-fields.md) for the `n_failed_mlps`, `failure_breakdown`, and per-MLP error fields.

## What a good score looks like

A score near zero means your predictions are highly accurate and (typically) you are using a reasonable fraction of the compute budget. A score well above zero means either your accuracy is poor (high `final_layer_mse`) or your submission failed on some MLPs (multiplier forced to 1.0 with predictions zeroed). Read the diagnostics:

- High `final_layer_mse` with low `mean_compute_utilization`: your method is fast but inaccurate — bigger compute would help.
- High `final_layer_mse` with high `mean_compute_utilization`: your method is expensive and inaccurate — re-design.
- Low `final_layer_mse` but high `adjusted_final_layer_score`: you're hitting the multiplier floor and need to spend more compute *or* you have failed MLPs (check `n_failed_mlps` and `failure_breakdown`).

Scores below what brute-force Monte Carlo sampling achieves at the same budget indicate your structural approach is genuinely better than sampling. That is the research milestone this challenge targets.

## Practical tuning intuition

- Start with a safe method that consistently emits valid rows and stays within budget — the failure path is harsh (multiplier 1.0).
- Use `flop_budget` to gate whether to run more expensive methods. The `combined_estimator` example does this.
- Tune switching behavior using local reports across budgets.
- Compare `final_layer_mse` and `all_layers_mse` in your reports to diagnose which depths are hurting your score.
- Watch `mean_compute_utilization` and `mean_score_multiplier` — if both are low (e.g. ≤0.1), you are hitting the multiplier floor and can spend more compute "for free" up to the budget.
- Use [evaluation datasets](../how-to/use-evaluation-datasets.md) to fix networks and ground truth across runs — this makes score comparisons meaningful and skips repeated sampling.

## Worked example

Suppose ground truth for an MLP's 3-neuron final layer is `[0.42, 0.38, 0.51]` and your estimator predicts `[0.40, 0.35, 0.55]`.

    final_layer_mse_m = mean([(0.40 - 0.42)^2, (0.35 - 0.38)^2, (0.55 - 0.51)^2])
                      = mean([0.0004, 0.0009, 0.0016])
                      = 0.000967

That `0.000967` is this MLP's per-MLP `final_layer_mse_m`. Now apply the budget multiplier. Suppose `flop_budget = 6.8e10`, you used `flops_used = 1.34e8`, and `residual_wall_time_s = 0.48 s` (so `λ·R_m = 1e11 × 0.48 = 4.8e10`):

    C_m   = flops_used + λ·residual_wall_time_s
          = 1.34e8 + 4.8e10
          = 4.814e10                       ≈ 70.8% of budget

    mult_m = max(0.1, C_m / B) = max(0.1, 0.708) = 0.708

    s_m   = final_layer_mse_m × mult_m
          = 0.000967 × 0.708
          = 0.000684                       ← this MLP's contribution

The leaderboard `adjusted_final_layer_score` is the **mean of `s_m` values across all MLPs** — a mean of budget-adjusted per-MLP scores, not a mean of raw MSEs.

If your residual wall time were much larger (say 5s of Python looping), `λ·R_m` would dominate, `C_m` could exceed `B`, and the **combined-budget post-check** would zero this MLP's predictions and force `mult_m = 1.0` — see [Budget enforcement and failure handling](#budget-enforcement-and-failure-handling).

## Example estimator benchmarks

The table below shows real scores from the four bundled example estimators, run with default settings (`width=256`, `depth=8`, `n_mlps=10`, `flop_budget=6.8e10`, `--seed 42`). Use these as calibration points for your own estimator.

| Estimator | `adjusted_final_layer_score` | `final_layer_mse` | `all_layers_mse` | `mean_compute_utilization` | Approach |
|---|---:|---:|---:|---:|---|
| `01_random` | 5.6e-2 | 5.6e-1 | 4.2e-1 | 3.5e-5 | Uniform random values seeded from `mlp.seed`. The bundled [`estimator.py`](../../estimator.py) at the repo root is the true (all-zeros) baseline; running `uv run whest init <dir>` in a fresh directory produces the same template. |
| `02_mean_propagation` | 7.7e-5 | 7.7e-4 | 4.3e-4 | 0.0048 | Diagonal variance, O(depth × width²). ~700× better raw MSE than random. |
| `03_covariance_propagation` | 3.7e-6 | 3.7e-5 | 1.8e-5 | 0.0067 | Full covariance, O(depth × width³). ~20× better raw MSE than mean propagation. The pre-activation covariance `W^T cov W` is computed with `fnp.einsum("ij,ia,jb->ab", cov, w, w)` so flopscope tags the result symmetric and prunes redundant work. |
| `04_combined` | 3.7e-6 | 3.7e-5 | 1.8e-5 | 0.0067 | Routes to covariance when budget allows. At the default budget always routes to covariance — same numbers as `03`. |

**How to read these numbers:**

- The **0.1 multiplier floor is active for every baseline at the default budget.** All four baselines use under 1% of the effective compute, so `mean_score_multiplier` is pinned at `0.1` and `adjusted_final_layer_score = final_layer_mse × 0.1` across the board. To beat the floor you need an estimator that spends meaningfully more effective compute — see [Algorithm Ideas](../how-to/algorithm-ideas.md) for budget-aware approaches.
- **Random baseline** reflects the natural scale of the ground truth activations. Its raw `final_layer_mse ≈ 0.56` is what predicting random uniform values produces on the default ReLU MLP shape.
- **Mean propagation** is ~700× more accurate than random — a huge improvement from a simple analytical formula with very low FLOP cost (~0.5% utilization).
- **Covariance propagation** is another ~20× better, but costs O(width³) per layer. At width=256 this still uses only ~0.7% of the budget, so the multiplier floor still applies — the per-MLP score is `0.1 × final_layer_mse`.
- **The combined estimator** routes to covariance whenever budget allows. At the default budget (6.8e10) and width=256, the routing threshold (30·width² = 1.97M) is always exceeded, so combined always picks covariance and the numbers are identical to `03`.

To reproduce: `uv run whest run --estimator examples/<NN>_<name>.py --runner local --n-mlps 10 --seed 42` (e.g. `examples/02_mean_propagation.py`).

Scores are reproducible with `--seed 42` (same seed → same MLPs → same `mlp.seed` per MLP → same predictions). Without `--seed`, scores vary slightly between runs due to random MLP generation and Monte Carlo ground truth noise.

## ➡️ Next step

- [Score Report Fields](../reference/score-report-fields.md)
- [Validate, Run, and Package](../how-to/validate-run-package.md)
