# Stage 3: Run Locally (In-Process Harness)

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · **3** · [4](stage-4-run-subprocess.md) · [5](stage-5-run-docker.md) · [6](stage-6-package.md)

Stage 2 confirms the contract. Stage 3 actually scores you against the same MLPs the grader uses — but in-process, so you can drop `import pdb; pdb.set_trace()` anywhere in `predict()` and step through it.

## 🚀 Run it

```bash
uv run whest run --estimator estimator.py --runner local
```

The default runner is `local` — you can omit `--runner local`. Defaults match the grader: `--n-mlps 10`, `--width 256`, `--depth 8`, `--flop-budget 6.8e10` (68B effective-compute budget), `--wall-time-limit 60.0`. See [CLI Reference](../reference/cli-reference.md) for the full list.

You'll see a Rich-rendered report with five panels:

1. **Run Context** — estimator class, path, timestamps, `n_mlps`, `width`, `depth`, `flop_budget`.
2. **Hardware & Runtime** — host, OS, CPU, RAM, Python and NumPy versions (so a leaderboard score is reproducible across machines).
3. **Sampling Budget Breakdown (Ground Truth)** — total FLOPs and time spent generating Monte-Carlo ground truth.
4. **Estimator Budget Breakdown** — same fields for your `predict()` call(s).
5. **Final Score** — the headline metrics:

```
╭──────────────────────────── Final Score ────────────────────────────╮
│                                                                     │
│   metric                                       value                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│   Adjusted Final-Layer Score                ≈ 0.05  ← primary score │
│   [adjusted_final_layer_score]                                      │
│   Raw Final-Layer MSE [final_layer_mse]      ≈ 0.5                  │
│   All-Layers MSE [all_layers_mse]            ≈ 0.5                  │
│   ────────                                  ────────                │
│   Best MLP                                   ≈ 0.04                 │
│   [best_mlp_adjusted_final_layer_score]                             │
│   Worst MLP                                  ≈ 0.06                 │
│   [worst_mlp_adjusted_final_layer_score]                            │
│   ────────                                  ────────                │
│   Mean Score Multiplier                     0.10000000              │
│   [mean_score_multiplier]                                           │
│   Mean Compute Utilization                  0.00000001              │
│   [mean_compute_utilization]                                        │
│   Failed MLPs [n_failed_mlps]                  0 of 10              │
│                                                                     │
╰─ per-MLP score = final_layer_mse × max(0.1, effective_compute/flop_budget) ─╯
```

With the zeros template, the raw `final_layer_mse` and `all_layers_mse` hover around 0.5 — the natural variance of the ReLU activations. The leaderboard metric `adjusted_final_layer_score` is a 10× discount of `final_layer_mse` (the multiplier hits the 0.1 floor) because the zeros template uses essentially none of the FLOP budget; `mean_score_multiplier` will read `0.10000000` for any cheap estimator. See [Scoring Model](../concepts/scoring-model.md) for the budget-adjustment formula and [score-report-fields.md](../reference/score-report-fields.md) for the full schema.

## Reproducing your own local runs

`whest run --seed 42` does two things:

1. Fixes the suite root from which every `mlp.seed` is derived deterministically — same `--seed` → same MLPs and same per-MLP seeds.
2. Sets `ctx.seed = 42` on the `SetupContext` passed to your `setup()` — same `--seed` → same setup-time RNG state.

With `--dataset D --seed 42`, the dataset supplies `mlp.seed` values (baked at the dataset's own seed) and `--seed 42` controls `ctx.seed` only. `whest validate --estimator e.py --seed 42` is also accepted.

For an estimator that seeds both predict-time (`mlp.seed`) and setup-time (`ctx.seed`) randomness, two runs at the same `--seed` produce byte-identical scores. The value is recorded in the JSON output under `run_config.seed` for audit purposes (`null` when `--seed` was omitted). Test reproducibility locally by running twice with the same `--seed` and diffing the JSON reports.

## FLOP-budget callout: Stage 1 vs Stage 3

Stage 1's `local_engine.compare_against_monte_carlo` uses `estimator_budget=1e9` (more headroom for prototyping at the tiny standalone shape). Stage 3's default is `flop_budget=6.8e10` (the grader effective-compute budget — caps `C_m = F_m + λ·R_m`, not just analytical FLOPs). If your estimator's cost grows fast (e.g. covariance propagation at large widths), your Stage 3 score may differ from Stage 1 — try `--flop-budget 1e11` to confirm before optimizing.

## Why a different MSE than Stage 1?

Stage 1 uses one fixed MLP (`build_mlp(width=256, depth=8, seed=0)`) — same shape as the grader, same single MLP every run. Stage 3 generates a fresh **suite** of random MLPs at the same shape (or loads a pre-created dataset via `--dataset`), so the raw `final_layer_mse` mean averages over more variance than Stage 1's single-MLP MSE. Stage 3 also applies the budget multiplier `max(0.1, C_m / B)` on top of the raw MSE, so Stage 3's `adjusted_final_layer_score` is generally smaller than Stage 1's pure MSE.

## Debugging

Because `--runner local` runs in-process, `pdb` works:

```python
def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
    import pdb; pdb.set_trace()
    ...
```

## ✅ Expected outcome

| Estimator | Typical `adjusted_final_layer_score` (default settings, `--seed 42`) |
|---|---|
| Zeros template (`estimator.py`) | ~0.05 (raw `final_layer_mse ≈ 0.5` × 0.1 multiplier floor) |
| `01_random` | ~0.056 (raw `final_layer_mse ≈ 0.56`) |
| `02_mean_propagation` | ~7.7e-5 |
| `03_covariance_propagation` | ~3.7e-6 |
| `04_combined` | ~3.7e-6 (routes to covariance at the default budget) |

Same underlying analytical methods as the Stage 1 table, but the numbers differ because Stage 3 uses the grader shape (`width=256, depth=8`, `flop_budget=6.8e10`) and applies the budget multiplier `max(0.1, C_m / B)` on top of the raw MSE. Full benchmark methodology in [scoring-model.md](../concepts/scoring-model.md#example-estimator-benchmarks).

## ✅ When you're ready

Move on to [Stage 4: subprocess runner](stage-4-run-subprocess.md) for grader parity.
