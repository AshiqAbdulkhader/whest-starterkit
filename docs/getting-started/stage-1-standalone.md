# Stage 1: Iterate Locally (Just `flopscope`)

> [← Tutorial](README.md)

> Ladder: **1** · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · [5](stage-5-run-docker.md) · [6](stage-6-package.md)

"*Just `flopscope`*" means: **no `whest` CLI required**. You run `python estimator.py` and the bundled [`local_engine.py`](../../local_engine.py) constructs an MLP, calls your `predict()` inside a `flopscope.BudgetContext`, and sweeps Monte-Carlo sample counts to print a FLOPs-vs-MSE table. The `whestbench.BaseEstimator` and `whestbench.MLP` types you'll see imported are just the shared dataclasses — they don't pull in the harness.

Iterate here until `predict()` converges, then climb to Stage 2 to confirm the contract.

## 🚀 Run it

```bash
uv run python estimator.py
```

You should see a table like:

```
--- Your estimator ---
MLP: width=256 depth=8 seed=0

 n_samples | sampling_flops | estimator_flops |        MSE
----------------------------------------------------------
        10 |      5,329,408 |               0 |   0.706450
       100 |     53,275,648 |               0 |   0.718928
     1,000 |    532,738,048 |               0 |   0.739125
    10,000 |  5,327,362,048 |               0 |   0.735092
   100,000 | 53,273,602,048 |               0 |   0.737172
```

The MLP shape matches the Stage-3 grader defaults (`width=256, depth=8`) so the numbers you see here transfer directly to what `whest run` produces — same MLPs and same per-MLP `mlp.seed`. The stub `predict()` returns all zeros, so `estimator_flops` is `0` and the MSE plateaus at the variance of the true outputs — once you put real math in `predict()`, both columns come alive and the MSE should shrink roughly as `1/sqrt(n_samples)` (Monte Carlo converging to your estimator's answer).

## Edit `predict()`

Open [estimator.py](../../estimator.py). The body of `predict()` returns all zeros — replace it with your idea. The template already imports `flopscope as flops` and `flopscope.numpy as fnp`, so any array op you write through `fnp` (or via Python operators on `fnp` arrays) is FLOP-counted automatically. Re-run; the MSE column tells you how close you are, and `estimator_flops` shows what your math cost.

## Compare against a baseline

```bash
uv run python estimator.py --baseline mean_propagation
```

This loads `examples/02_mean_propagation.py` and runs both estimators on the same MLP.

## ✅ Expected outcome

| Estimator | MSE on the default MLP (n_samples=100,000) | `estimator_flops` | Status |
|---|---|---|---|
| Zeros template (default) | ~0.737 | 0 | floor — natural variance of the activations |
| `--baseline random` | ~0.478 | ~2,000 | uniform random — only beats zeros by chance |
| `--baseline mean_propagation` | ~0.00052 | ~270 M | ~1,400× better than zeros; first-order analytical |
| `--baseline covariance_propagation` | ~0.000024 | ~337 M | tracks neuron correlations; ~20× better than mean propagation |
| `--baseline combined` | ~0.000024 | ~337 M | budget-aware routing — at this budget always picks covariance (see [examples/README.md](../../examples/README.md)) |

You're ready for Stage 2 once your estimator's MSE is comfortably below
the zeros floor and `estimator_flops` stays under whatever budget you'd
target downstream (Stage 3's grader default is `6.8e10` — the new effective-compute budget; see [Scoring Model](../concepts/scoring-model.md) for the `C_m = F_m + λ·R_m` formula).

## ✅ When you're ready

Move on to [Stage 2: validate the contract](stage-2-validate.md).
