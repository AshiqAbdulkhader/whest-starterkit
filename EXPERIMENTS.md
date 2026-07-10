# Experiment Log — ARC WhiteBox Estimation Challenge 2026

Tracks every approach tried for `estimator.py`, what it scored, and what we learned. Newest entries on top.

**Challenge:** [aicrowd.com/challenges/arc-white-box-estimation-challenge-2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
**Task:** given a random ReLU MLP (He init, width=256, depth=32) and a FLOP budget (2.72e11), predict the per-neuron mean activation under `N(0, 1)` input without running anywhere near budget's worth of forward passes. Ranked on `adjusted_final_layer_score` = `final_layer_mse × max(0.1, C_m / B)`, averaged over M=10 MLPs — so lower error *and* lower compute both help, down to a 10× discount floor at 10% budget use.

**Companion paper:** Wu, Lecomte, Winer, Robinson, Hilton, Christiano — *"Estimating the expected output of wide random MLPs more efficiently than sampling"* ([arXiv:2605.05179](https://arxiv.org/pdf/2605.05179)). Core idea: represent activation distributions via cumulants / Hermite expansions instead of drawing samples, giving both theoretical and empirical advantages over Monte Carlo for wide networks — exactly the estimation problem this challenge poses.

## How scoring works (for reference)

- `adjusted_final_layer_score` (leaderboard metric) = mean over MLPs of `final_layer_mse × max(0.1, C_m/B)`.
- `C_m = F_m + λ·R_m`: analytical FLOPs + residual wall time converted at λ = 1e11 FLOPs/sec.
- Exceeding budget zeroes all predictions for that MLP **and** forces multiplier to 1.0 — worse than the cheapest valid submission.
- Multiplier floors at 0.1 (10× discount) once you're under 10% of budget — no further reward for going cheaper past that point, only accuracy matters below the floor.
- Full details: [docs/concepts/scoring-model.md](docs/concepts/scoring-model.md)

## Baseline reference numbers (from repo docs, `mini` split, 256×32 @ 2.72e11 FLOPs)

| Approach | `final_layer_mse` | `all_layers_mse` | FLOPs used | Notes |
|---|---|---|---|---|
| Zeros (template default) | ~0.91 | — | 0 | "doing nothing" scale |
| Random | ~0.75 | ~0.62 | negligible | interface walkthrough only |
| Mean propagation (diagonal variance) | ~9.5e-04 | ~8.2e-04 | ~11M (<1% budget) | O(depth·width²), ~1000x better than zeros |
| Full covariance propagation | ~8.4e-05 | ~5.6e-05 | ~1.6B (<1% budget) | O(depth·width³), ~11x better than mean prop |

All three bundled examples spend <1% of budget, so they all bottom out at the 0.1 multiplier floor — ranked score is just `final_layer_mse / 10`.

---

## Log

### 2026-07-10 — Submission #2: mean propagation (diagonal variance)

- **What:** Replaced the zeros baseline in `estimator.py` with diagonal mean propagation — tracks per-neuron mean and variance through each ReLU layer via the analytical ReLU expectation formula (assumes independent neurons), ported from `examples/02_mean_propagation.py`. Git tag: `submission-2` (commit at time of submit).
- **Why:** Biggest accuracy jump per line of code per the algorithm-ideas doc; establishes an analytical, low-FLOP baseline to compare future approaches against.
- **Local result** (`whest run --split mini --runner local`, public `mini` split, 100 MLPs, 256×32 @ 2.72e11 budget):
  - `adjusted_final_layer_score` = **9.48e-05**
  - `final_layer_mse` = 9.48e-04, `all_layers_mse` = 8.15e-04
  - Best MLP 1.58e-05, worst MLP 4.93e-04
  - FLOPs used: 1.12e9/MLP total effective compute 2.40e10 → 0.088% of budget → multiplier floored at 0.1
  - Matches the repo's documented calibration numbers for mean propagation almost exactly.
- **Leaderboard result:** submitted twice by accident (`--watch` without a submission-id argument re-packages and re-submits rather than attaching to the prior id) — ids **315590** and **315591**. Track at `https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/315591`. Leaderboard score TBD (grading in progress at submit time) — expect ≈ local result above since the estimator is deterministic and the mini split matches the grader's dataset/budget.
- **Learned:** `whest submit --watch` requires `--estimator` (or an artifact path) even when watching — there's no "attach to existing submission id" mode in this CLI version. To watch without re-submitting, don't pass `--watch` and instead check the tracking URL directly. Also: Windows terminal needs `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` set, or `whest` crashes trying to print unicode box-drawing/checkmark characters on the default cp1252 console.
- **Next:** Try full covariance propagation (~8.4e-05 raw MSE expected, ~1.6B FLOPs, still <1% of budget) — should beat this by ~11x on raw MSE with the score dominated by the 0.1 floor either way.

### 2026-07-08 — Submission #1: zeros baseline (unmodified template)

- **What:** Submitted `estimator.py` exactly as scaffolded — `predict()` returns `fnp.zeros((mlp.depth, mlp.width))`.
- **Why:** Establish a working end-to-end submission pipeline (login → package → submit) before investing in an algorithm.
- **Expected result:** `final_layer_mse` ≈ 0.91, multiplier = 0.1 (zero FLOPs used) → `adjusted_final_layer_score` ≈ 0.091. This is the worst reasonable score — pure floor for "did nothing wrong but predicted nothing useful."
- **Next:** Swap in mean propagation as the first real estimator (biggest accuracy jump per line of code), then decide whether covariance propagation's cubic cost is worth it at width=256/depth=32.

---

## Ideas queue (not yet tried)

Pulled from [docs/how-to/algorithm-ideas.md](docs/how-to/algorithm-ideas.md) and the companion paper:

- [ ] **Mean propagation** (diagonal variance, ReLU expectation formula) — expected ~9.5e-04 raw MSE, ~11M FLOPs. Obvious next step.
- [ ] **Full covariance propagation** — expected ~8.4e-05 raw MSE, ~1.6B FLOPs (still <1% of 2.72e11 budget at width=256). Cubic cost — check where it stops being "under 1% of budget."
- [ ] **Cumulant / Hermite-expansion method from the companion paper** — richer distributional representation than mean+covariance; the paper's whole point is doing better than sampling for wide nets. Worth prototyping once diagonal/covariance baselines are in place, to see how much of the paper's method the harness needs (it targets exactly this estimation problem).
- [ ] **Low-rank covariance** (`cov ≈ U Uᵀ`, rank k) — between diagonal and full cost, `O(depth·width²·k)`. Try if full covariance is accurate but too expensive.
- [ ] **Layer-adaptive routing** — full covariance for early layers (where correlations build), diagonal once distribution looks factored. Use per-layer `all_layers_mse` from a covariance-only run to find the crossover layer.
- [ ] **Spectral / weight-statistics methods** — precompute SVD of each `W` in `setup()` (off-budget), predict per-layer gain/variance analytically, near-zero `predict()` cost. Sensitive to depth/init scaling; candidate for extreme-budget regimes. (Pennington & Worah 2017, Saxe et al. 2014)
- [ ] **Importance sampling** — bias inputs toward high-variance regions, reweight. Try if plain MC plateaus above 1/√samples.
- [ ] **Higher-order moments** (skew/kurtosis) — correct for asymmetry the Gaussian ReLU-expectation formula misses in deep layers. (Schoenholz et al. 2017)

## Template for future entries

```
### YYYY-MM-DD — Submission #N: <short name>

- **What:** <approach summary, link to estimator.py version/commit if applicable>
- **Why:** <what we expected to gain>
- **Local result:** `final_layer_mse` / `all_layers_mse` from `whest run --runner local`
- **Leaderboard result:** `adjusted_final_layer_score`, FLOPs used, multiplier
- **Learned:** <what worked, what didn't, surprises>
- **Next:** <follow-up idea>
```
