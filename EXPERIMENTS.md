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

## Leaderboard target (checked 2026-07-10)

Top of the live leaderboard clusters around `adjusted_final_layer_score` ≈ **1e-7 to 2e-7**:

| Rank | Participant | Adjusted score | `final_layer_mse` |
|---|---|---|---|
| 1 | andrew_epstein | 9.16e-8 | 3.55e-7 |
| 2 | pluto | 9.27e-8 | 9.27e-7 |
| 3 | SOX (team) | 1.40e-7 | 3.73e-7 |
| 5 | ionel_chiosa | 1.51e-7 | 3.20e-7 |

That's ~100-1000x better raw MSE than our covariance-propagation (K=2) baseline (8.4e-5). The gap points squarely at the companion paper's higher-order **cumulant propagation** (K=3/K=4), not incremental tuning of K≤2.

## Research: K=3 cumulant propagation feasibility at our depth (2026-07-10)

Investigated whether the paper's K=3 "factored" cumulant propagation algorithm can hit leaderboard-level accuracy within our FLOP budget. Findings:

1. **Found a directly relevant reference implementation**: [ascender1729/whestbench-cumulant-propagation](https://github.com/ascender1729/whestbench-cumulant-propagation) (MIT) — an independent NumPy/flopscope port of ARC's official `mlp_cumulant_propagation` repo, built specifically for this challenge. Its `RESULTS.md` documents confirmed live-grader submissions at the **warmup shape (depth=8)**: k=3 factored cumulant propagation scored `adjusted_final_layer_score` ≈ **6.65e-7 to 7.53e-7** — squarely in leaderboard territory — vs ~3.6e-6 for covariance-propagation-only at that depth.
2. **The vendored code didn't run as-is** against our current `whestbench 0.12.0rc3` / `flopscope 0.8.0rc5` (it targeted `flopscope 0.5.0`). Two classes of bugs, both from flopscope's array immutability being enforced more strictly now:
   - `arr += x` / `arr *= x` etc. on flopscope-wrapped arrays now raise `TypeError` ("flopscope arrays are immutable"). Fixed ~40 call sites across `factor_k3_np.py`, `harmonic_np.py`, `cumulants_np.py`, `diagslice_np.py`, `kprop_np.py`, `wick_np.py`, `partitions_np.py` by rewriting in-place ops as reassignment (`x = x + y`).
   - `np.zeros_like` / `np.ones_like` on a flopscope-wrapped array raise `TypeError: no implementation found ... __array_function__`. Fixed 8 sites by replacing with `x * 0`, `x * 0 + 1`.
   - After both fixes, `kprop_layer_means(Ws, k_max=3, kind=SIMPLE, factor=True)` runs correctly and produces sane, non-degenerate output (verified: its own K=2 path gives `final_layer_mse` 1.334e-4 on our test MLP, essentially identical to our own trusted `examples/03_covariance_propagation.py`'s 1.334e-4 on the same MLP — confirms the port is numerically correct, not just "runs without crashing").
3. **Cost problem: our shape is much deeper than what's been validated.** The paper's factored-K3 runtime is `O(L² n^K)` — quadratic in depth. The reference repo's numbers above are for **depth=8**; we're at **depth=32** (4x deeper → predicted 16x more expensive). Measured directly: full-depth K=3 costs **~2.7-3.0e11 analytical FLOPs — 99-110% of our entire 2.72e11 budget** — before even counting residual Python wall-time (which added another ~45% of budget-equivalent from cold `@cache` misses in one run). One direct `whest run` attempt at full depth **exceeded budget and got zeroed** (effective compute 143% of B).
4. **Tried "layer-adaptive routing" to cut cost — doesn't work.** Idea: cheap K=2 (covariance) propagation for the first `L - tail` layers, then switch to expensive factored K=3 only for the last `tail` layers (implemented as `kprop_layer_means_tail()`, restarting the K=3 phase from the head's output mean/covariance treated as a fresh Gaussian). Measured across `tail` = 0, 4, 6, 8, 10, 12, 16, 20, 24 layers: `final_layer_mse` stayed flat at **~1.2e-4 to 1.3e-4 regardless of tail length** — i.e., adding an expensive K=3 tail barely moves the needle over plain K=2. Root cause (consistent with the paper's own error-scaling conjecture, MSE ~ `c_K(L/n)^K`): the head's K=2 pass already bakes in a whole-network-scale approximation error that a K=3 tail cannot retroactively correct — K=3 tracking only helps if it covers the layers where the correlations it's designed to capture actually build up, which is the **whole depth**, not just the end.

**Conclusion:** matching the leaderboard's ~1e-7 requires full-depth K=3 (or higher) cumulant propagation, which is right at (or slightly over) our FLOP budget with the current reference port's implementation — this needs real performance engineering (better cache reuse across the depth loop, tighter einsum paths, maybe a leaner "BASE" kind or an actually-quadratic-cost-reducing algorithmic tweak), not just a bug fix, to land reliably under budget. This is a good next research thread but is a bigger lift than a single session.

**Artifacts:** patched port saved at `scratchpad/kprop_port/` this session (not committed — reference code, not our submission). Includes `port_np/kprop_np.py` with the added `kprop_layer_means_tail()` driver (negative result, kept for reference).

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
