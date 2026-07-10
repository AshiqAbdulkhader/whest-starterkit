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

**Follow-up: profiled where the 3.0e11 FLOPs actually go** (`BudgetContext.summary_dict(by_namespace=True)`, full depth=32, K=3 factored/SIMPLE): split almost exactly 50/50 between `einsum` (1.49e11 across 3417 calls) and `matmul` (1.49e11 across just 93 calls — ~1.6e9 FLOPs/call on average). This is **not wasted/redundant computation** we can cache or short-circuit away — the `FactoredTensor` representation's rank grows by a fixed increment every layer (`add_factors_` concatenates new factors each layer and is never truncated), so the per-layer `contract_W` matmul cost grows roughly linearly with depth-so-far, and the cumulative sum over 32 layers is the `O(L²)` the paper predicts. Confirmed this is genuine, structural cost inherent to the algorithm as implemented, not an inefficiency in our patch.

**What it would take to go further:** bound the factored tensor's rank growth (periodic low-rank compression/truncation of the accumulated `FactoredTensor` factors, e.g. via truncated SVD every few layers) so cost stops scaling quadratically with depth. This is a genuine new numerical algorithm on top of the ported code, not a bug fix or perf tweak — it trades some accuracy for a lot of cost and needs its own careful validation (this is exactly the "Low-rank covariance" idea already flagged in [docs/how-to/algorithm-ideas.md](docs/how-to/algorithm-ideas.md), generalized to K=3's factored tensor). Scoping this as its own follow-up rather than rushing it into this session.

**Artifacts:** patched port saved at `scratchpad/kprop_port/` this session (not committed — reference code, not our submission). Includes `port_np/kprop_np.py` with the added `kprop_layer_means_tail()` driver (negative result, kept for reference).

## Research round 2 (2026-07-10): how the leaders actually do it

Web research turned up the decisive hints:

1. **ARC's own update** ([LessWrong announcement](https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c/announcing-the-arc-white-box-estimation-challenge)): warm-up winners were "variants on the factorized 3rd cumulant propagation algorithm", and — crucially — "most top submissions combined **learned networks that consume the cumulant estimates as features**". Training is off-budget; only `predict()` FLOPs count. `examples/04_shipped_weights.py` shows the shipping mechanism (`flopscope.Module` → pickle-free `.npz`, 0 FLOPs to load in `setup()`).
2. **ARC admits the depth weakness**: "Our existing algorithms scale poorly with depth, and so we expect there to be significant room for improvement" — phase 1 went to depth 32 deliberately to break pure cumulant propagation. Matches our finding that full-depth K=3 eats the whole budget.
3. **[galfaroi's public repo](https://github.com/galfaroi/Can-You-Predict-a-Network-Without-Running-It-)**: whitened antithetic Monte Carlo (antithetic pairs `[u; -u]` + folding the empirical-covariance inverse-sqrt into the first weight matrix) scored **3.36e-7 adjusted** — ~1.3x behind the then-leader — with *no learned component at all*.
4. **Scoring economics insight**: the 10% multiplier floor means anything up to 2.72e10 FLOPs is "free" (score multiplier is 0.1 regardless). Our K=2 submission used only 0.6% — we were leaving ~15x more free compute on the table. Antithetic sampling kills all odd-order MC noise; whitening kills all quadratic noise (empirical covariance is exactly identity ⇒ any quadratic form's sample average equals its expectation).

**Measured locally (10 mini MLPs, phase-1 shape):**
| Estimator | raw final-layer MSE | notes |
|---|---|---|
| K=2 covariance propagation | 8.4e-5 | submission #3 |
| antithetic MC, 2750 pairs | 1.03e-5 | odd-noise cancellation |
| whitened antithetic MC, 2750 pairs | 4.45e-6 | +2.31x from whitening |
| **hybrid in-harness (whest run)** | **2.22e-6** | adjusted **2.52e-7** at mult 0.113 |

Learned corrector status: concept validated on mini (2.6x over K=2 from 14 features/100 MLPs; MC-noise is *not* learnable — must be reduced at the source, hence whitening). Full-split feature extraction (1000 MLPs, ~26k neuron-rows... 256k rows) running; corrector will ship as submission #5.

## Research round 3 (2026-07-10): moment-matched sampling (post-rank-71 push)

Grader confirmed our scores match local (315616 = 3.807e-7, 315622 = 3.468e-7 graded) — rank 71 just means the field is packed between 9e-8 and 3.5e-7. Needed: 4x+ more variance reduction. Findings:

1. **First-layer regression control variates fail** (`scripts/test_h1_cv.py`): using centered basis functions of z1 (exactly Gaussian ⇒ exact expectations) as cross-fitted regression CVs *loses* to whitening — the K=256-basis regression noise (K/N ≈ 10-28%) eats the gains, and h1's linear content is already covered by antithetic (odd) + whitening (quadratic).
2. **Layer-1 moment matching wins** (`scripts/test_h1_momentmatch.py`): z1 is exactly Gaussian, so both E[h1] *and* Cov[h1] have closed forms (bivariate Gaussian ReLU formula). Affinely renormalizing the sampled batch so its empirical mean/cov exactly equal the analytic ones kills all noise entering through h1's first two moments — **1.23x** over whitening alone, and it *replaces* whitening (combining both is worse than matching alone).
3. **Deep pinning to mech targets has a bias wall** (`scripts/test_deep_momentmatch.py`): matching layers 1..T against the K=2 mech trajectory (full-cov or diagonal) improves through T≈2-4, then mech bias dominates (T=8: 3.8e-5; T=32: 7.8e-5 ≈ pure mech error). Partial (shrunk) pinning at deeper layers also fails (`test_hybrid_momentmatch.py`).
4. **Winner: `fullL1_diag234`** — exact full-covariance match at layer 1 + diagonal (mean+var) pinning at layers 2-4 to the K=2 trajectory: **2.78e-6** (vs whitening 4.45e-6, **1.60x**) at ~+2e9 FLOPs. In-harness without corrector: raw 3.16e-6 on 15 mini MLPs.
5. Bonus: the K=2 pass now uses the exact bivariate ReLU covariance at layer 1 (previously the gain approximation there too).

Corrector v3 retraining on re-extracted full-split features (moment-matched MC), plus a second extraction pass at shifted MC seeds for noise-augmentation. Target: raw ~2.2-2.5e-6 → adjusted ~2.3e-7.

**Additional negative result — low-rank sampled tail** (`scripts/test_lowrank_tail.py`): deep-layer mech covariance is strongly rank-concentrated (rank-32 captures 94-97% of variance past layer 16), suggesting the deep-tail matmuls could run in coefficient space (`z = mu@W + c@(VᵀW) + diagonal-noise`) at ~2.7x lower per-sample cost, buying more samples. Measured: 14x bias blowup (3.6e-5 vs 2.6e-6 at equal N) — replacing the discarded orthogonal fluctuation with independent diagonal Gaussian noise destroys the true fluctuation structure that 24 layers of ReLU nonlinearity propagate, and breaks antithetic pairing. Consistent with the deep-pinning bias wall: He-init ReLU forward dynamics amplify mid-depth distributional perturbations, so any approximation injected at mid-depth costs more downstream than the variance it saves. **Rule of thumb established: approximate near the input (where exact moments exist) or at the output (where the corrector can learn the bias) — never in the middle.**

## Log

### 2026-07-10 — Submission #5: + learned corrector trained on the full public split

- **What:** Submission #4's pipeline plus the learned per-neuron corrector, shipped as a folder submission (`submission/` = `estimator.py` + `corrector.npz`). Corrector: 20 features per final-layer neuron (K=1/K=2 predictions, pre-activation stats, off-diagonal covariance strength, MC estimate/SEM/disagreement, 4-layer mean trajectory) → 2×96 tanh MLP → residual from the whitened-MC estimate. Trained on the full public split (1000 MLPs × 256 neurons = 256k rows, 15% MLP-grouped holdout, zero-init output layer so it can only improve on the MC baseline). Git tag: `submission-5`.
- **Local result** (full mini split, 100 MLPs): `adjusted_final_layer_score` **3.96e-07** (raw 3.95e-06, multiplier 0.1005) vs **4.86e-07** (raw 4.84e-06) for the identical pipeline without the corrector — 1.23x, matching the offline grouped-holdout ratio exactly (3.91e-6 → 3.17e-6).
- **Leaderboard result:** submission id **315622** — https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/315622. Expected ~rank 15-20 (leader 9.16e-8).
- **Learned / negative results this round:**
  - **Telescoping layerwise control variate is provably redundant with antithetic sampling** — with antithetic pairs the input sample mean is exactly zero, and the linear CV telescopes to the identity (verified: bit-identical output). Ascender's 4x CV gain was measured *without* antithetic inputs.
  - **Scrambled-Sobol QMC doesn't help** at 256 dims / 2750 points (0.86x — slightly worse than pseudo-random).
  - **MC noise is not learnable** — the corrector's val loss ≈ 0.69 of residual variance means it removes the mech-predictable ~31% (whitening bias + shrinkage) and the rest is genuine sampling noise.
  - **Scoring-economics dead end**: beyond the 10% floor, MSE ∝ 1/N cancels multiplier ∝ N exactly — extra samples are score-neutral. The leaders' edge at multiplier 0.26 implies their per-sample residual variance is ~6x lower than ours (0.0028 vs 0.017), i.e. they have a structurally better variance-reduction or estimation scheme, not just more compute.
- **Next ideas:** quartic+ control variates (antithetic+whitening kill orders 1-3; the remaining noise is 4th-order+), learned control variates with analytically-known expectation, partial-depth sampling with mech-corrected restarts, K=3 features for the corrector.

### 2026-07-10 — Submission #4: whitened antithetic MC + mechanistic hybrid

- **What:** Complete estimator rewrite. `predict()` now runs (a) K=1+K=2 propagation (features + non-final-layer rows), (b) whitened antithetic Monte Carlo (2750 pairs, `C^{-1/2}` folded into layer-1 weights, seeded from `mlp.seed`), and (c) an optional learned per-neuron corrector loaded from `corrector.npz` (not yet shipped in this submission — falls back to plain whitened MC). ~2.4e10 analytical FLOPs ≈ 8.8% of budget → at the 0.1 multiplier floor on the grader. Git tag: `submission-4`.
- **Local result** (10 mini MLPs): `adjusted_final_layer_score` **2.52e-07**, raw `final_layer_mse` 2.22e-06, multiplier 0.113 (local residual-time artifact; analytical is 8.8%).
- **Leaderboard result:** submission id **315616** — ~33x better than submission #3; would sit around rank 12-15 on the current leaderboard (leader: 9.16e-8).
- **Learned:** (1) The multiplier floor makes ≤2.72e10 FLOPs free — use all of it. (2) Antithetic + whitening are the two exact variance-annihilation tricks (odd orders + quadratic). (3) MC noise cannot be regressed away post-hoc — variance reduction must happen at the sampling stage; learned models are for the *mechanistic bias*, not the noise.
- **Next:** submission #5 = this + trained corrector (in progress on the 1000-MLP full split).

### 2026-07-10 — Submission #3: covariance propagation (full off-diagonal)

- **What:** Replaced diagonal mean propagation in `estimator.py` with full covariance propagation — tracks the (width x width) covariance matrix through each ReLU layer using the "gain" approximation for off-diagonal terms and the exact ReLU marginal variance on the diagonal, ported from `examples/03_covariance_propagation.py`. Git tag: `submission-3`.
- **Why:** Safe, already-validated incremental win while K=3 cumulant-propagation performance work continues (see research section above) — ~11x better raw MSE than mean propagation, still <1% of budget.
- **Local result** (`whest run --split mini --runner local`, public `mini` split, 100 MLPs, 256×32 @ 2.72e11 budget):
  - `adjusted_final_layer_score` = **8.37e-06** (vs 9.48e-05 for submission #2 — ~11x better)
  - `final_layer_mse` = 8.37e-05, `all_layers_mse` = 5.57e-05
  - Best MLP 1.80e-06, worst MLP 3.27e-05
  - FLOPs used: 1.62e9/MLP → 0.8% of budget → multiplier still floored at 0.1
- **Leaderboard result:** submission id **315610** — https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/315610. Still ~90x short of the leaderboard's ~1e-7 (see research section above); K=3 cumulant propagation is the path to close that gap, pending the budget-fit work.
- **Next:** Finish squeezing full-depth K=3 cumulant propagation under budget (cache-reuse/perf work on the patched `mlp_cumulant_propagation` port).

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
