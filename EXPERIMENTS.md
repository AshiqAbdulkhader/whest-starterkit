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

### Round 3b: methodology fix and properly-powered sampler comparison

**The 15-MLP sweeps above were subset noise.** Discovered when the full-split extraction landed: over 1000 MLPs, `fullL1_diag234` (4.86e-6) is *worse* than plain whitening (4.51e-6), and mini/full splits contain *different MLPs* (different seeds — which also means `whest run --split mini` is a clean holdout for full-split-trained correctors). Re-ran all variants paired over 200 full-split MLPs (`scripts/compare_samplers_200.py`):

| Sampler | mean MSE (200 MLPs) |
|---|---|
| **mmL1 (antithetic + exact layer-1 moment match only)** | **3.87e-06** |
| whiten + mmL1 | 4.01e-06 |
| whiten (submissions #4/#5) | 4.39e-06 |
| mm + diag pinning layers 2-4 | 4.68e-06 |
| whiten + diag pinning 2-4 | 4.78e-06 |

**Verdict:** exact layer-1 moment matching *replaces* whitening (1.13x better); diagonal pinning at layers 2-4 hurts (mech bias amplified by the chaotic forward dynamics — consistent with the mid-depth rule). Shipped sampler = mmL1 only.

**Noise-floor decomposition** (two independent seeds, 30 MLPs): the whitened-MC error is **100% pure sampling noise** — systematic residual ≈ 0. Consequences: (a) the corrector's gain is optimal-shrinkage-toward-mech, with limited further headroom on these features; (b) `sem²` *overestimates* the true noise 2x (antithetic/whitening induce negative sample correlations); (c) below-floor score scales purely with per-sample variance — the leaders' 3.5x edge implies a structurally better estimator, not tuning.

**Radial stratification** (chi-quantile norm remapping): +2% only — norms already concentrate at width 256. Not worth shipping.

**Additional negative result — low-rank sampled tail** (`scripts/test_lowrank_tail.py`): deep-layer mech covariance is strongly rank-concentrated (rank-32 captures 94-97% of variance past layer 16), suggesting the deep-tail matmuls could run in coefficient space (`z = mu@W + c@(VᵀW) + diagonal-noise`) at ~2.7x lower per-sample cost, buying more samples. Measured: 14x bias blowup (3.6e-5 vs 2.6e-6 at equal N) — replacing the discarded orthogonal fluctuation with independent diagonal Gaussian noise destroys the true fluctuation structure that 24 layers of ReLU nonlinearity propagate, and breaks antithetic pairing. Consistent with the deep-pinning bias wall: He-init ReLU forward dynamics amplify mid-depth distributional perturbations, so any approximation injected at mid-depth costs more downstream than the variance it saves. **Rule of thumb established: approximate near the input (where exact moments exist) or at the output (where the corrector can learn the bias) — never in the middle.**

## Research round 4 (2026-07-10, post rank-50): chasing the leaders' variance edge

Reconfirmed via the grader API that submission #6 graded at 3.02e-7 (matches local closely). Leaderboard snapshot: top raw `final_layer_mse` spans 2e-7 (rank 1) to ~1.7e-6 (rank ~15-19), almost all sitting at the 0.1 multiplier floor like us — meaning the difference vs. us (raw ~3e-6) is purely variance/bias in the estimator, not budget usage. Two threads this round:

**Telescoping control variate — proved to be a mathematical no-op, not a bug.** `scripts/test_telescoping_cv.py`'s earlier "exactly 1.00x, identical to plain MC" result looked like a bug but is provably exact: the recursion `E_hat_l = gain_l·(Wᵀ E_hat_{l-1}) + mean_k[x_l,k − gain_l·z_l,k]`, when `E_hat_{l-1}` is itself the running estimate built the same way from the SAME batch, telescopes by induction to *exactly* `mean_k[x_L,k]` — the plain sample mean — for *any* choice of gain, at every layer. It only becomes real variance reduction if the injected "previous layer mean" is an INDEPENDENT, lower-variance (or exact) estimate rather than the batch's own running mean. That's exactly why mmL1 (layer 1's mean/cov are EXACT, zero-variance, not sample-derived) works while deeper pinning to the noisy/biased K=2 mech trajectory (`test_deep_momentmatch.py`) doesn't. General rule now confirmed twice: **only inject an estimate that is either exact or independently-sourced; injecting the same batch's own (possibly biased) running statistic changes nothing or hurts.**

**Marginal quantile matching at layer 1 — negative, redundant with mmL1.** `scripts/test_quantile_match.py`: rank-transforming each of the 256 layer-1 neurons to its EXACT closed-form marginal CDF (point-mass-at-zero + half-Gaussian, since z1 is exactly Gaussian) fixes every marginal moment, not just mean+variance — but combined with the existing affine mean+covariance match it's statistically indistinguishable (3.856e-6 vs 3.873e-6 over 200 MLPs). The noise mmL1 leaves on the table isn't in the marginal shape; matching mean+covariance already captures essentially all of it.

**Quasi-Monte Carlo (Sobol) — initially looked like a 1.30x win, corrected to ~1.0x (no real gain) after fixing a sample-count confound.** Long investigation, worth recording in full since it nearly shipped a bug:

1. `scripts/test_qmc.py` (first pass): scrambled Sobol vs pseudo-random MC, both "at `n_pairs=2750`", gave 2.75e-6 vs 3.58e-6 (1.30x). Two follow-ups to make this shippable without a `scipy` runtime dependency (not declared/locked in `pyproject.toml`, and the grader runs a restricted `flopscope-client` proxy per `docs/troubleshooting/common-participant-errors.md` — "local runs use the full flopscope; the grader uses the flopscope *client*... write all array code against `flopscope.numpy`... never reach for plain numpy") both failed to reproduce the gain:
   - `scripts/test_lhs.py`: pure-numpy Latin Hypercube (per-dimension stratification, no joint structure, no external tables) — **0.93x, actually worse** than plain MC.
   - `scripts/test_lattice.py`: a dependency-free Korobov lattice rule (single generating-vector modular construction, no tables) — **0.96x, no gain.** Lattice quality is extremely sensitive to the generating vector; a good one needs its own optimization (component-by-component construction), which is a research problem in itself.
   - Concluded a *real* Sobol implementation was needed. Extracted scipy's own precomputed direction-number matrix (`scipy.stats.qmc.Sobol()._sv` — plain data, harvested once offline) and wrote a from-scratch point generator (`scripts/sobol_np.py`), verified **bit-for-bit identical** to `scipy.stats.qmc.Sobol(scramble=False).random()` for both power-of-2 and arbitrary N. Ported the same algorithm to `flopscope.numpy` (only `bitwise_xor`/`right_shift`/`arange`/`where`, all confirmed present) and vectorized the point construction over bit-positions (~30 iterations) instead of the textbook sequential Gray-code walk over samples (~thousands of Python-level iterations, which would have repeated the K=3 cumulant port's residual-wall-time failure mode) — verified this vectorized form yields the *same point set* as the sequential walk (order differs, irrelevant for a mean), but **only at power-of-2 sample counts** (confirmed both empirically and via scipy's own runtime warning: "The balance properties of Sobol' points require n to be a power of 2").
2. **The catch, caught before shipping:** our own `flopscope.numpy` Sobol reproduced the 1.30-1.38x gain in early checks — but those checks (and the original `test_qmc.py`) rounded the Sobol sample count *up* to the next power of 2 above `2*n_pairs` (5500 → 8192, **49% more samples** than the plain-MC comparator at the same nominal `n_pairs`). Re-running with N *exactly* matched (both 4096) gave **1.04x** — statistically indistinguishable from noise. Re-tested with scipy's own Owen-scrambled Sobol at the same matched N: also ~no gain (in fact slightly worse than our simpler digital-shift randomization). **The entire apparent win was a sample-count accounting bug in the test harness, not a real effect.**
3. **Why this makes sense in hindsight:** QMC's discrepancy advantage over pseudo-random MC is strongest for smooth, low-effective-dimension integrands. A 32-layer ReLU forward pass is the opposite — piecewise-linear (non-smooth) with a kink at every neuron at every layer, and no reason to expect low effective dimension after 32 rounds of dense mixing. This is a known regime where QMC's theoretical edge is known to evaporate.

**Methodological lesson banked for future comparisons:** when a variance-reduction candidate requires rounding sample count to satisfy some structural constraint (power-of-2 for Sobol, etc.), always verify the comparator uses the *same* rounded N — round-up asymmetry silently inflates apparent gains by exactly the sample-count ratio, and that ratio (1.2-1.5x here) is uncomfortably close to the range of gains we're actually hunting for in this challenge.

## Research round 5 (2026-07-10): K=3 rank-truncation, tested and rejected

With the sampling side exhausted (round 4), attempted the remaining identified lever: bound the factored K=3 tensor's `O(L²)` rank growth via truncation, to fit full-depth K=3 propagation under budget (round 1 showed untruncated full-depth K=3 costs ~99-110% of the 2.72e11 budget).

**Instrumented the actual rank growth first** (patched port from round 1, `port_np/factor_k3_np.py`'s `FactoredTensor.add_factors_`): rank grows by ~256-512 per layer, reaching **23,552 by layer 32** at width 256. Tracing *why*: two of the `add_factors_` calls per layer include a `fac2 = np.eye(n) * 3` term — an exactly full-rank (rank-256) identity contribution, not accumulated noise. This is a structural fact about the algorithm (that term captures a specific diagram in the cumulant expansion that is inherently full-rank), not an implementation inefficiency.

**Tested two truncation schemes on depth=8** (where a trusted N=2e6-Monte-Carlo ground truth and the untruncated K=3 baseline — MSE 1.44e-6 — are cheap to compute):
1. **Random Gaussian projection** (shared random sketch matrix applied to all 3 CP modes, a standard JL-style rank-reduction trick): MSE degrades monotonically and badly — 7.65e-6 at cap=4000 (barely below the untruncated ceiling), 4.0e-5 at cap=2000, 7.0e-3 at cap=500. **5x worse even at the mildest cap tested.**
2. **SVD-based projection** (stack all 3 modes, take a shared top-singular-vector subspace — theoretically "smarter" since it's data-driven, not random): **worse still** — 2.9e-3 at cap=4000, 0.25 at cap=2000 (i.e., no better than the zeros baseline). Stacking all 3 modes into one SVD ignores the tensor's symmetric trilinear contraction structure, actively corrupting it more than a naive random sketch does.

**Verdict: rejected.** Both are decisive, order-of-magnitude failures, not marginal ones — this isn't a case for more tuning. Recovering the L² cost would require a properly tensor-decomposition-aware compression (e.g., an ALS-based CP-rank reduction that explicitly minimizes reconstruction error of the *actual* trilinear tensor, respecting its symmetric structure and treating the identity-derived full-rank component separately from the compressible part) — a genuine, nontrivial numerical-methods research problem in its own right, not an engineering afternoon, with no guarantee it would even work given how badly the two most natural naive approaches failed. Not pursued further given the effort/uncertain-payoff balance.

**Conclusion for this challenge, current state:** both identified major levers (deeper sampling-side variance reduction, and squeezing K=3 under budget via rank truncation) have now been tried and closed off with well-documented negative results. Submission #6 (mmL1 sampler + learned corrector, graded 3.02e-7, rank 50) stands as our best validated result. Closing the remaining ~3x gap to the leaderboard's top 20 (raw MSE ~2e-7 to 1.7e-6) likely needs either a genuinely new algorithmic idea not yet identified, or the harder tensor-decomposition research above.

## Research round 6 (2026-07-10): checked ARC's official repo + one more paper-sourced idea

**Explored [alignment-research-center/mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation) directly** (the paper's reference implementation, updated 2 days before phase 1 launched). Confirms rather than contradicts round 5:
- **The paper's own Appendix D says depth-scaling beyond sampling is an open problem**: "This depth scaling is worse than Monte Carlo sampling, whose error does not increase with depth. We believe that a sample-free algorithm can be developed whose error also does not increase with depth either, but we leave this problem to future work." At L=32 this is a known, acknowledged limitation of the whole cumulant-propagation family — not something a cleverer implementation of the *same* method fixes.
- ARC themselves built a fancier tensor-network reformulation (`src/mlp_kprop/symb/`, using `quimb` + a `pruning.py` for smart diagram elimination) that could in principle use better contraction paths than the layer-by-layer method we tried to truncate in round 5. Their own README: *"this can be asymptotically faster... though, in practice, the constants are too large to be practical."* They tried the smarter approach and rejected it themselves — strong outside confirmation that round 5's rank-truncation dead end wasn't from us missing an obvious fix.
- The one real bugfix the repo received recently (variance-clamping "for deeper MLPs", merged 2026-07-07) is already present in our patched port independently.

## Research round 7 (2026-07-10): asked Codex for a fresh idea

Ran `/codex` (consult mode, `gpt-5.1-codex`, medium reasoning effort) with the full research history from EXPERIMENTS.md and the exact current implementation as context, explicitly asking for something genuinely different from everything already tried. (Needed `npm install -g @openai/codex` first — the installed CLI, 0.98.0, was too old for its own default model.)

**Codex's idea: output-side Rao-Blackwellization of the final layer only.** Explicitly designed to avoid both prior failure modes: unlike the rejected low-rank sampled tail (which approximated a *mid-depth* layer, letting 16-24 remaining ReLU layers amplify the resulting bias), and unlike cumulant extrapolation (which compressed the *whole* final pre-activation distribution to 2-3 scalar moments), this touches only the very last operation and keeps the *dominant* sampled structure exact:
1. Run the current mmL1 sampler through layer 31 unchanged (exact, as now).
2. Decompose the penultimate activation `H`'s sample covariance into a top-rank eigenspace `U` and an orthogonal residual.
3. Split each final-layer weight column `w_j = U·a_j + w_res_j`. The low-rank part of the pre-activation (`mean(H)·w_j + (H_centered·U)·a_j`) is kept exact, per-sample — no approximation.
4. Only the orthogonal residual (`H_res·w_res_j`) is treated as independent Gaussian noise with variance `τ_j² = w_res_j·C·w_res_j` (exact magnitude, only the *shape* is approximated).
5. Replace the raw `ReLU(pre-activation)` with its analytic conditional expectation over that residual: `E[ReLU(z)|m_ij] = m_ij·Φ(m_ij/τ_j) + τ_j·φ(m_ij/τ_j)` — a genuine Rao-Blackwellization, provably variance-reducing *if* the residual is close enough to Gaussian.

**Tested rigorously** (`scripts/test_rao_blackwell.py`): swept rank ∈ {16,32,48,64,96,128} × {plain PCA, output-aware PCA(C·W·Wᵀ·C)} on 100 full-split MLPs. **Result: never beats the raw baseline.** At high rank (64-128) it converges to exactly 1.00x (expected — the residual shrinks toward zero, so the method degenerates to the raw estimator with nothing left to smooth). At low rank (16-32) it's actively *worse* (0.92-0.99x) — monotonically worse as rank decreases, i.e. the more aggressively you try to Rao-Blackwellize, the worse it gets. No sweet spot.

**Diagnosis, fed back to Codex:** the orthogonal-residual-is-Gaussian assumption doesn't hold well enough here, even restricted to a small, last-operation-only, dominant-subspace-preserving approximation. This is the *same* underlying obstacle as every other failed idea this session: any approximation of the sampled distribution — at the input (beaten by exact layer-1 moments), mid-depth (catastrophic), or now the output (this test) — discards real non-Gaussian structure that raw sample averaging already captures for free, and that lost signal outweighs whatever variance the approximation saves. The one thing that *did* work all session was substituting an *exact* closed-form quantity (layer 1's true Gaussian moments) for part of the sampling, not an approximation of any kind.

**Follow-up idea, also tested and rejected — exact zero-mean quartic Hermite control variates.** Fed the Rao-Blackwell result back to Codex (same session, resumed) and asked explicitly for something that substitutes an *exact* quantity rather than an approximation, the way layer-1 moment matching does. Codex's answer: for a fixed direction `a_m` in INPUT space, `t_m = u·a_m` is exactly `N(0, ||a_m||²)` regardless of the rest of the network (linear combination of iid Gaussians) — so the standardized 4th Hermite polynomial `He_4(t_m/s_m)` has *exact* zero mean, a free control variate that needs no extra sampling. Crucially different in construction from the earlier-failed `test_h1_cv.py` (which used a large 256-dim basis of *post-ReLU* linear features and got swamped by `K/N` regression noise): this uses a *small* (M=8-32) basis of *4th-order* input-space features, explicitly targeting the noise order that survives antithetic pairing (kills odd orders) + exact layer-1 matching (kills up to 2nd order) — i.e. even-order noise beyond the 2nd moment.

Tested (`scripts/test_quartic_hermite_cv.py`) with cross-fitted (split-batch) regression to avoid overfitting bias, sweeping M ∈ {8,16,32} and three direction choices (random orthonormal, top singular vectors of `W1`, and an output-aware sensitivity basis built by propagating an approximate per-layer linear gain through the mech trajectory). **Result: decisively worse at every setting** — even the best config (`w1_svd`, M=8, the smallest/most favorable) was **~3.8x worse** than raw mmL1 averaged over 15 MLPs (1.16e-5 vs 3.07e-6), and every config got monotonically worse with larger M, mirroring the original `h1_cv` failure pattern almost exactly. The true correlation between an early-layer quartic feature and the final-layer output 31 nonlinear layers downstream is apparently too weak for the cross-fitted regression to estimate reliably — the fitted coefficients are mostly capturing sampling noise in the correlation itself, and using them to "correct" the estimate adds more variance than the (real but tiny) captured signal removes.

**Where this leaves the "exact substitution" idea:** two exact constructions were tried at the input/layer-1 boundary (moment matching — works; quartic Hermite CV — fails) and the Rao-Blackwell attempt at the output also fails. Codex's own assessment after this result: "there is no tractable new exact closed-form substitution at layer 2+ without summing over ReLU gate regions" — i.e., the reason layer 1 works is that it's the *only* point in the network where the pre-activation distribution is exactly, unconditionally Gaussian; every layer after that is conditioned on the ReLU gating pattern of all prior layers, which has no closed form. This appears to be a hard structural boundary, not a tuning problem.

**New idea sourced from paper Appendix G, tested and rejected — "Monte Carlo sampling with cumulant extrapolation."** The paper describes this as a strong baseline for *low-probability* (rare-event) estimation: instead of averaging raw samples of an indicator function, estimate the pre-activation's mean/variance/third-cumulant via sampling (cheap, smooth quantities), then apply the analytic Hermite/Edgeworth-corrected formula for the nonlinearity's expectation. Adapted to our task (`scripts/test_cumulant_extrapolation.py`): estimate mean, variance, and sample third central moment of the LAST layer's pre-activation `z_32` from the existing mmL1 batch (free — same batch, just a different reduction), then use `E[ReLU(Z)] ≈ E_Gaussian[ReLU(Y)] + (1/6)·b₃(μ,σ)·κ₃` (b₃ derived from the paper's own `relu_wick_coef` formula) instead of the raw sample average of `ReLU(z_32,k)`.

Result (150 full-split MLPs): **0.96x — slightly worse than just averaging the raw samples**; using only mean+variance (no skew correction at all) is markedly worse still (0.77x). **Why this makes sense:** cumulant extrapolation's power in the paper comes specifically from avoiding the *explosive relative variance of an indicator function* in the rare-event regime — reconstructing from smooth low-order moments beats sampling a function that's mostly zero. Our target (`E[ReLU(Z)]`) is already a *smooth, bounded-variance* quantity in the normal-probability regime, where the raw sample average is already close to a minimum-variance unbiased estimator; truncating to a 3-moment (Edgeworth) reconstruction only throws away real 4th-order-and-up signal that the raw average captures for free. The technique is real, just not for this shape of problem.

## Research round 8 (2026-07-10): community intel — an independent, convergent research writeup

Swept the AIcrowd Discourse forum for participant discussion (not just official announcements) and found the **Algorithmic Contribution Prize** thread ([topic 18041](https://discourse.aicrowd.com/t/algorithmic-contribution-prize-guidelines-how-arc-judges-these-prizes-discretion-technical-writeups-llm-usage/18041)) referencing a public Phase-1 writeup by participant **pscamillo**: *"[Phase 1 write-up] Characterizing a systematic scale bias in the Gaussian-closure estimator"* ([topic 18063](https://discourse.aicrowd.com/t/phase-1-write-up-characterizing-a-systematic-scale-bias-in-the-gaussian-closure-estimator-submission-314331/18063), 12-page PDF, submission #314331). This is a genuinely rigorous, independent research program — worth documenting in full because it's the closest thing to ground truth on "what does and doesn't work" that exists outside our own testing.

**Their approach and result (not directly applicable to us, but important context):** the K=2 covariance-propagation estimator systematically *overestimates* the final-layer mean by a stable multiplicative factor — optimal correction **0.9916 ± 0.0027** across 100 official networks, cross-validated to **0.9921 ± 0.0001**, generalizing cleanly to held-out networks. A single free scalar multiply cuts K=2's raw MSE from 8.37e-5 to 2.59e-5 (~3x). Graded at adjusted 2.45e-6 — worse than our submission #6 (3.02e-7), because it's a *pure mechanistic* estimator with zero sampling, not competitive with a good sampler. **Doesn't transfer to our pipeline**: our estimator uses actual Monte Carlo samples (not the raw K=2 formula) for the scored output, and MC sampling is close to unbiased by construction — this specific bias is an artifact of the Gaussian-closure *approximation*, which we don't rely on for the final answer.

**Why the bias exists (mechanistically localized, not just curve-fit):** measured directly against the organizers' own `mlp_kprop` reference — the optimal per-layer correction factor is exactly 1.000 at layer 1 (no bias, matches our own finding that layer 1 is exactly Gaussian) and decreases monotonically with depth to ~0.992 by layer 32. Ruled out skewness as the cause (measured skewness ≈0, symmetric) — it's driven by **excess kurtosis accumulating monotonically with depth (0 → +0.40)**, i.e. a k≥4 non-Gaussian effect, consistent with the "irreducible kink variance" language both they and we independently arrived at.

**Their falsification map is nearly a superset of ours — strong convergent validation:**
| Their finding | Matches our finding |
|---|---|
| Third-cumulant (k=3) propagation doesn't beat the free scalar correction at L=32, despite 500x the cost — ran the *organizers' own reference implementation* directly | Matches round 1: full-depth K=3 costs ~99-110% of budget for little gain |
| "CP-rank cap of degree-3 cumulant... gain from capping is regularization against depth-accumulated error, not compression" | Matches round 5: our rank truncation attempts failed decisively, and interestingly they frame *any* apparent gain from capping as an artifact, not real compression working |
| "Subspace projection of degree-3: signal subspace is emergent (not from W's SVD) and rotates fast" | Matches round 7: our Rao-Blackwell attempt used PCA/output-aware (SVD-derived) subspaces and got nothing — they independently found the same static-SVD-subspace approach doesn't align with the real (rotating) signal structure |
| "Hybrid MC + control variate: per-sample variance reduction only ~1.1x; the ReLU knee is not capturable by low-order moments" | Matches rounds 4 & 7: our h1_cv and quartic Hermite CV attempts both failed for the same reason |
| Exact-moment Edgeworth correction (using an *independent ground-truth higher-moments dataset*, so no estimation error) still doesn't help | Matches round 6: our cumulant-extrapolation test also failed |
| Learned nonlinear corrector on the final layer: 0.95x (hurts), audited against shuffled-label controls (real vs shuffled both give test R²≈-0.06 — no signal) | We get modest real gains (1.2-1.4x) from our corrector, likely because ours targets the *sampler's* residual (mostly the K=2 baseline blend) rather than the *raw K=2 estimator's* residual directly — a narrower target |
| **Key new data point**: predictive signal in the K=2 residual **decays exponentially with depth, τ ≈ 5.1 layers**, and is statistically indistinguishable from zero by ~layer 15, vanishing entirely at the scored layer 31 | New to us — a precise, well-validated number for *why* nothing works at the final layer: there's a real, measurable "horizon" past which no cheap correction sees anything |
| **Explicitly cites an "orthogonal" Phase-1 competitor** (QMC + Rao-Blackwellized exact first layer — sounds like a similar family to our own mmL1 approach) reporting the residual is **"irreducible kink variance" that control variates and analytic correctors do not close at depth** | Nearly verbatim our own round-4 conclusion ("MC error is 100% pure sampling noise, no exploitable systematic residual") — independent confirmation from a completely different research program |

**Strategic intelligence (their §8, directly useful):** "The Phase-1 score leaders sit near adjusted 1.7e-7 using ~10% of the FLOP budget (the compute multiplier at its floor). This indicates the binding constraint at the top is **accuracy, not compute** — a qualitatively different idea than cost-optimized cumulant propagation." Consistent with everything we've found: nobody is winning via a clever budget trick, they're winning via genuinely lower per-sample variance. The writeup's own hybrid MC+CV only got ~1.1x — so the leaders' edge over *both* of us remains unexplained. They describe the leading approach class only vaguely as "variance-reduced samplers (randomized QMC, Rao-Blackwellized exact layers) operating at several times the efficiency of plain Monte Carlo" — the same broad family we're in, just apparently executed further than either of us has managed.

**Net effect on our research posture:** two independent, methodologically different research programs (ours: empirical sampler engineering; theirs: mechanistic bias decomposition) converge on the same boundary using almost entirely non-overlapping techniques. This is real evidence we're near a genuine wall for moderate-effort ideas — but it does *not* explain the leaders, so the search continues. Two techniques neither research program has tested: **multilevel Monte Carlo** (a cheap correlated coarse estimator, e.g. a width-subsampled sub-network, control-variated against the full network) and **importance sampling** (changing the input measure itself, not just reweighting samples to match its true moments). Proceeding to test MLMC next.

**Multilevel Monte Carlo — tested, killed immediately by the same chaos.** Built a "coarse" network from the same weight matrices with the internal width bottlenecked (256 → n_c ∈ {32,64,128,192}, keeping the true 256-dim input and output; layer 1 maps 256→n_c, layers 2-31 run n_c→n_c cheaply, layer 32 maps n_c→256) — cheaper by `(n_c/256)²` per bottlenecked layer, and sharing input randomness (common random numbers) with the true "fine" network, which is the standard MLMC construction. The idea only pays off if `Var(fine − coarse) ≪ Var(fine)`, i.e. fine and coarse need to be correlated.

They aren't. `scripts/test_mlmc.py`, 20 full-split MLPs: median correlation between fine and coarse final-layer outputs is **0.000 at every n_c tested, including n_c=192** (only 64 of 256 neurons removed) — and `Var(fine−coarse)/Var(fine) = 1.000` exactly, meaning subtracting the coarse network removes *zero* variance; they're statistically independent for practical purposes. This is the same chaotic-decorrelation story as every mid-network approximation this session (round 4's low-rank tail, round 1's layer-adaptive K=3 routing): the depth-32 ReLU map is sensitive enough to *any* structural change to the computation — even keeping 75% of the width — that trajectories decorrelate almost completely by the final layer. A width-bottlenecked coarse model is, in this sense, just another mid-network approximation wearing an MLMC costume, and it dies for the same reason. Killed in under 10 minutes of testing, cheaply ruling out the whole naive-MLMC direction; a coarse model built from an *exact* quantity (e.g. mech K=2 as the coarse level) reduces to the "hybrid MC + control variate" construction the independent writeup already tested and found weak (~1.1x) — not re-tested here given that prior result.

### 2026-07-10 — Submission #6: mmL1 sampler + corrector retrained on it

- **What:** Submission #5's pipeline with the sampler fix from the 200-MLP comparison above (whitening removed, replaced by antithetic + exact layer-1 moment matching only) and the corrector retrained on features extracted with the new sampler (1000-MLP full split). Same 20-feature / 2×96-tanh architecture. Git tag: `submission-6`.
- **Local result** (full mini split, 100 MLPs — disjoint from the full split the corrector trained on, so this is a clean holdout): `adjusted_final_layer_score` **3.83e-07** (raw 3.52e-06, multiplier 0.109) vs submission #5's 3.96e-07 (raw 3.95e-06) — modest but real, ~1.12x on raw MSE, roughly in line with the sampler's 1.13x gain alone (corrector gain was smaller in-harness than offline, likely because in-harness noise/seed differs from the offline holdout's).
- **Leaderboard result:** submission id **315633** — https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/315633.
- **Also ruled out this round** (`scripts/test_bayes_shrinkage.py`): per-MLP Bayesian shrinkage between the mech (K=2) prediction and the MC estimate, both scalar (global weight) and full Wiener-filter (per-MLP covariance-aware) forms, using a fitted mech-error variance τ². Both are *much worse* than MC alone (scalar: 6.1e-6, Wiener: 5.9e-5 vs MC's 4.0e-6) — the mech (K=2) error variance (~7.3e-5) is ~18x the MC noise variance, so any shrinkage toward it is actively harmful; the per-neuron learned corrector avoids this because it conditions on richer features than a single scalar/global covariance, rather than blindly blending toward the weak baseline.
- **Next:** rank 71 → the leaders' ~3.5x edge in adjusted score is a genuinely lower per-sample variance, not tuning (established this round: our MC error is 100% pure sampling noise, no exploitable systematic residual on current features). Ideas not yet tried: higher-order (cubic+) exact moment matching at layer 1 using the known third/fourth cumulants of a chi-squared-adjacent quantity; per-neuron adaptive sample allocation (spend more antithetic pairs on high-variance neurons within the same total FLOP budget); investigating whether leaders are using a fundamentally different sampling substrate (e.g. quasi-MC in a rotated/whitened+low-discrepancy space, or full K=3 cumulants restricted to just the final layer's mean, which needs far less compute than the full-depth K=3 we ruled out in round 1).

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
