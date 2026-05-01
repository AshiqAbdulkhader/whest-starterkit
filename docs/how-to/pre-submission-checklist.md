# Pre-Submission Checklist

> [← Documentation](../README.md)

## 🎯 When to use this page

The minute before you click "submit" on AIcrowd. Run through these
checks; each one maps to a single command or a one-line confirmation.

## Correctness

- [ ] **`uv run whest validate --estimator estimator.py`** ends with a
      green `Status: success` panel. (Catches: wrong shape, non-finite
      values, broken `setup()`.)
- [ ] **`uv run whest run --estimator estimator.py --runner local --seed 42 --n-mlps 3`**
      produces a `primary_score` you recognize.
- [ ] **`uv run whest run --estimator estimator.py --runner subprocess --seed 42 --n-mlps 3`**
      produces a score within ~1% of the local-runner score above.
      (Catches: shared global state, RNG re-seed differences, imports
      that fail in clean processes — see [FAQ](../troubleshooting/faq.md#my-local-score-is-great-but-my-submission-scores-10x-worse--why).)

## Budget hygiene

- [ ] In the run report, **`per_mlp[i].budget_exhausted` is `false`** for
      every MLP. Any `true` means that MLP scored against zeros.
- [ ] **`per_mlp[i].time_exhausted`** and
      **`residual_wall_time_exhausted`** are also `false` (only relevant if
      you set `--wall-time-limit` or `--residual-wall-time-limit`).
- [ ] **`flops_used`** is comfortably under
      `flop_budget` — leaves headroom for the harder MLPs in the grader
      suite.

## Reproducibility

- [ ] **`requirements.txt`** lists every non-flopscope, non-whestbench
      import your estimator pulls in. `scipy`, `numpy`-only utilities
      etc. — anything you `import`. Test with
      `uv pip install --target /tmp/probe -r requirements.txt && rm -rf /tmp/probe`
      to confirm every name resolves.
- [ ] No filesystem reads from outside `SetupContext.scratch_dir`. The
      grader can't see your laptop.
- [ ] No network calls in `setup()` or `predict()`. The grader has no
      outbound network.
- [ ] No time-based seeds (`time.time()`, `os.urandom`, …). Use
      `numpy.random.default_rng(<fixed_int>)`.

## Sanity

- [ ] `predict()` returns the **post-ReLU** mean for **every** layer,
      shape `(mlp.depth, mlp.width)`. Off-by-one (returning depth+1 or
      depth-1 layers) is the most common silent bug.
- [ ] If you ship a `setup()`: it's idempotent and stays under the ~5s
      `setup_timeout_s`. Heavy precompute belongs in
      `SetupContext.scratch_dir`.
- [ ] No `print()` left in `predict()`. The grader runs many MLPs;
      stdout flooding is a reliable way to lose `residual_wall_time_s`.

## Final command

```bash
uv run whest package --estimator estimator.py -o submission.tar.gz
```

Inspect the artifact (`tar tf submission.tar.gz`) — it should contain
`estimator.py`, `requirements.txt`, and `metadata.json`. Upload to
[AIcrowd](https://www.aicrowd.com/) and move on.

## ➡️ See also

- [Stage 6: Package Your Submission](../getting-started/stage-6-package.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
- [Score Report Fields](../reference/score-report-fields.md)
