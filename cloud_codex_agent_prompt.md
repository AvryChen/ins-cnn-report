# ins-cnn Cloud Codex Agent Prompt

You are running on the remote training machine for `AvryChen/ins-cnn`.

## Operating Rules

- Work in `~/work/ins-cnn`.
- Keep secrets out of git. Never print or commit proxy nodes, subscriptions, tokens, private keys, or API keys.
- Use `~/anaconda3/envs/torch2.4_cuda12.1/bin/python` for Python training.
- Use `~/.local/bin/julia` for Julia/Sunny work.
- Prefer scripts already in this repo over ad hoc commands.
- Keep generated datasets/checkpoints in `data/` and `runs/`; do not commit those folders unless a report copy under `docs/` is intentionally produced.
- Publish public reports only through `AvryChen/ins-cnn-report`; if the deploy key is not ready, skip public publish and record that fact.

## Current Scientific Goal

We are fitting La3Ni2O7 / LNO327 INS spin-wave spectra in reverse:

- Sunny.jl generates synthetic Fig.3 path spectra and Fig.2/4D maps.
- CNN/4D CNN models infer `[SJc, SJ1a, SJ1b, SJ2, SA]`.
- Weak peaks, shoulders, and intensity redistribution matter. Do not optimize only for bright-peak regions.

## Current Baseline

The primary line is **pure-wide n1024 4D inverse fitting**, not the smaller
`cloud_fwhm10_n512` diagnostic suite. The user is referring to the historical
best result:

```text
pure-wide n1024 scratch + old seed2/seed3 per-parameter ensemble: 2.3839%
```

The historical best artifacts are documented in:

```text
docs/lno327_ai_handoff.html
docs/lno327_cnn_weakpeak_summary.md
PROJECT_SUMMARY.md
scripts/lno327_write_4d_final_report.py
scripts/lno327_write_training_visual_report.py
scripts/lno327_write_data_parameter_report.py
```

The target dataset is:

```text
data/lno327_inverse_4d_wide_extra_augtrain_n1024_q160_e180_h81_k121_l5.h5
```

This dataset is not guaranteed to exist on the cloud machine because `data/`
and `runs/` are intentionally not tracked. If it is missing, rebuild it from
the repo scripts instead of falling back to the smaller FWHM10 n512 suite:

```bash
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_generate_wide_extra_augtrain.py \
  --root . \
  --julia ~/.local/bin/julia \
  --base-samples 512 \
  --samples 512 \
  --base-chunk-size 32 \
  --chunk-size 32 \
  --base-max-parallel 10 \
  --max-parallel 10 \
  --force
```

Before launching generation, check for an existing active run:

```bash
pgrep -af 'lno327_generate_wide_extra_augtrain|lno327_generate_inverse_dataset.jl' || true
```

If an active pure-wide n1024 generation is already running, monitor it instead
of launching a duplicate. The 12-core cloud CPU should be used through chunked
multi-process generation, but do not exceed 10 concurrent Julia processes
unless memory measurements show a clear margin.

The FWHM=10 meV n=512 cloud suite is a secondary diagnostic line:

```text
data/lno327_inverse_4d_wide_fwhm10_n512_q160_e180_h81_k121_l5.h5
```

Known completed cloud suite:

```text
runs/cloud_fwhm10_n512/
```

The current suite compares:

- Fig.3-only baseline
- gated4d
- residual4d warm-started from Fig.3
- head-only cooldown
- gated/residual validation-selected ensemble

## Your Job

1. Check the state:

```bash
cd ~/work/ins-cnn
git status --short
git pull --ff-only origin main
bash scripts/cloud_netcheck.sh || true
nvidia-smi
```

2. If Julia dependencies are missing, fix them yourself:

```bash
JULIA_DEPOT_PATH=$PWD/.julia-depot ~/.local/bin/julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

3. Restore the pure-wide n1024 line if needed:

- If `data/lno327_inverse_4d_wide_extra_augtrain_n1024_q160_e180_h81_k121_l5.h5`
  exists, inspect it and continue.
- If it is missing and no generator is currently running, run
  `scripts/lno327_generate_wide_extra_augtrain.py` with
  `--base-chunk-size 32 --chunk-size 32 --base-max-parallel 10 --max-parallel 10`.
- If a generator is already running, monitor PID/log progress and resource use
  instead of starting a second copy.
- If generation fails, diagnose and fix the root cause: Julia deps, CIF path,
  Sunny/HDF5 issues, memory, disk, or script assumptions.
- Keep generation logs under `runs/lno327_4d_task/`.

4. Continue from the best known recipe, with freedom to adapt:

```text
gated4d scratch seed -> residual4d scratch seed -> validation-selected ensemble
```

Baseline targets to reproduce and then beat:

```text
scratch gated4d seed20260705: 2.8056%
scratch residual4d seed20260706: 2.8354%
scratch gated + residual per-param ensemble: 2.4242%
scratch + old seed2/seed3 per-param ensemble: 2.3839%
```

If old checkpoints are absent, rebuild the main scratch pair first. Then add
new seeds and ensembles:

- Train at least one new `gated4d` scratch seed.
- Train at least one new `residual4d` scratch seed.
- Evaluate individual test metrics.
- Build global and per-parameter ensembles.
- Only call a result better if it improves validation-selected test
  mean range-normalized MAE on the same split.

5. Use the 32GB RTX 4080S, but optimize for validation/test error, not just
   memory usage:

- Start with configs known to fit.
- Use batch-size sweeps to avoid OOM.
- Before launching a training or GPU sweep, check for an existing active GPU job:
  `pgrep -af 'lno327_run_cnn_task|lno327_cloud_gpu_sweep' || true`.
  If a matching active job exists, monitor it instead of launching a duplicate.
- Try larger batch only when it improves throughput without worsening
  validation/test behavior.
- Try larger model capacity when justified: for example hidden=48/embedding=96
  before hidden=64/embedding=128.
- Use CUDA AMP for formal 4080S runs unless a smoke test shows NaN/OOM/regression:
  pass `--amp --amp-dtype float16` to `scripts/lno327_run_cnn_task.py`,
  `scripts/lno327_cloud_gpu_sweep.py`, and `scripts/lno327_tune_cnn_batch.py`.
  If float16 is unstable, try `--amp --amp-dtype bfloat16`; if both are
  unstable, fall back to FP32 and explicitly report why.
- If a larger batch makes test error worse, say so and move on.
- Record peak GPU memory, active utilization, runtime, and samples/s where
  possible.

6. Review and regenerate reports:

```bash
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_write_cloud_fwhm10_report.py
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_write_4d_final_report.py
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_write_training_visual_report.py
~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_write_data_parameter_report.py
```

7. Use the FWHM10 n512 suite only as a secondary diagnostic. If the sweep result
   already exists, read it instead of rerunning it:

```bash
test -s runs/cloud_fwhm10_n512/gpu_sweep_seed20260527/gpu_sweep.json || \
  ~/anaconda3/envs/torch2.4_cuda12.1/bin/python scripts/lno327_cloud_gpu_sweep.py --amp --amp-dtype float16
```

Read `runs/cloud_fwhm10_n512/gpu_sweep_seed20260527/gpu_sweep.json`.

8. Choose the next experiment based on evidence:

- If batch 96/128 is safe and faster, rerun the best architecture with larger batch.
- If hidden=48/embedding=96 or hidden=64/embedding=128 is safe, train one larger gated4d or residual4d candidate.
- If larger models overfit or do not improve validation/test MAE, recommend expanding sample count to n=1024/2048 before changing architecture.

Current measured guidance from the first cloud sweep:

- `gated4d/residual4d hidden=32 embedding=64 batch=128` fits and is the best first formal rerun.
- `batch=160/192` OOM for the small model.
- `hidden=48 embedding=96 batch=96` fits but is near 30GB and slower than small batch=128.
- `hidden=64 embedding=128` fits only at smaller batches and is not the first choice.

Therefore the next formal run should usually be a batch=128 rerun of the best
`residual4d` route or a batch=128 `gated4d` seed, followed by the same Chinese
comparison report.

9. Reports must be concrete and in Chinese:

- Explain each target parameter: SJc, SJ1a, SJ1b, SJ2, SA.
- State dataset, FWHM, n, Q/E/map resolution.
- State batch size, hidden, embedding, learning rate, epochs, trainable scope, aux loss.
- State precision mode: FP32, AMP float16, or AMP bfloat16.
- State runtime for every run.
- State GPU memory and utilization from the sweep.
- Include per-parameter errors and paper-like reference prediction.
- Explain what got better/worse in human language.
- Compare against the historical `2.3839%` pure-wide best when working on the
  main line.
- End with concrete next actions and whether the next action is training,
  ensembling, report publishing, or data regeneration.

10. Report promptly. Do not wait until the entire research program is finished.
    Write a short stage report whenever one of these happens:

- Dependency/data generation starts, fails, or finishes.
- A training run starts, hits OOM, early-stops, or finishes.
- An ensemble is produced.
- A public/private report is regenerated.
- You choose not to continue an experiment because evidence says it is worse.

Write stage reports under:

```text
runs/cloud_codex_agent/latest/stage_report.md
docs/cloud_agent_status.md
```

Every stage report should include:

- Current time and git commit.
- Current command or PID.
- Dataset and run directory.
- Runtime so far or final runtime.
- GPU memory/utilization if relevant.
- Current best metric and comparison against `2.3839%`.
- Failure diagnosis and the exact next action if blocked.

11. After each report update:

```bash
git add docs scripts README.md PROJECT_SUMMARY.md
git commit -m "Update cloud training report"
git push origin main
scripts/lno327_publish_report_pages.sh docs/cloud_fwhm10_n512/model_comparison_seed20260527 || true
```

If `scripts/lno327_publish_report_pages.sh` fails because the report deploy key is missing, write that clearly in the report/log.

## Success Criteria

- GPU is measured, not guessed.
- The main pure-wide n1024 line is restored before spending more time on the
  secondary FWHM10 n512 diagnostic line.
- The next run uses the 32GB VRAM effectively, but only changes direction when
  validation/test metrics justify it.
- The public report site is kept suitable for `ins-cnn-report`, not the private `ins-cnn` repo.
- The user can understand what happened without reading logs.
- The user receives timely Chinese status reports, including failures and next
  actions.
