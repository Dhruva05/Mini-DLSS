# Experiment Tracker Template

Use one row per run. Keep immutable run IDs.

## Run Log

| run_id | date | model | frames | loss | scale | train_data | eval_data | steps | batch | lr | seed | gpu | ckpt_path | psnr_y | ssim_y | psnr_rgb | ssim_rgb | stability_metric | ms_per_frame | status | notes |
|---|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|---|---|
| run_0001 | 2026-02-16 | bicubic | 1 | n/a | 2 | n/a | REDS-val | 0 | n/a | n/a | n/a | CPU | n/a | 0.00 | 0.0000 | 0.00 | 0.0000 | diff_energy=0.0000 | 0.00 | done | baseline placeholder |
| run_0002 | YYYY-MM-DD | single_frame_sr | 1 | L1 | 2 | Vimeo-90K | REDS-val | 50000 | 8 | 2e-4 | 42 | T4 | path/to/ckpt.pth | 0.00 | 0.0000 | 0.00 | 0.0000 | tPSNR=0.00 | 0.00 | running | fast cycle |

## Required Reporting Rules

- `scale` must always be `2`.
- `psnr_y` and `ssim_y` are primary report metrics.
- Crop border for quality metrics: `2` pixels per side.
- Use scene-disjoint train/eval split.
- `stability_metric` must be either:
  - `tPSNR=<value>` (preferred), or
  - `diff_energy=<value>` (fallback).
- `ms_per_frame` measured on fixed 10-second clip for final-model runs.

## Milestone Checks

| milestone | pass_condition | run_id | status |
|---|---|---|---|
| baseline_ready | Bicubic and single-frame baselines logged | run_xxxx | pending |
| temporal_beats_single | `psnr_y` gain >= `0.2 dB` vs single-frame SR | run_xxxx | pending |
| stability_non_regression | Stability metric better or equal vs single-frame SR | run_xxxx | pending |
| onnx_demo_ready | ONNX inference succeeds on fixed 10s clip with `ms/frame` | run_xxxx | pending |
