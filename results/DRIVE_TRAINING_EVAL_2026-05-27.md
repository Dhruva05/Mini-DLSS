# Drive Training Evaluation - 2026-05-27

Source inspected through Google Drive:
`MyDrive/mini-dlss/results/runs/temporal_vsr_5f_small_2x`

## Artifacts Found

Current run:

| artifact | status |
| --- | --- |
| `train_log.jsonl` | present |
| `val_metrics.jsonl` | present |
| `checkpoints/best.pt` | present |
| `checkpoints/latest.pt` | present |
| `checkpoints/step_0005000.pt` | present |
| `checkpoints/step_0010000.pt` | present |
| `checkpoints/step_0015000.pt` | present |
| `checkpoints/step_0020000.pt` | present |
| `checkpoints/step_0025000.pt` | present |

Backup runs are also present from earlier restarts:

| backup run | observed status |
| --- | --- |
| `temporal_vsr_5f_small_2x_backup_20260523_224143` | validation through 4k steps |
| `temporal_vsr_5f_small_2x_backup_20260523_224449` | short train log only |
| `temporal_vsr_5f_small_2x_backup_20260526_234931` | validation through 15k steps |

## Current Run Summary

The current run is the best run so far. Validation PSNR-Y improves steadily from step 5k to 25k.

| step | PSNR-Y | SSIM-Y | PSNR-RGB | SSIM-RGB |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 36.7536 | 0.9541 | 34.9872 | 0.9408 |
| 10,000 | 37.2704 | 0.9579 | 35.5022 | 0.9464 |
| 15,000 | 37.6654 | 0.9594 | 35.8638 | 0.9501 |
| 20,000 | 37.6736 | 0.9571 | 35.8400 | 0.9474 |
| 25,000 | 37.7600 | 0.9584 | 35.8739 | 0.9491 |

Best checkpoint by PSNR-Y: `checkpoints/best.pt`, matching step 25,000.

Training log reached step 26,300, with 420,800 samples seen in 14,499 seconds. That implies an inferred batch size of 16 and about 1.81 steps/sec, or about 29 samples/sec.

## Interpretation

This is no longer just a smoke test. The temporal model is learning useful reconstruction.

Compared with the earlier local REDS-style report in `results/week3/tables/week3_localreds_comparison.md`:

| comparison | PSNR-Y |
| --- | ---: |
| Earlier bicubic baseline | 36.8033 |
| Earlier temporal 2k run | 33.5138 |
| Current Drive temporal 25k run | 37.7600 |

If the Colab validation set matches the local REDS-style validation set, the current Drive model is about +0.96 dB over bicubic and +4.25 dB over the earlier 2k temporal model. That clears the initial definition-of-done target for PSNR improvement.

The caveat: the Drive run folder does not include a saved config, full `eval.py` output, temporal stability metrics, or generated comparison videos. So the current numbers are promising training-validation numbers, not yet a complete final benchmark package.

## Missing Pieces

| missing item | why it matters |
| --- | --- |
| saved run config | proves exact dataset paths, batch size, validation interval, and overrides |
| full `eval.py` output | adds `tPSNR`, `diff_energy`, image/video examples, and comparable tables |
| same-run bicubic and single-frame baselines | makes the improvement defensible |
| official REDS4 / official REDS-val eval | avoids overclaiming from generated local REDS-style validation |
| ONNX export from `best.pt` | needed for deployment/demo milestone |

## Recommended Next Steps

1. Treat `temporal_vsr_5f_small_2x/checkpoints/best.pt` as the current best model.
2. Resume the Drive training run from `latest.pt` to 50k steps, because PSNR-Y is still improving at 25k.
3. Save the exact resolved config into the run folder before the next run.
4. Run full evaluation from `best.pt` against bicubic and single-frame baselines using the same validation split.
5. Generate comparison videos from the current best checkpoint.
6. Export the current best model to ONNX and run the fixed 10-second demo clip.
7. Start ablations after the 50k run: 3-frame vs 5-frame vs 7-frame, then tiny-model latency.
