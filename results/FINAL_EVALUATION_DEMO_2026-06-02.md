# Mini-DLSS Final Evaluation And Demo

Generated on 2026-06-02 from the trained Drive checkpoint:

```text
/content/drive/MyDrive/mini-dlss/results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt
```

Local checkpoint mirror:

```text
results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt
```

## Training Setup

| item | value |
| --- | --- |
| Model | BasicVSR-style temporal SR, `temporal_small` |
| Scale | 2x |
| Temporal window | 5 LR frames |
| Channels / blocks | 48 base channels, 6 residual blocks |
| Training data | `data/raw/vimeo90k_union/sequence` |
| Eval/demo data | local Vimeo-derived REDS-style set at `data/raw/reds/val_sharp`, `data/splits/reds_val.txt` |
| Loss | L1 |
| Optimizer LR | 0.0002 |
| Batch size | 4 |
| Best checkpoint step | 80000 |
| Best validation metric in checkpoint | 38.3265 PSNR-Y |

The single-frame SR baseline below uses the available local fast-cycle checkpoint:
`results/runs/week3_single_frame_fast_localreds/checkpoints/best.pt`.

These metrics are local pipeline/demo validation results. They should not be cited as official REDS benchmark numbers unless the eval/demo paths are replaced with official REDS data and the evaluation is rerun.

## Evaluation Results

Evaluation command family:

```bash
.venv/bin/python eval.py --config configs/temporal_small.toml --mode bicubic --device cpu --output-dir results/final
.venv/bin/python eval.py --config configs/single_frame_sr.toml --mode model --checkpoint results/runs/week3_single_frame_fast_localreds/checkpoints/best.pt --device cpu --output-dir results/final
.venv/bin/python eval.py --config configs/temporal_small.toml --mode model --checkpoint results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt --device cpu --output-dir results/final
```

| Method | Checkpoint | PSNR-Y | SSIM-Y | tPSNR | diff_energy |
| --- | --- | ---: | ---: | ---: | ---: |
| Bicubic | none | 36.8033 | 0.9601 | 36.3441 | 0.0217 |
| Single-frame SR fast-cycle (300 steps) | `week3_single_frame_fast_localreds/best.pt` | 32.3755 | 0.8807 | 33.8817 | 0.0183 |
| Temporal SR 5f small | `temporal_vsr_5f_small_2x/best.pt` | 38.3265 | 0.9604 | 36.6018 | 0.0201 |

Temporal SR improves over bicubic by `+1.5231 dB` PSNR-Y and `+0.2578 dB` tPSNR, with lower `diff_energy` (`0.0201` vs `0.0217`). Compared with the undertrained single-frame baseline, temporal SR is much stronger in PSNR-Y and tPSNR, but the single-frame baseline has lower raw `diff_energy`; avoid claiming every temporal metric improves against that baseline until a budget-matched single-frame checkpoint is trained.

Raw metric files:

- [Bicubic JSON](final/tables/final_bicubic_reds_2x_eval.json)
- [Single-frame JSON](final/tables/final_single_frame_fast_localreds_2x_eval.json)
- [Temporal JSON](final/tables/final_temporal_vsr_5f_small_2x_eval.json)

## Comparison Videos

Panel order is:

```text
LR input | Bicubic | Temporal SR | HR target
```

- [Labeled all-validation comparison, 10 seconds](final/videos/final_temporal_vsr_5f_small_2x/all_val_comparison_10s_labeled.mp4)
- [Scene 240 comparison](final/videos/final_temporal_vsr_5f_small_2x/240_comparison.mp4)
- [Scene 245 comparison](final/videos/final_temporal_vsr_5f_small_2x/245_comparison.mp4)
- [Scene 252 comparison](final/videos/final_temporal_vsr_5f_small_2x/252_comparison.mp4)

The labeled package video was generated with:

```bash
.venv/bin/python scripts/make_final_comparison_video.py \
  --config configs/temporal_small.toml \
  --checkpoint results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt \
  --output results/final/videos/final_temporal_vsr_5f_small_2x/all_val_comparison_10s_labeled.mp4 \
  --duration-sec 10 \
  --device cpu
```

## ONNX Export

```bash
.venv/bin/python export_onnx.py \
  --config configs/temporal_small.toml \
  --checkpoint results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt \
  --output results/onnx/temporal_vsr_5f_small_2x.onnx
```

Output:

- [ONNX model](onnx/temporal_vsr_5f_small_2x.onnx)

## 10-Second Demo

Input clip:

```text
results/week3/latency_input_10s_lr.mp4
```

It is 240 frames at 24 fps, exactly 10.0 seconds, 64x64 LR.

Demo command:

```bash
.venv/bin/python demo_video.py \
  --input results/week3/latency_input_10s_lr.mp4 \
  --output results/final/demo/temporal_vsr_5f_small_2x_10s.mp4 \
  --config configs/temporal_small.toml \
  --checkpoint results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt \
  --device cpu \
  --fps 24
```

Result:

| item | value |
| --- | --- |
| Output video | [temporal_vsr_5f_small_2x_10s.mp4](final/demo/temporal_vsr_5f_small_2x_10s.mp4) |
| Frames | 240 |
| CPU elapsed | 8.386 sec |
| PyTorch CPU latency | 34.941 ms/frame |
| Checkpoint | `results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt` |

The ONNX model was exported successfully, but ONNX Runtime latency has not been benchmarked yet.

## Failure Cases

- Fast motion and occlusions can still produce edge wobble because the model uses lightweight recurrent alignment rather than explicit optical flow or deformable alignment.
- Thin text, fences, and repeated high-frequency textures can look oversmoothed or temporally inconsistent.
- The CPU demo is useful for reproducibility, but it is not a real-time deployment benchmark. GPU, ONNX Runtime, and quantization should be measured separately.
- The current single-frame comparison is not budget-matched with the temporal model; it is useful as a pipeline baseline, not as a final single-frame ablation claim.

## Improvements Next

- Train a budget-matched single-frame SR baseline and rerun this exact evaluation table.
- Add flow-warped temporal consistency loss or a stronger alignment module for tougher motion.
- Benchmark ONNX Runtime latency and try fp16 or int8 quantization.
- Expand the final report with per-scene metrics and selected failure clips.
