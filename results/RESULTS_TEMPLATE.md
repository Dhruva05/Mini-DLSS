# Mini-DLSS Results

## Protocol

- Scale: `2x`
- Metric domain: Y channel (primary), RGB (secondary)
- Border crop: 2 pixels
- Validation set: REDS-val (`data/splits/reds_val.txt`)
- Stability metric: `tPSNR` (preferred) or `diff_energy` fallback

## Main Comparison

| method | psnr_y | ssim_y | psnr_rgb | ssim_rgb | tpsnr | diff_energy | ms_per_frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicubic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| single_frame_sr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| temporal_vsr_best | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |

## Ablations

See `/results/tables/ablation_table.md`.

## Qualitative Examples

- Comparison videos: `/results/videos/<run_name>/`
- Sample frames: `/results/videos/<run_name>/images/`

## Failure Cases

- Add at least 2 clips where temporal model struggles (motion blur, disocclusions, repetitive textures).

## Conclusion

- State whether DoD thresholds were met:
  - PSNR gain >= +0.2 dB vs single-frame baseline
  - Stability non-regression
  - ONNX + 10s demo latency reported
