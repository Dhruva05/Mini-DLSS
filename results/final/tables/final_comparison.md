# Final Comparison

| Method | Checkpoint | PSNR-Y | SSIM-Y | PSNR-RGB | SSIM-RGB | tPSNR | diff_energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | none | 36.8033 | 0.9601 | 35.1789 | 0.9511 | 36.3441 | 0.0217 |
| Single-frame SR fast-cycle (300 steps) | `results/runs/week3_single_frame_fast_localreds/checkpoints/best.pt` | 32.3755 | 0.8807 | 28.3284 | 0.7745 | 33.8817 | 0.0183 |
| Temporal SR 5f small | `results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt` | 38.3265 | 0.9604 | 36.4981 | 0.9505 | 36.6018 | 0.0201 |
