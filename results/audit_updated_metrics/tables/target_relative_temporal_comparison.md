# Target-Relative Temporal Audit

This audit reruns `eval.py` with target-relative temporal diagnostics. The single-frame row uses a 5-frame eval-window override so it is evaluated on the same center-frame sample set as the temporal model.

| Method | PSNR-Y | SSIM-Y | PSNR-RGB | SSIM-RGB | tPSNR | diff_energy | target_tPSNR | tPSNR_delta | target_diff_energy | diff_energy_delta | temporal_error_energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 36.8033 | 0.9601 | 35.1789 | 0.9511 | 36.3441 | 0.0217 | 34.7109 | 1.6331 | 0.0251 | -0.0035 | 0.0108 |
| Single-frame SR fast-cycle, 5f eval window | 32.3765 | 0.8808 | 28.3342 | 0.7747 | 33.9608 | 0.0188 | 34.7109 | -0.7501 | 0.0251 | -0.0063 | 0.0145 |
| Temporal SR 5f small | 38.3265 | 0.9604 | 36.4981 | 0.9505 | 36.6018 | 0.0201 | 34.7109 | 1.8909 | 0.0251 | -0.0050 | 0.0105 |
