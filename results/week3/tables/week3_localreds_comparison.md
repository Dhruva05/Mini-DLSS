# Week 3 Local REDS-Style Comparison

Note: evaluation uses the local REDS-style set generated from Vimeo for pipeline progression, not official REDS.

| method | params_m | psnr_y | ssim_y | psnr_rgb | ssim_rgb | tpsnr | diff_energy | ms_per_frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 0.000 | 36.8033 | 0.9601 | 35.1789 | 0.9511 | 36.3441 | 0.0217 | 0.238 |
| Single-frame SR (fast-cycle) | 0.742 | 32.3755 | 0.8807 | 28.3284 | 0.7745 | 33.8817 | 0.0183 | 9.105 |
| Temporal SR (fast-cycle, 600 steps) | 0.716 | 29.2989 | 0.8532 | 25.5841 | 0.7553 | 37.6005 | 0.0134 | 34.660 |
| Temporal SR (extended local run, 2000 steps) | 0.716 | 33.5138 | 0.9247 | 30.9240 | 0.8804 | 37.6004 | 0.0175 | 34.660 |
