# Experiment Tracker (Week 3 Local REDS-Style)

| run_id | date | model | frames | loss | scale | train_data | eval_data | steps | batch | lr | seed | gpu | ckpt_path | psnr_y | ssim_y | psnr_rgb | ssim_rgb | stability_metric | ms_per_frame | status | notes |
|---|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|---:|---|---|
| run_w3_bicubic_001 | 2026-02-19 | bicubic | 5 | n/a | 2 | n/a | local REDS-style (20 seq) | 0 | n/a | n/a | 123 | CPU | n/a | 36.8033 | 0.9601 | 35.1789 | 0.9511 | tPSNR=36.3441 | 0.238 | done | eval-only baseline |
| run_w3_sisr_001 | 2026-02-19 | single_frame_sr | 1 | L1 | 2 | Vimeo90K union | local REDS-style (20 seq) | 300 | 1 | 1e-4 | 123 | CPU | results/runs/week3_single_frame_fast_localreds/checkpoints/best.pt | 32.3755 | 0.8807 | 28.3284 | 0.7745 | tPSNR=33.8817 | 9.105 | done | fast-cycle training |
| run_w3_tvsr_001 | 2026-02-19 | temporal_vsr_small | 5 | L1 | 2 | Vimeo90K union | local REDS-style (20 seq) | 600 | 1 | 1e-4 | 123 | CPU | results/runs/week3_temporal_fast_localreds/checkpoints/best.pt | 29.2989 | 0.8532 | 25.5841 | 0.7553 | tPSNR=37.6005 | 34.660 | done | fast-cycle training |
| run_w3_tvsr_002 | 2026-02-19 | temporal_vsr_small | 5 | L1 | 2 | Vimeo90K union | local REDS-style (20 seq) | 2000 | 1 | 1e-4 | 123 | CPU | results/runs/week3_temporal_localreds_2k/checkpoints/best.pt | 33.5138 | 0.9247 | 30.9240 | 0.8804 | tPSNR=37.6004 | 34.660 | done | extended local CPU run |
