# Mini-DLSS

Mini-DLSS (DLSS-inspired, not NVIDIA DLSS): temporal video super-resolution trained on public VSR datasets, evaluated with standard SR metrics (PSNR/SSIM) plus temporal stability checks, and deployed via ONNX for reproducible inference.

## 1) Scope

- Task: temporal video super-resolution.
- Upscale factor: `2x` (fixed for all experiments).
- Focus: ML pipeline, model training, benchmarking, and deployment.
- Out of scope (for now): game-engine integration and custom rendering pipelines.

## 2) Datasets

- Train: Vimeo-90K Septuplets.
- Eval/demo: REDS validation split (scene-disjoint from training data).
- Split rule: no overlapping scenes between train/val/test subsets.
- Local default setup in this repo uses merged Kaggle subsets (`vimeo-90k-3` + `vimeo-90k-4`) via:
  - sequence root: `data/raw/vimeo90k_union/sequence`
  - manifests: `data/splits/vimeo90k_union_{train,val,test}.txt`

## 3) Baselines and Model Path

- Baseline A: bicubic upscaling.
- Baseline B: single-image SR applied frame-by-frame.
- Ours: BasicVSR-style temporal model.
- Training path:
  - Stage 1: `BasicVSR-Small` for pipeline validation and quick cycles.
  - Stage 2: scaled model for quality-focused runs.
- Optional extension: temporal consistency loss and/or flow-assisted alignment.

## 4) Evaluation Protocol (Locked)

- Primary quality metrics: `PSNR` and `SSIM`.
- Primary reporting domain: `Y` channel in YCbCr.
- Secondary sanity report: RGB PSNR/SSIM (optional).
- Border crop before PSNR/SSIM: crop `scale` pixels on each side (`2` pixels for 2x).
- Temporal stability metric:
  - Preferred: flow-warped temporal PSNR (`tPSNR`) between consecutive outputs.
  - Fallback: frame-to-frame difference energy in low-texture regions.
- Report set:
  - REDS validation aggregate metrics.
  - Curated qualitative clips (side-by-side LR/Bicubic/Single-frame/Temporal/GT).

## 5) Compute and Reproducibility

- Runtime target: Google Colab GPU for training.
- All runs resumable with periodic checkpoints.
- Validation runs at fixed interval during training.
- Two experiment tiers:
  - Fast cycle: short training budget for iteration/debugging.
  - Full cycle: long budget for final reporting.
- Seeds/configs/checkpoints tracked per run for reproducibility.

## 6) Definition of Done

- Temporal model improves REDS-val PSNR over single-frame SR by at least `+0.2 dB` (target range `+0.2` to `+0.5 dB`).
- Temporal stability metric improves over single-frame SR, or does not regress.
- ONNX export succeeds and runs on a fixed 10-second clip with reported latency (`ms/frame`) on local machine.
- Repo includes reproducible train/eval/export/demo commands and final results artifacts.

## 7) Execution Plan (8 Weeks)

- Week 1: lock protocol, metrics, splits, and run tracker format.
- Week 2: data ingestion, alignment checks, window sampling checks.
- Week 3: baseline training/eval and auto-generated markdown results table.
- Week 4: BasicVSR-Small training and validation loop stabilization.
- Week 5: scaled model training and reproducible best-checkpoint eval.
- Week 6: ablations (3/5/7 frames, loss variants, tiny speed model) and latency chart.
- Week 7: ONNX export + mp4 demo CLI.
- Week 8: final report page with metrics, videos, failure cases, and tradeoff analysis.

## 8) Experiment Matrix

1. `bicubic`
2. `single_frame_sr`
3. `temporal_vsr_3f`
4. `temporal_vsr_5f`
5. `temporal_vsr_7f`
6. `temporal_vsr_5f_l1_perceptual`
7. `temporal_vsr_5f_tiny` (speed-oriented)

## 9) Repo Layout

```text
mini-dlss/
  configs/
  data/
    scripts/
    splits/
  models/
  metrics/
  notebooks/
  results/
    tables/
    videos/
  train.py
  eval.py
  export_onnx.py
  demo_video.py
  README.md
```

## 10) Command Checklist

Week 2 dataset check (clip counts + LR/HR alignment + temporal windows):

```bash
python data/scripts/freeze_vimeo_splits.py --vimeo-root /path/to/vimeo90k
python data/scripts/freeze_vimeo_union_splits.py \
  --vimeo-a data/raw/kagglehub/datasets/wangsally/vimeo-90k-3/versions/1 \
  --vimeo-b data/raw/kagglehub/datasets/wangsally/vimeo-90k-4/versions/1
python data/scripts/build_vimeo_union_root.py \
  --seq-a data/raw/kagglehub/datasets/wangsally/vimeo-90k-3/versions/1/sequence \
  --seq-b data/raw/kagglehub/datasets/wangsally/vimeo-90k-4/versions/1/sequence

python data/scripts/check_dataset.py \
  --hr-root /path/to/reds/hr \
  --lr-root /path/to/reds/lr \
  --manifest data/splits/reds_val.txt \
  --num-frames 5 \
  --scale 2
```

If REDS is not available yet, build a local REDS-style val set from Vimeo for pipeline validation:

```bash
python data/scripts/create_reds_val_from_vimeo.py
```

Week 3 baseline eval (writes markdown table + example media):

```bash
python eval.py \
  --config configs/temporal_small.toml \
  --override '{"dataset":{"val_hr_root":"/path/to/reds/hr","val_lr_root":"/path/to/reds/lr"}}' \
  --mode bicubic \
  --save-videos 2 \
  --save-images 8
```

Week 4/5 temporal training (small -> full):

```bash
python train.py --config configs/temporal_small.toml \
  --override '{"dataset":{"train_hr_root":"/path/to/vimeo/hr","val_hr_root":"/path/to/reds/hr","val_lr_root":"/path/to/reds/lr"}}'
python train.py --config configs/temporal_full.toml \
  --override '{"dataset":{"train_hr_root":"/path/to/vimeo/hr","val_hr_root":"/path/to/reds/hr","val_lr_root":"/path/to/reds/lr"}}'
```

Week 5 reproducible best-checkpoint evaluation:

```bash
python eval.py \
  --config configs/temporal_full.toml \
  --override '{"dataset":{"val_hr_root":"/path/to/reds/hr","val_lr_root":"/path/to/reds/lr"}}' \
  --checkpoint results/runs/temporal_vsr_5f_full_2x/checkpoints/best.pt
```

Week 6 ablation tracking:

```bash
# Fill run rows in results/tables/experiment_tracker_template.md
# Then copy summary into results/tables/ablation_table.md
```

Week 7 ONNX export:

```bash
python export_onnx.py \
  --config configs/temporal_full.toml \
  --checkpoint results/runs/temporal_vsr_5f_full_2x/checkpoints/best.pt \
  --output results/onnx/mini_dlss_temporal_full.onnx
```

Week 7 demo inference on any LR mp4:

```bash
python demo_video.py \
  --input /path/to/lr_clip.mp4 \
  --output results/videos/demo_sr.mp4 \
  --config configs/temporal_full.toml \
  --checkpoint results/runs/temporal_vsr_5f_full_2x/checkpoints/best.pt
```

## 11) Reproducibility Rules

- Keep `scale=2` in all configs.
- Use only manifests in `data/splits/`.
- Log every run in `results/tables/experiment_tracker_template.md`.
- Report Y-channel metrics with crop border = 2.
- Record the same fixed 10-second clip for latency (`ms/frame`).
