#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/single_frame_budget_matched.toml}
DEVICE=${2:-cuda}
EVAL_RUN_NAME=${3:-single_frame_sr_2x_budget_matched_5f_eval}

python train.py --config "$CONFIG" --device "$DEVICE"
python eval.py \
  --config "$CONFIG" \
  --override "{\"project\":{\"run_name\":\"$EVAL_RUN_NAME\"},\"dataset\":{\"num_frames\":5}}" \
  --mode model \
  --checkpoint results/runs/single_frame_sr_2x_budget_matched/checkpoints/best.pt \
  --device "$DEVICE" \
  --output-dir results/final
