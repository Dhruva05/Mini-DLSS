#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablations.sh /path/train_hr /path/val_hr /path/val_lr [max_steps] [device]

TRAIN_HR=${1:-}
VAL_HR=${2:-}
VAL_LR=${3:-}
MAX_STEPS=${4:-50000}
DEVICE=${5:-cuda}

if [[ -z "$TRAIN_HR" || -z "$VAL_HR" || -z "$VAL_LR" ]]; then
  echo "Usage: bash scripts/run_ablations.sh <train_hr_root> <val_hr_root> <val_lr_root> [max_steps] [device]"
  exit 1
fi

run_train_eval () {
  local run_name=$1
  local frames=$2
  local loss=$3
  local perceptual_weight=$4
  local base_channels=$5
  local num_blocks=$6
  local override
  override="{\"project\":{\"run_name\":\"$run_name\"},\"dataset\":{\"train_hr_root\":\"$TRAIN_HR\",\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"num_frames\":$frames},\"train\":{\"max_steps\":$MAX_STEPS,\"val_interval\":5000,\"checkpoint_interval\":5000,\"loss\":\"$loss\",\"perceptual_weight\":$perceptual_weight},\"eval\":{\"max_batches\":50},\"model\":{\"base_channels\":$base_channels,\"num_blocks\":$num_blocks}}"

  python train.py --config configs/temporal_small.toml --override "$override" --device "$DEVICE"
  python eval.py \
    --config configs/temporal_small.toml \
    --override "$override" \
    --mode model \
    --checkpoint "results/runs/$run_name/checkpoints/best.pt" \
    --device "$DEVICE" \
    --output-dir results/ablations
}

# 3-frame temporal
run_train_eval "temporal_vsr_3f_small_2x" 3 "l1" 0.0 48 6

# 5-frame temporal
run_train_eval "temporal_vsr_5f_small_2x_ablation" 5 "l1" 0.0 48 6

# 7-frame temporal
run_train_eval "temporal_vsr_7f_small_2x" 7 "l1" 0.0 48 6

# 5-frame temporal with perceptual loss.
run_train_eval "temporal_vsr_5f_l1_perceptual_2x" 5 "l1_perceptual" 0.01 48 6

# 5-frame speed-oriented tiny model.
run_train_eval "temporal_vsr_5f_tiny_2x" 5 "l1" 0.0 32 4
