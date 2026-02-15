#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablations.sh /path/train_hr /path/val_hr /path/val_lr

TRAIN_HR=${1:-}
VAL_HR=${2:-}
VAL_LR=${3:-}

if [[ -z "$TRAIN_HR" || -z "$VAL_HR" || -z "$VAL_LR" ]]; then
  echo "Usage: bash scripts/run_ablations.sh <train_hr_root> <val_hr_root> <val_lr_root>"
  exit 1
fi

run_train_eval () {
  local config=$1
  local override=$2
  python train.py --config "$config" --override "$override"
  python eval.py --config "$config" --override "$override"
}

# 3-frame temporal
run_train_eval \
  configs/temporal_small.toml \
  "{\"dataset\":{\"train_hr_root\":\"$TRAIN_HR\",\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"num_frames\":3}}"

# 5-frame temporal
run_train_eval \
  configs/temporal_small.toml \
  "{\"dataset\":{\"train_hr_root\":\"$TRAIN_HR\",\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"num_frames\":5}}"

# 7-frame temporal
run_train_eval \
  configs/temporal_small.toml \
  "{\"dataset\":{\"train_hr_root\":\"$TRAIN_HR\",\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"num_frames\":7}}"
