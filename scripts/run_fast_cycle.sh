#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_fast_cycle.sh /path/train_hr /path/val_hr /path/val_lr

TRAIN_HR=${1:-}
VAL_HR=${2:-}
VAL_LR=${3:-}

if [[ -z "$TRAIN_HR" || -z "$VAL_HR" || -z "$VAL_LR" ]]; then
  echo "Usage: bash scripts/run_fast_cycle.sh <train_hr_root> <val_hr_root> <val_lr_root>"
  exit 1
fi

OVERRIDE="{\"dataset\":{\"train_hr_root\":\"$TRAIN_HR\",\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\"}}"
python train.py --config configs/temporal_small.toml --override "$OVERRIDE"
python eval.py --config configs/temporal_small.toml --override "$OVERRIDE"
