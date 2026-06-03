#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_official_reds.sh <official_val_hr_root> <official_val_lr_x2_root> <official_manifest> [checkpoint] [output_dir] [device]

VAL_HR=${1:-}
VAL_LR=${2:-}
MANIFEST=${3:-}
CHECKPOINT=${4:-results/runs/temporal_vsr_5f_small_2x/checkpoints/best.pt}
OUTPUT_DIR=${5:-results/official_reds}
DEVICE=${6:-cpu}

if [[ -z "$VAL_HR" || -z "$VAL_LR" || -z "$MANIFEST" ]]; then
  echo "Usage: bash scripts/eval_official_reds.sh <official_val_hr_root> <official_val_lr_x2_root> <official_manifest> [checkpoint] [output_dir] [device]"
  exit 1
fi

BICUBIC_OVERRIDE="{\"project\":{\"run_name\":\"official_bicubic_reds_2x\"},\"dataset\":{\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"val_manifest\":\"$MANIFEST\",\"generate_lr_on_the_fly\":false}}"
MODEL_OVERRIDE="{\"project\":{\"run_name\":\"official_temporal_vsr_5f_small_2x\"},\"dataset\":{\"val_hr_root\":\"$VAL_HR\",\"val_lr_root\":\"$VAL_LR\",\"val_manifest\":\"$MANIFEST\",\"generate_lr_on_the_fly\":false}}"

python eval.py \
  --config configs/temporal_small.toml \
  --override "$BICUBIC_OVERRIDE" \
  --mode bicubic \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR"

python eval.py \
  --config configs/temporal_small.toml \
  --override "$MODEL_OVERRIDE" \
  --mode model \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR"
