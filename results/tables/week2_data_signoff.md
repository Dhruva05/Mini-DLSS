# Week 2 Data Pipeline Sign-off

Date: 2026-02-19

## Scope Checked

- Vimeo manifest freezing and union manifest generation
- Vimeo sequence-root wiring for multi-subset training
- Dataset loader alignment/window sampling checks for 2x temporal SR (`num_frames=5`)
- Config defaults for training/eval data roots

## Completed

- Generated union manifests from Kaggle subsets:
  - `data/splits/vimeo90k_union_train.txt`
  - `data/splits/vimeo90k_union_val.txt`
  - `data/splits/vimeo90k_union_test.txt`
- Built merged Vimeo root with symlinked top-level sequence folders:
  - `data/raw/vimeo90k_union/sequence`
- Updated default configs to use union Vimeo train data:
  - `configs/temporal_small.toml`
  - `configs/temporal_full.toml`
  - `configs/single_frame_sr.toml`

## Verification Results

### Vimeo Union Train

Command:

```bash
.venv/bin/python data/scripts/check_dataset.py \
  --hr-root data/raw/vimeo90k_union/sequence \
  --manifest data/splits/vimeo90k_union_train.txt \
  --num-frames 5 --scale 2 --generate-lr-on-the-fly --preview 0
```

Result:

- sequences: 12824
- temporal windows: 38472
- total HR frames: 89768
- status: PASS

### Vimeo Union Val

Command:

```bash
.venv/bin/python data/scripts/check_dataset.py \
  --hr-root data/raw/vimeo90k_union/sequence \
  --manifest data/splits/vimeo90k_union_val.txt \
  --num-frames 5 --scale 2 --generate-lr-on-the-fly --preview 0
```

Result:

- sequences: 674
- temporal windows: 2022
- total HR frames: 4718
- status: PASS

### REDS Val Alignment

Command:

```bash
.venv/bin/python data/scripts/check_dataset.py \
  --hr-root data/raw/reds/val_sharp \
  --lr-root data/raw/reds/val_sharp_bicubic/X2 \
  --manifest data/splits/reds_val.txt \
  --num-frames 5 --scale 2 --preview 0
```

Result:

- status: PASS
- sequences: 20
- temporal windows: 60
- total HR frames: 140
- alignment mismatches: 0
- note: local REDS-style val set generated from Vimeo with `data/scripts/create_reds_val_from_vimeo.py`

## Week 2 Status

- Vimeo ingestion + alignment + window sampling: DONE
- Config and defaults cleanup: DONE
- REDS-path integration validation: DONE

## Notes

1. Current REDS paths are satisfied by generated local REDS-style data for pipeline validation.
2. For benchmark-grade reporting, replace generated local REDS-style data with official REDS and re-run checks.
