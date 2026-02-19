# Config Notes

- All configs are locked to `scale = 2`.
- Current defaults assume:
  - Vimeo train root: `data/raw/vimeo90k_union/sequence`
  - REDS val roots: `data/raw/reds/val_sharp` and `data/raw/reds/val_sharp_bicubic/X2`
- If REDS is not present locally, set `val_*` roots via `--override` for smoke runs.
- `temporal_small.toml` is the default fast-cycle config.
- `temporal_full.toml` is the full-budget config for final reporting.
