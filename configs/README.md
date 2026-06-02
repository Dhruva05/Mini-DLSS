# Config Notes

- All configs are locked to `scale = 2`.
- Current defaults assume:
  - Vimeo train root: `data/raw/vimeo90k_union/sequence`
  - local REDS-style val roots: `data/raw/reds/val_sharp` and `data/raw/reds/val_sharp_bicubic/X2`
- If official REDS is used, replace `val_*` roots via `--override` and rerun evaluation before making benchmark claims.
- `temporal_small.toml` defines the final reported architecture and local defaults.
- The final `temporal_vsr_5f_small_2x` checkpoint was Drive-trained with an override recorded in checkpoint metadata (`train.max_steps = 150000`, best step `80000`).
- `temporal_full.toml` remains an unreported scale-up config for future experiments.
