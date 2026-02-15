# Split Manifests

All manifests in this folder are **authoritative** for Mini-DLSS experiments.

## REDS split policy (scene-disjoint)

- `reds_train.txt`: sequences `000` to `239`
- `reds_val.txt`: sequences `240` to `259`
- `reds_test.txt`: sequences `260` to `269`
- `reds_demo_clips.txt`: curated subset used in qualitative videos

These lists are fixed and should not be edited between experiments.

## Vimeo-90K split policy

Vimeo-90K uses the official septuplet list files distributed with the dataset.

- `vimeo90k_train.txt`, `vimeo90k_val.txt`, `vimeo90k_test.txt` are frozen manifests generated via:

```bash
python data/scripts/freeze_vimeo_splits.py --vimeo-root /path/to/vimeo90k
```

The generator creates deterministic train/val/test text files under this directory.
