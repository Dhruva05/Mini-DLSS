# Split Manifests

All manifests in this folder are **authoritative** for Mini-DLSS experiments.

## REDS split policy (scene-disjoint)

- `reds_train.txt`: sequences `000` to `239`
- `reds_val.txt`: sequences `240` to `259`
- `reds_test.txt`: sequences `260` to `269`
- `reds_demo_clips.txt`: curated subset used in qualitative videos

These lists are fixed and should not be edited between experiments.

If official REDS files are unavailable locally, create a REDS-style validation set for
pipeline checks via:

```bash
python data/scripts/create_reds_val_from_vimeo.py
```

## Vimeo-90K split policy

Vimeo-90K uses the official septuplet list files distributed with the dataset.

- `vimeo90k_train.txt`, `vimeo90k_val.txt`, `vimeo90k_test.txt` are frozen manifests generated via:

```bash
python data/scripts/freeze_vimeo_splits.py --vimeo-root /path/to/vimeo90k
```

The generator creates deterministic train/val/test text files under this directory.

### Union manifests (default local setup)

If using Kaggle subsets `vimeo-90k-3` and `vimeo-90k-4`, generate merged manifests via:

```bash
python data/scripts/freeze_vimeo_union_splits.py \
  --vimeo-a data/raw/kagglehub/datasets/wangsally/vimeo-90k-3/versions/1 \
  --vimeo-b data/raw/kagglehub/datasets/wangsally/vimeo-90k-4/versions/1
```

This writes:
- `vimeo90k_union_train.txt`
- `vimeo90k_union_val.txt`
- `vimeo90k_union_test.txt`
