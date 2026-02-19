#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def read_manifest(path: Path) -> List[str]:
    items = [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith('#')]
    if not items:
        raise ValueError(f"No sequences in manifest: {path}")
    return items


def list_frames(seq_dir: Path) -> List[Path]:
    frames = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not frames:
        raise ValueError(f"No frames found: {seq_dir}")
    return frames


def save_bicubic_x2(hr_img: Image.Image, out_path: Path) -> None:
    w, h = hr_img.size
    lr = hr_img.resize((w // 2, h // 2), Image.Resampling.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lr.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create local REDS-style val set from Vimeo sequences')
    parser.add_argument('--vimeo-root', type=Path, default=Path('data/raw/vimeo90k_union/sequence'))
    parser.add_argument('--source-manifest', type=Path, default=Path('data/splits/vimeo90k_union_val.txt'))
    parser.add_argument('--reds-ids', type=Path, default=Path('data/splits/reds_val.txt'))
    parser.add_argument('--out-hr', type=Path, default=Path('data/raw/reds/val_sharp'))
    parser.add_argument('--out-lr-x2', type=Path, default=Path('data/raw/reds/val_sharp_bicubic/X2'))
    args = parser.parse_args()

    source_ids = read_manifest(args.source_manifest)
    reds_ids = read_manifest(args.reds_ids)

    if len(source_ids) < len(reds_ids):
        raise ValueError(
            f"Not enough source sequences ({len(source_ids)}) for REDS ids ({len(reds_ids)})"
        )

    used = source_ids[: len(reds_ids)]

    for reds_id, src_id in zip(reds_ids, used):
        src_dir = args.vimeo_root / src_id
        src_frames = list_frames(src_dir)

        hr_seq_dir = args.out_hr / reds_id
        lr_seq_dir = args.out_lr_x2 / reds_id
        hr_seq_dir.mkdir(parents=True, exist_ok=True)
        lr_seq_dir.mkdir(parents=True, exist_ok=True)

        for frame in src_frames:
            hr_out = hr_seq_dir / frame.name
            lr_out = lr_seq_dir / frame.name

            img = Image.open(frame).convert('RGB')
            img.save(hr_out)
            save_bicubic_x2(img, lr_out)

    print('Created local REDS-style validation set:')
    print(f'  HR root: {args.out_hr}')
    print(f'  LR x2 root: {args.out_lr_x2}')
    print(f'  sequences: {len(reds_ids)}')
    print(f'  source manifest: {args.source_manifest}')


if __name__ == '__main__':
    main()
