#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import MultiFrameSRDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset alignment and temporal sampling")
    parser.add_argument("--hr-root", type=Path, required=True)
    parser.add_argument("--lr-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--generate-lr-on-the-fly", action="store_true")
    parser.add_argument("--preview", type=int, default=5)
    args = parser.parse_args()

    dataset = MultiFrameSRDataset(
        hr_root=args.hr_root,
        lr_root=args.lr_root,
        manifest_path=args.manifest,
        num_frames=args.num_frames,
        scale=args.scale,
        generate_lr_on_the_fly=args.generate_lr_on_the_fly,
        training=False,
    )

    stats = dataset.sequence_stats()
    print("Dataset check summary")
    print(f"  manifest: {args.manifest}")
    print(f"  sequences: {len(stats)}")
    print(f"  temporal windows (samples): {len(dataset)}")
    print(f"  num_frames: {args.num_frames} (radius={args.num_frames // 2})")
    print(f"  scale: {args.scale}")

    mismatches = []
    hr_total = 0
    for seq_id, hr_count, lr_count in stats:
        hr_total += hr_count
        if lr_count is not None and hr_count != lr_count:
            mismatches.append((seq_id, hr_count, lr_count))

    print(f"  total HR frames: {hr_total}")
    if args.lr_root is not None:
        print(f"  alignment mismatches: {len(mismatches)}")
        for seq_id, hr_count, lr_count in mismatches[:10]:
            print(f"    - {seq_id}: hr={hr_count}, lr={lr_count}")

    print("Preview samples")
    for i in range(min(args.preview, len(dataset))):
        sample = dataset[i]
        lr = sample["lr"]
        hr = sample["hr"]
        print(
            f"  [{i}] seq={sample['sequence_id']} frame={sample['frame_name']} "
            f"lr={tuple(lr.shape)} hr={tuple(hr.shape)}"
        )


if __name__ == "__main__":
    main()
