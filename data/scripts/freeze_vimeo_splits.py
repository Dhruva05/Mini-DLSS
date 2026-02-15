#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path}")
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No entries found in: {path}")
    return lines


def _split_train_val(entries: List[str], val_ratio: float, val_count: int | None, seed: int) -> Tuple[List[str], List[str]]:
    if val_count is None:
        val_count = max(1, int(len(entries) * val_ratio))
    if val_count >= len(entries):
        raise ValueError(
            f"val_count ({val_count}) must be smaller than number of train entries ({len(entries)})"
        )

    indices = list(range(len(entries)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_idx = set(indices[:val_count])
    train, val = [], []
    for i, item in enumerate(entries):
        (val if i in val_idx else train).append(item)
    return train, val


def _write_manifest(path: Path, entries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Vimeo-90K manifests for Mini-DLSS")
    parser.add_argument("--vimeo-root", type=Path, required=True, help="Path containing sep_trainlist.txt and sep_testlist.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"), help="Directory to write output manifests")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio from official train list")
    parser.add_argument("--val-count", type=int, default=None, help="Optional absolute val count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    train_list_path = args.vimeo_root / "sep_trainlist.txt"
    test_list_path = args.vimeo_root / "sep_testlist.txt"

    train_entries = _read_list(train_list_path)
    test_entries = _read_list(test_list_path)

    frozen_train, frozen_val = _split_train_val(
        entries=train_entries,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        seed=args.seed,
    )

    out = args.output_dir
    _write_manifest(out / "vimeo90k_train.txt", frozen_train)
    _write_manifest(out / "vimeo90k_val.txt", frozen_val)
    _write_manifest(out / "vimeo90k_test.txt", test_entries)

    print("Wrote Vimeo-90K split manifests:")
    print(f"  train: {len(frozen_train)} -> {out / 'vimeo90k_train.txt'}")
    print(f"  val:   {len(frozen_val)} -> {out / 'vimeo90k_val.txt'}")
    print(f"  test:  {len(test_entries)} -> {out / 'vimeo90k_test.txt'}")


if __name__ == "__main__":
    main()
