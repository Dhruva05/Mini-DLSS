#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Set, Tuple


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path}")
    items = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not items:
        raise ValueError(f"No entries found in: {path}")
    return items


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _split_train_val(entries: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1)")
    idx = list(range(len(entries)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_count = max(1, int(len(entries) * val_ratio))
    val_idx = set(idx[:val_count])
    train: List[str] = []
    val: List[str] = []
    for i, item in enumerate(entries):
        (val if i in val_idx else train).append(item)
    return train, val


def _write(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create union Vimeo-90K manifests from two Kaggle subsets")
    parser.add_argument("--vimeo-a", type=Path, required=True, help="Path containing sep_trainlist.txt + sep_testlist.txt")
    parser.add_argument("--vimeo-b", type=Path, required=True, help="Path containing sep_trainlist.txt + sep_testlist.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    a_train = _read_list(args.vimeo_a / "sep_trainlist.txt")
    b_train = _read_list(args.vimeo_b / "sep_trainlist.txt")
    a_test = _read_list(args.vimeo_a / "sep_testlist.txt")
    b_test = _read_list(args.vimeo_b / "sep_testlist.txt")

    train_union = _dedupe_preserve_order(a_train + b_train)
    test_union = _dedupe_preserve_order(a_test + b_test)

    train, val = _split_train_val(train_union, val_ratio=args.val_ratio, seed=args.seed)

    out = args.output_dir
    train_path = out / "vimeo90k_union_train.txt"
    val_path = out / "vimeo90k_union_val.txt"
    test_path = out / "vimeo90k_union_test.txt"

    _write(train_path, train)
    _write(val_path, val)
    _write(test_path, test_union)

    print("Wrote union manifests:")
    print(f"  train: {len(train)} -> {train_path}")
    print(f"  val:   {len(val)} -> {val_path}")
    print(f"  test:  {len(test_union)} -> {test_path}")


if __name__ == "__main__":
    main()
