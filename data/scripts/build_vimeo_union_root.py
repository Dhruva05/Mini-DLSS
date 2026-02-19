#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _symlink_children(src: Path, dst: Path) -> tuple[int, int]:
    created = 0
    skipped = 0
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        target = dst / child.name
        if target.exists() or target.is_symlink():
            skipped += 1
            continue
        target.symlink_to(child.resolve())
        created += 1
    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged Vimeo sequence root from two disjoint subsets")
    parser.add_argument("--seq-a", type=Path, required=True, help="Path to subset A sequence dir")
    parser.add_argument("--seq-b", type=Path, required=True, help="Path to subset B sequence dir")
    parser.add_argument("--output", type=Path, default=Path("data/raw/vimeo90k_union/sequence"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    a_created, a_skipped = _symlink_children(args.seq_a, args.output)
    b_created, b_skipped = _symlink_children(args.seq_b, args.output)

    print("Union root ready:")
    print(f"  output: {args.output}")
    print(f"  from A: created={a_created} skipped={a_skipped}")
    print(f"  from B: created={b_created} skipped={b_skipped}")


if __name__ == "__main__":
    main()
