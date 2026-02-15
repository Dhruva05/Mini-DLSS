#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quality-latency tradeoff")
    parser.add_argument("--csv", type=Path, default=Path("results/tables/latency_quality_template.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/latency_quality_curve.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "ms_per_frame" not in df.columns or "psnr_y" not in df.columns:
        raise ValueError("CSV must contain ms_per_frame and psnr_y columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["ms_per_frame"], df["psnr_y"], s=80)
    for _, row in df.iterrows():
        ax.annotate(str(row.get("model", "model")), (row["ms_per_frame"], row["psnr_y"]))

    ax.set_xlabel("Latency (ms/frame)")
    ax.set_ylabel("PSNR-Y (dB)")
    ax.set_title("Mini-DLSS Quality vs Latency (2x)")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
