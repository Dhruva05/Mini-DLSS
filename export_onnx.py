#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import build_model
from utils import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Mini-DLSS model to ONNX")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/onnx/mini_dlss.onnx"))
    parser.add_argument("--height", type=int, default=180, help="LR input height for sample export")
    parser.add_argument("--width", type=int, default=320, help="LR input width for sample export")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    scale = int(cfg["project"]["scale"])
    num_frames = int(cfg["dataset"]["num_frames"])

    model = build_model(cfg["model"], scale=scale).to(args.device)
    load_checkpoint(args.checkpoint, model, optimizer=None, device=args.device)
    model.eval()

    dummy = torch.randn(1, num_frames, 3, args.height, args.width, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["lr_frames"],
        output_names=["sr_frame"],
        dynamic_axes={
            "lr_frames": {0: "batch", 3: "height", 4: "width"},
            "sr_frame": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model: {args.output}")


if __name__ == "__main__":
    main()
