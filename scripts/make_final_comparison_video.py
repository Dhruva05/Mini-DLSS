#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import read_manifest
from models import build_model
from utils import load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a labeled final SR comparison video")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/reds_val.txt"))
    parser.add_argument("--hr-root", type=Path, default=Path("data/raw/reds/val_sharp"))
    parser.add_argument("--lr-root", type=Path, default=Path("data/raw/reds/val_sharp_bicubic/X2"))
    parser.add_argument("--output", type=Path, default=Path("results/final/videos/final_temporal_vsr_5f_small_2x/all_val_comparison_10s_labeled.mp4"))
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def frame_paths(root: Path, sequence_id: str) -> List[Path]:
    paths = [p for p in (root / sequence_id).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return sorted(paths)


def window_indices(i: int, n: int, radius: int) -> List[int]:
    return [min(max(j, 0), n - 1) for j in range(i - radius, i + radius + 1)]


def labeled_strip(panels: List[np.ndarray], labels: List[str]) -> np.ndarray:
    height, width = panels[0].shape[:2]
    header_h = 36
    strip = np.concatenate(panels, axis=1)
    canvas = np.zeros((height + header_h, strip.shape[1], 3), dtype=np.uint8)
    canvas[:header_h, :, :] = 24
    canvas[header_h:, :, :] = strip
    for i, label in enumerate(labels):
        x = i * width + 14
        cv2.putText(
            canvas,
            label,
            (x, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    scale = int(cfg["project"]["scale"])
    num_frames = int(cfg["dataset"]["num_frames"])
    radius = num_frames // 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(cfg["model"], scale=scale).to(device)
    load_checkpoint(args.checkpoint, model=model, optimizer=None, device=device)
    model.eval()

    frames: List[np.ndarray] = []
    labels = ["LR input", "Bicubic", "Temporal SR", "HR target"]
    with torch.no_grad():
        for sequence_id in read_manifest(args.manifest):
            hr_paths = frame_paths(args.hr_root, sequence_id)
            lr_paths = frame_paths(args.lr_root, sequence_id)
            if not hr_paths or not lr_paths:
                continue
            hr_tensors = [load_rgb(p) for p in hr_paths]
            lr_tensors = [load_rgb(p).to(device) for p in lr_paths]
            for i in range(len(hr_tensors)):
                idxs = window_indices(i, len(lr_tensors), radius)
                window = torch.stack([lr_tensors[j] for j in idxs], dim=0).unsqueeze(0)
                hr = hr_tensors[i]
                hr_size = (hr.shape[-2], hr.shape[-1])
                center = window[:, radius]
                bicubic = F.interpolate(center, size=hr_size, mode="bicubic", align_corners=False)[0]
                lr_up = F.interpolate(center, size=hr_size, mode="nearest")[0]
                pred = model(window)[0]
                frames.append(
                    labeled_strip(
                        [to_uint8(lr_up), to_uint8(bicubic), to_uint8(pred), to_uint8(hr)],
                        labels,
                    )
                )

    if not frames:
        raise RuntimeError("No frames were generated")

    fps = len(frames) / args.duration_sec if args.duration_sec > 0 else 24.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print("Comparison video complete")
    print(f"  output: {args.output}")
    print(f"  frames: {len(frames)}")
    print(f"  fps: {fps:.3f}")
    print(f"  duration_sec: {len(frames) / fps:.3f}")


if __name__ == "__main__":
    main()
