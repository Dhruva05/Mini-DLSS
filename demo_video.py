#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from utils import load_checkpoint, load_config

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mini-DLSS video demo on an LR mp4")
    parser.add_argument("--input", type=Path, required=True, help="LR input mp4")
    parser.add_argument("--output", type=Path, required=True, help="SR output mp4")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["model", "bicubic"], default="model")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _read_video_rgb(path: Path) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for demo_video.py")
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise ValueError(f"No frames loaded from {path}")
    return frames


def _write_video_rgb(path: Path, frames: List[np.ndarray], fps: float) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for demo_video.py")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    arr = torch.from_numpy(frame).float().to(device) / 255.0
    return arr.permute(2, 0, 1).contiguous()


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _window_indices(i: int, n: int, radius: int) -> List[int]:
    idxs = []
    for j in range(i - radius, i + radius + 1):
        if j < 0:
            idxs.append(0)
        elif j >= n:
            idxs.append(n - 1)
        else:
            idxs.append(j)
    return idxs


def main() -> None:
    args = parse_args()
    if cv2 is None:
        raise RuntimeError("OpenCV is not available; install opencv-python for demo_video.py")

    cfg = load_config(args.config)
    scale = int(cfg["project"]["scale"])
    num_frames = int(cfg["dataset"]["num_frames"])
    radius = num_frames // 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    frames = _read_video_rgb(args.input)
    tensors = [_to_tensor(f, device=device) for f in frames]
    n = len(tensors)

    model = None
    if args.mode == "model":
        model = build_model(cfg["model"], scale=scale).to(device)
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = Path("results") / "runs" / cfg["project"]["run_name"] / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        load_checkpoint(ckpt, model=model, optimizer=None, device=device)
        model.eval()

    out_frames: List[np.ndarray] = []
    t_start = time.time()

    with torch.no_grad():
        for i in range(n):
            idxs = _window_indices(i, n, radius)
            window = torch.stack([tensors[j] for j in idxs], dim=0).unsqueeze(0)
            center = window[:, radius]

            if args.mode == "bicubic":
                pred = F.interpolate(center, scale_factor=scale, mode="bicubic", align_corners=False)
            else:
                pred = model(window)  # type: ignore[misc]

            out_frames.append(_to_uint8(pred[0]))

    elapsed = time.time() - t_start
    ms_per_frame = (elapsed / n) * 1000.0

    _write_video_rgb(args.output, out_frames, fps=args.fps)

    print("Demo complete")
    print(f"  input: {args.input}")
    print(f"  output: {args.output}")
    print(f"  frames: {n}")
    print(f"  elapsed_sec: {elapsed:.3f}")
    print(f"  ms_per_frame: {ms_per_frame:.3f}")


if __name__ == "__main__":
    main()
