#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from utils import load_config

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Mini-DLSS ONNX Runtime inference on an LR mp4")
    parser.add_argument("--onnx", type=Path, default=Path("results/onnx/temporal_vsr_5f_small_2x.onnx"))
    parser.add_argument("--input", type=Path, default=Path("results/week3/latency_input_10s_lr.mp4"))
    parser.add_argument("--config", type=Path, default=Path("configs/temporal_small.toml"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/final/tables/onnxruntime_temporal_vsr_5f_small_2x_latency.json"),
    )
    parser.add_argument("--output-video", type=Path, default=None, help="Optional SR mp4 from the last timed run")
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args()


def _read_video_rgb(path: Path) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required; install opencv-python")
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
        raise RuntimeError("OpenCV is required; install opencv-python")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _to_chw_float(frame: np.ndarray) -> np.ndarray:
    arr = frame.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)).copy()


def _to_uint8_hwc(tensor: np.ndarray) -> np.ndarray:
    arr = np.clip(tensor, 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    return np.round(arr * 255.0).astype(np.uint8)


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


def _make_window(tensors: List[np.ndarray], i: int, radius: int) -> np.ndarray:
    idxs = _window_indices(i, len(tensors), radius)
    return np.stack([tensors[j] for j in idxs], axis=0)[None]


def main() -> None:
    args = parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime is required; install it with `pip install onnxruntime`")
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    cfg = load_config(args.config)
    num_frames = int(cfg["dataset"]["num_frames"])
    radius = num_frames // 2

    frames = _read_video_rgb(args.input)
    tensors = [_to_chw_float(frame) for frame in frames]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=session_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    first_window = _make_window(tensors, 0, radius)
    for _ in range(max(0, args.warmup)):
        session.run([output_name], {input_name: first_window})

    output_frames: List[np.ndarray] = []
    elapsed_total = 0.0
    for run_idx in range(max(1, args.runs)):
        run_outputs: List[np.ndarray] = []
        start = time.perf_counter()
        for i in range(len(tensors)):
            window = _make_window(tensors, i, radius)
            pred = session.run([output_name], {input_name: window})[0]
            if args.output_video is not None and run_idx == max(1, args.runs) - 1:
                run_outputs.append(_to_uint8_hwc(pred))
        elapsed_total += time.perf_counter() - start
        if run_outputs:
            output_frames = run_outputs

    frames_timed = len(tensors) * max(1, args.runs)
    ms_per_frame = (elapsed_total / frames_timed) * 1000.0
    result = {
        "onnx": str(args.onnx),
        "input": str(args.input),
        "config": str(args.config),
        "frames": len(tensors),
        "runs": max(1, args.runs),
        "warmup": max(0, args.warmup),
        "elapsed_sec": elapsed_total,
        "ms_per_frame": ms_per_frame,
        "providers_requested": providers,
        "providers_used": session.get_providers(),
        "input_name": input_name,
        "output_name": output_name,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.output_video is not None:
        _write_video_rgb(args.output_video, output_frames, fps=args.fps)

    print("ONNX Runtime benchmark complete")
    print(f"  frames: {result['frames']}")
    print(f"  runs: {result['runs']}")
    print(f"  elapsed_sec: {result['elapsed_sec']:.3f}")
    print(f"  ms_per_frame: {result['ms_per_frame']:.3f}")
    print(f"  providers_used: {', '.join(result['providers_used'])}")
    print(f"  output_json: {args.output_json}")


if __name__ == "__main__":
    main()
