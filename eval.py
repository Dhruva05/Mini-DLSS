#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import MultiFrameSRDataset, read_manifest
from metrics import compute_image_metrics, compute_temporal_metrics
from models import build_model
from utils import load_checkpoint, load_config, seed_everything, write_markdown_table

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mini-DLSS models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=str, default="", help='JSON dict to override config')
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["auto", "bicubic", "model"], default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--save-videos", type=int, default=2)
    parser.add_argument("--demo-manifest", type=Path, default=Path("data/splits/reds_demo_clips.txt"))
    return parser.parse_args()


def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    arr = t.detach().cpu().clamp(0.0, 1.0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _bicubic_center(lr: torch.Tensor, hr_size: Tuple[int, int]) -> torch.Tensor:
    center = lr[:, lr.shape[1] // 2]
    return F.interpolate(center, size=hr_size, mode="bicubic", align_corners=False)

def _nearest_center(lr: torch.Tensor, hr_size: Tuple[int, int]) -> torch.Tensor:
    center = lr[:, lr.shape[1] // 2]
    return F.interpolate(center, size=hr_size, mode="nearest")


def _compose_strip(lr_up: torch.Tensor, bicubic: torch.Tensor, pred: torch.Tensor, hr: torch.Tensor) -> np.ndarray:
    panels = [_to_uint8_hwc(x) for x in [lr_up, bicubic, pred, hr]]
    return np.concatenate(panels, axis=1)


def _write_video_mp4(frames: List[np.ndarray], path: Path, fps: int = 24) -> None:
    if cv2 is None:
        print(f"[warn] cv2 missing, cannot write video: {path}")
        return
    if not frames:
        return
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        from utils.config import deep_update

        cfg = deep_update(cfg, json.loads(args.override))
    seed_everything(int(cfg["project"]["seed"]))

    device = args.device or cfg["project"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    scale = int(cfg["project"]["scale"])
    run_name = cfg["project"]["run_name"]

    dcfg = cfg["dataset"]
    val_dataset = MultiFrameSRDataset(
        hr_root=Path(dcfg["val_hr_root"]),
        lr_root=Path(dcfg["val_lr_root"]) if dcfg.get("val_lr_root", "") else None,
        manifest_path=Path(dcfg["val_manifest"]),
        num_frames=int(dcfg["num_frames"]),
        scale=scale,
        generate_lr_on_the_fly=bool(dcfg.get("generate_lr_on_the_fly", False)),
        training=False,
    )
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    use_bicubic = args.mode == "bicubic"

    model = None
    if args.mode in {"auto", "model"}:
        model = build_model(cfg["model"], scale=scale).to(device)
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = Path("results") / "runs" / run_name / "checkpoints" / "best.pt"
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model=model, optimizer=None, device=device)
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"[warn] checkpoint not found at {ckpt_path}, falling back to bicubic mode")
            use_bicubic = True
            model = None

    if args.mode == "bicubic":
        use_bicubic = True
        model = None

    if model is not None:
        model.eval()

    out_root = args.output_dir
    table_out = out_root / "tables" / f"{run_name}_eval.md"
    image_out = out_root / "videos" / run_name / "images"
    video_out = out_root / "videos" / run_name
    image_out.mkdir(parents=True, exist_ok=True)

    demo_ids: List[str] = []
    if args.demo_manifest.exists():
        demo_ids = read_manifest(args.demo_manifest)

    agg: Dict[str, float] = defaultdict(float)
    count = 0

    temporal_buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    target_temporal_buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
    demo_buckets: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)

    for idx, batch in enumerate(loader):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        hr_size = (hr.shape[-2], hr.shape[-1])
        bicubic = _bicubic_center(lr, hr_size).clamp(0.0, 1.0)
        lr_up = _nearest_center(lr, hr_size).clamp(0.0, 1.0)

        if use_bicubic:
            pred = bicubic
        else:
            with torch.no_grad():
                pred = model(lr).clamp(0.0, 1.0)  # type: ignore[misc]

        m = compute_image_metrics(pred[0], hr[0], scale=scale, report_rgb=True)
        for k, v in m.items():
            agg[k] += float(v)
        count += 1

        seq_id = batch["sequence_id"][0]
        frame_name = batch["frame_name"][0]
        temporal_buckets[seq_id].append((frame_name, pred[0].detach().cpu()))
        target_temporal_buckets[seq_id].append((frame_name, hr[0].detach().cpu()))

        strip = _compose_strip(
            lr_up[0].detach().cpu(),
            bicubic[0].detach().cpu(),
            pred[0].detach().cpu(),
            hr[0].detach().cpu(),
        )

        if idx < args.save_images:
            from PIL import Image

            Image.fromarray(strip).save(image_out / f"sample_{idx:04d}.png")

        if seq_id in demo_ids:
            demo_buckets[seq_id].append((frame_name, strip))

    if count == 0:
        raise RuntimeError("No samples were evaluated")

    metrics = {k: v / count for k, v in agg.items()}

    temporal_scores: Dict[str, float] = defaultdict(float)
    temporal_count = 0
    for seq_id, frames in temporal_buckets.items():
        frames = sorted(frames, key=lambda x: x[0])
        targets = sorted(target_temporal_buckets[seq_id], key=lambda x: x[0])
        seq_metrics = compute_temporal_metrics([f[1] for f in frames], target_frames=[f[1] for f in targets])
        for k, v in seq_metrics.items():
            temporal_scores[k] += v
        temporal_count += 1

    if temporal_count > 0:
        for k, v in temporal_scores.items():
            metrics[k] = v / temporal_count

    rows = [[k, f"{v:.4f}"] for k, v in sorted(metrics.items())]
    write_markdown_table(table_out, headers=["metric", "value"], rows=rows)

    json_out = out_root / "tables" / f"{run_name}_eval.json"
    json_out.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.save_videos > 0:
        made = 0
        for seq_id in demo_ids:
            if seq_id not in demo_buckets:
                continue
            if made >= args.save_videos:
                break
            seq_frames = sorted(demo_buckets[seq_id], key=lambda x: x[0])
            _write_video_mp4([x[1] for x in seq_frames], video_out / f"{seq_id}_comparison.mp4")
            made += 1

    print("Evaluation complete")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"Table: {table_out}")


if __name__ == "__main__":
    main()
