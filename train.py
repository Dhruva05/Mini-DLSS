#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.dataset import MultiFrameSRDataset
from metrics import compute_image_metrics
from models import build_model
from utils import load_checkpoint, load_config, save_checkpoint, seed_everything


class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.available = False
        self.features: nn.Module
        try:
            from torchvision.models import VGG16_Weights, vgg16

            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(model.features.children())[:16])
            self.features.eval()
            for p in self.features.parameters():
                p.requires_grad = False
            self.available = True
        except Exception:
            self.features = nn.Identity()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return torch.zeros((), device=pred.device)
        p = self.features(pred)
        t = self.features(target)
        return F.l1_loss(p, t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mini-DLSS models")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML/JSON config")
    parser.add_argument("--override", type=str, default="", help='JSON dict to override config, e.g. "{\"train\":{\"max_steps\":1000}}"')
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def make_dataloader(cfg: Dict, training: bool) -> DataLoader:
    dcfg = cfg["dataset"]
    split = "train" if training else "val"

    hr_root = Path(dcfg[f"{split}_hr_root"])
    lr_root_raw = dcfg.get(f"{split}_lr_root", "")
    lr_root = Path(lr_root_raw) if lr_root_raw else None

    dataset = MultiFrameSRDataset(
        hr_root=hr_root,
        lr_root=lr_root,
        manifest_path=Path(dcfg[f"{split}_manifest"]),
        num_frames=int(dcfg["num_frames"]),
        scale=int(cfg["project"]["scale"]),
        generate_lr_on_the_fly=bool(dcfg.get("generate_lr_on_the_fly", False)),
        random_crop_size=int(dcfg.get("crop_size_hr", 0)) or None,
        training=training,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=training,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=training,
    )
    return loader


def cycle(loader: DataLoader) -> Iterator[Dict]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, scale: int, device: str, max_batches: int = 0) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = defaultdict(float)
    count = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred = model(lr).clamp(0.0, 1.0)

        for b in range(pred.shape[0]):
            m = compute_image_metrics(pred[b], hr[b], scale=scale, report_rgb=True)
            for k, v in m.items():
                agg[k] += float(v)
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader produced zero samples")

    return {k: v / count for k, v in agg.items()}


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.override:
        override = json.loads(args.override)
        from utils.config import deep_update

        cfg = deep_update(cfg, override)

    if args.device is not None:
        cfg["project"]["device"] = args.device

    device = cfg["project"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seed_everything(int(cfg["project"]["seed"]))
    scale = int(cfg["project"]["scale"])
    run_name = cfg["project"]["run_name"]

    run_dir = Path("results") / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_dataloader(cfg, training=True)
    val_loader = make_dataloader(cfg, training=False)
    train_iter = cycle(train_loader)

    model = build_model(cfg["model"], scale=scale).to(device)
    optimizer = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    l1 = nn.L1Loss()

    loss_name = cfg["train"].get("loss", "l1")
    perceptual = PerceptualLoss().to(device)
    perceptual_weight = float(cfg["train"].get("perceptual_weight", 0.0))
    if "perceptual" in loss_name and not perceptual.available:
        print("[warn] perceptual loss requested but torchvision/VGG unavailable; using L1 only")

    step = 0
    best_psnr_y: float | None = None

    resume_path = cfg["train"].get("resume", "")
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path), model, optimizer=optimizer, device=device)
        step = int(ckpt.get("step", 0))
        best_psnr_y = ckpt.get("best_metric", None)
        print(f"Resumed from step={step} checkpoint={resume_path}")

    max_steps = int(cfg["train"]["max_steps"])
    log_interval = int(cfg["train"].get("log_interval", 100))
    val_interval = int(cfg["train"].get("val_interval", 2000))
    ckpt_interval = int(cfg["train"].get("checkpoint_interval", 2000))

    metrics_log_path = run_dir / "train_log.jsonl"
    start_time = time.time()

    model.train()
    while step < max_steps:
        batch = next(train_iter)
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(lr)

        loss = l1(pred, hr)
        if "perceptual" in loss_name and perceptual_weight > 0.0:
            loss = loss + perceptual_weight * perceptual(pred, hr)

        loss.backward()
        optimizer.step()

        step += 1

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            samples = step * int(cfg["train"]["batch_size"])
            msg = {
                "step": step,
                "loss": float(loss.item()),
                "elapsed_sec": elapsed,
                "samples_seen": samples,
            }
            with metrics_log_path.open("a") as f:
                f.write(json.dumps(msg) + "\n")
            print(f"step={step} loss={loss.item():.4f} elapsed={elapsed:.1f}s")

        if step % val_interval == 0 or step == max_steps:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                scale=scale,
                device=device,
                max_batches=int(cfg["eval"].get("max_batches", 0)),
            )
            print("val", " ".join([f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())]))

            is_best = best_psnr_y is None or val_metrics["psnr_y"] > best_psnr_y
            if is_best:
                best_psnr_y = val_metrics["psnr_y"]
                save_checkpoint(
                    path=ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_metric=best_psnr_y,
                    config=cfg,
                )

            save_checkpoint(
                path=ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_metric=best_psnr_y,
                config=cfg,
            )

            with (run_dir / "val_metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **val_metrics}) + "\n")

            model.train()

        if step % ckpt_interval == 0:
            save_checkpoint(
                path=ckpt_dir / f"step_{step:07d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_metric=best_psnr_y,
                config=cfg,
            )

    print(f"Training done: run_dir={run_dir}")


if __name__ == "__main__":
    main()
