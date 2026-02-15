from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SequenceRecord:
    sequence_id: str
    hr_frames: List[Path]
    lr_frames: Optional[List[Path]]


@dataclass
class SampleIndex:
    sequence_idx: int
    center_idx: int


def read_manifest(manifest_path: Path) -> List[str]:
    ids: List[str] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    if not ids:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return ids


def _list_frames(sequence_dir: Path) -> List[Path]:
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence dir not found: {sequence_dir}")
    frames = [p for p in sequence_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    frames.sort()
    if not frames:
        raise ValueError(f"No frames found in: {sequence_dir}")
    return frames


def _load_image(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _downsample_bicubic(image: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return image
    image = image.unsqueeze(0)
    h, w = image.shape[-2:]
    lr_h, lr_w = h // scale, w // scale
    if lr_h <= 0 or lr_w <= 0:
        raise ValueError(f"Invalid image size for scale={scale}: {(h, w)}")
    return F.interpolate(image, size=(lr_h, lr_w), mode="bicubic", align_corners=False).squeeze(0)


class MultiFrameSRDataset(Dataset):
    """Generic temporal SR dataset using sequence directories and manifest file.

    Expected structure:
      hr_root/<sequence_id>/<frame_files>
      lr_root/<sequence_id>/<frame_files> (optional)

    Manifest lines are relative sequence ids (e.g., "000", "00001/0001").
    """

    def __init__(
        self,
        hr_root: Path,
        manifest_path: Path,
        num_frames: int,
        scale: int,
        lr_root: Optional[Path] = None,
        generate_lr_on_the_fly: bool = False,
        random_crop_size: Optional[int] = None,
        training: bool = True,
    ) -> None:
        if num_frames % 2 == 0:
            raise ValueError("num_frames must be odd")
        if scale < 1:
            raise ValueError("scale must be >= 1")

        self.hr_root = hr_root
        self.lr_root = lr_root
        self.generate_lr_on_the_fly = generate_lr_on_the_fly
        self.num_frames = num_frames
        self.radius = num_frames // 2
        self.scale = scale
        self.random_crop_size = random_crop_size
        self.training = training

        sequence_ids = read_manifest(manifest_path)
        self.sequences: List[SequenceRecord] = []
        for seq_id in sequence_ids:
            hr_frames = _list_frames(hr_root / seq_id)
            lr_frames = None
            if lr_root is not None:
                lr_frames = _list_frames(lr_root / seq_id)
                if len(lr_frames) != len(hr_frames):
                    raise ValueError(
                        f"Frame count mismatch for sequence {seq_id}: "
                        f"lr={len(lr_frames)} hr={len(hr_frames)}"
                    )
            self.sequences.append(
                SequenceRecord(sequence_id=seq_id, hr_frames=hr_frames, lr_frames=lr_frames)
            )

        self.samples: List[SampleIndex] = []
        for seq_idx, record in enumerate(self.sequences):
            frame_count = len(record.hr_frames)
            for center_idx in range(self.radius, frame_count - self.radius):
                self.samples.append(SampleIndex(sequence_idx=seq_idx, center_idx=center_idx))

        if not self.samples:
            raise ValueError(
                f"No temporal samples built. Check manifest={manifest_path}, num_frames={num_frames}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_crop(
        self, lr_window: torch.Tensor, hr_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.random_crop_size is None:
            return lr_window, hr_window

        crop_hr = (self.random_crop_size // self.scale) * self.scale
        if crop_hr <= 0:
            return lr_window, hr_window
        hr_h, hr_w = hr_window.shape[-2:]
        if crop_hr > hr_h or crop_hr > hr_w:
            return lr_window, hr_window

        top = torch.randint(0, hr_h - crop_hr + 1, size=(1,)).item()
        left = torch.randint(0, hr_w - crop_hr + 1, size=(1,)).item()
        hr_window = hr_window[:, :, top : top + crop_hr, left : left + crop_hr]

        lr_crop = crop_hr // self.scale
        lr_top = top // self.scale
        lr_left = left // self.scale
        lr_window = lr_window[:, :, lr_top : lr_top + lr_crop, lr_left : lr_left + lr_crop]
        return lr_window, hr_window

    def _augment(self, lr_window: torch.Tensor, hr_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return lr_window, hr_window

        if torch.rand(1).item() < 0.5:
            lr_window = torch.flip(lr_window, dims=[-1])
            hr_window = torch.flip(hr_window, dims=[-1])
        if torch.rand(1).item() < 0.5:
            lr_window = torch.flip(lr_window, dims=[-2])
            hr_window = torch.flip(hr_window, dims=[-2])
        if torch.rand(1).item() < 0.5:
            lr_window = torch.flip(lr_window, dims=[0])
            hr_window = torch.flip(hr_window, dims=[0])
        return lr_window, hr_window

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        record = self.sequences[sample.sequence_idx]

        start = sample.center_idx - self.radius
        end = sample.center_idx + self.radius + 1
        hr_paths = record.hr_frames[start:end]
        hr_window = torch.stack([_load_image(p) for p in hr_paths], dim=0)

        if record.lr_frames is not None:
            lr_paths = record.lr_frames[start:end]
            lr_window = torch.stack([_load_image(p) for p in lr_paths], dim=0)
        elif self.generate_lr_on_the_fly:
            lr_window = torch.stack([_downsample_bicubic(im, self.scale) for im in hr_window], dim=0)
        else:
            raise ValueError(
                "lr_root is missing and generate_lr_on_the_fly is False; cannot build LR input"
            )

        lr_window, hr_window = self._maybe_crop(lr_window, hr_window)
        lr_window, hr_window = self._augment(lr_window, hr_window)

        target = hr_window[self.radius]
        frame_name = record.hr_frames[sample.center_idx].name

        return {
            "lr": lr_window,
            "hr": target,
            "sequence_id": record.sequence_id,
            "frame_name": frame_name,
        }

    def sequence_stats(self) -> List[Tuple[str, int, Optional[int]]]:
        stats: List[Tuple[str, int, Optional[int]]] = []
        for seq in self.sequences:
            hr_count = len(seq.hr_frames)
            lr_count = len(seq.lr_frames) if seq.lr_frames is not None else None
            stats.append((seq.sequence_id, hr_count, lr_count))
        return stats
