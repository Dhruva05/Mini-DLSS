from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor_chw(img: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    else:
        t = img.detach().cpu()

    if t.dtype != torch.float32:
        t = t.float()

    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 4D tensor, got {tuple(t.shape)}")
        t = t[0]

    if t.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tuple(t.shape)}")

    # Convert HWC -> CHW if needed.
    if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)

    if t.max() > 1.0:
        t = t / 255.0

    return t.clamp(0.0, 1.0)


def crop_border(img: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return img
    return img[:, border:-border, border:-border]


def rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """Convert CHW RGB [0,1] to Y channel [0,1] using BT.601 (MATLAB-style)."""
    if img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {img.shape[0]}")
    r, g, b = img[0], img[1], img[2]
    y = (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0
    return y.unsqueeze(0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= eps:
        return 100.0
    return float(-10.0 * np.log10(mse))


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    return window_2d.repeat(channels, 1, 1, 1)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM for CHW tensors with values in [0,1]."""
    channels = pred.shape[0]
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)

    window = _gaussian_window(channels=channels).to(pred.dtype)
    padding = 11 // 2

    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


def compute_image_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    scale: int,
    report_rgb: bool = True,
) -> Dict[str, float]:
    p = _to_tensor_chw(pred)
    t = _to_tensor_chw(target)

    border = int(scale)
    p = crop_border(p, border)
    t = crop_border(t, border)

    py = rgb_to_y_channel(p)
    ty = rgb_to_y_channel(t)

    out: Dict[str, float] = {
        "psnr_y": compute_psnr(py, ty),
        "ssim_y": compute_ssim(py, ty),
    }

    if report_rgb:
        out["psnr_rgb"] = compute_psnr(p, t)
        out["ssim_rgb"] = compute_ssim(p, t)

    return out
