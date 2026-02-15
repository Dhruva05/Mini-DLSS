from __future__ import annotations

import torch
import torch.nn as nn

from .common import ResidualBlock


class SingleFrameSR(nn.Module):
    """Simple residual SR baseline for frame-by-frame super-resolution."""

    def __init__(self, scale: int = 2, base_channels: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, base_channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, C, H, W] or [B, T, C, H, W] (uses center frame).
        if x.dim() == 5:
            x = x[:, x.shape[1] // 2]
        feat = self.head(x)
        feat = feat + self.body(feat)
        return self.tail(feat)
