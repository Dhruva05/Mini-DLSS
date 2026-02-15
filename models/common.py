from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1)
        self.conv_h = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(concat))
        z, r = torch.chunk(zr, 2, dim=1)
        candidate = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * candidate
        return h_new
