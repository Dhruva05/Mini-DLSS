from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvGRUCell, ResidualBlock


class BasicVSRMini(nn.Module):
    """Minimal BasicVSR-style bidirectional temporal propagation.

    This is intentionally lightweight for reproducible training in Colab.
    """

    def __init__(self, scale: int = 2, base_channels: int = 64, num_blocks: int = 8) -> None:
        super().__init__()
        self.scale = scale

        self.feat_extractor = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_channels) for _ in range(max(1, num_blocks // 2))],
        )

        self.backward_gru = ConvGRUCell(base_channels)
        self.forward_gru = ConvGRUCell(base_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_channels) for _ in range(num_blocks)],
        )

        self.upsampler = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W]
        if x.dim() != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {tuple(x.shape)}")

        b, t, _, h, w = x.shape
        feats = [self.feat_extractor(x[:, i]) for i in range(t)]

        hidden_bwd = torch.zeros((b, feats[0].shape[1], h, w), dtype=feats[0].dtype, device=x.device)
        bwd_states = [None] * t
        for i in reversed(range(t)):
            hidden_bwd = self.backward_gru(feats[i], hidden_bwd)
            bwd_states[i] = hidden_bwd

        hidden_fwd = torch.zeros((b, feats[0].shape[1], h, w), dtype=feats[0].dtype, device=x.device)
        fwd_states = [None] * t
        for i in range(t):
            hidden_fwd = self.forward_gru(feats[i], hidden_fwd)
            fwd_states[i] = hidden_fwd

        center = t // 2
        fused = self.fusion(torch.cat([feats[center], fwd_states[center], bwd_states[center]], dim=1))
        return self.upsampler(fused)
