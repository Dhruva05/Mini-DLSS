from __future__ import annotations

from typing import Dict, Any

import torch.nn as nn

from .single_frame import SingleFrameSR
from .temporal_basicvsr import BasicVSRMini


def build_model(model_cfg: Dict[str, Any], scale: int) -> nn.Module:
    name = model_cfg.get("name", "temporal_small")
    channels = int(model_cfg.get("base_channels", 64))
    blocks = int(model_cfg.get("num_blocks", 8))

    if name == "single_frame":
        return SingleFrameSR(scale=scale, base_channels=channels, num_blocks=blocks)
    if name in {"temporal", "temporal_small"}:
        return BasicVSRMini(scale=scale, base_channels=channels, num_blocks=blocks)

    raise ValueError(f"Unknown model name: {name}")
