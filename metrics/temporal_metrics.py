from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .image_metrics import compute_psnr

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _to_hwc_uint8(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        t = frame.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    else:
        arr = frame
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def _to_chw_float(frame: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        t = frame.detach().cpu().float()
    else:
        t = torch.from_numpy(frame).float()
    if t.ndim == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    if t.max() > 1.0:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def frame_difference_energy(frames: List[torch.Tensor | np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = []
    prev = _to_chw_float(frames[0])
    for cur in frames[1:]:
        cur_t = _to_chw_float(cur)
        diffs.append(torch.mean(torch.abs(cur_t - prev)).item())
        prev = cur_t
    return float(np.mean(diffs))


def temporal_error_energy(
    frames: List[torch.Tensor | np.ndarray],
    target_frames: List[torch.Tensor | np.ndarray],
) -> float:
    if len(frames) != len(target_frames):
        raise ValueError(f"Expected equal frame counts, got pred={len(frames)} target={len(target_frames)}")
    if len(frames) < 2:
        return 0.0

    errors = []
    prev_pred = _to_chw_float(frames[0])
    prev_target = _to_chw_float(target_frames[0])
    for pred, target in zip(frames[1:], target_frames[1:]):
        cur_pred = _to_chw_float(pred)
        cur_target = _to_chw_float(target)
        pred_delta = cur_pred - prev_pred
        target_delta = cur_target - prev_target
        errors.append(torch.mean(torch.abs(pred_delta - target_delta)).item())
        prev_pred = cur_pred
        prev_target = cur_target
    return float(np.mean(errors))


def temporal_psnr_flow_warp(frames: List[torch.Tensor | np.ndarray]) -> float | None:
    if cv2 is None or len(frames) < 2:
        return None

    scores = []
    prev = _to_hwc_uint8(frames[0])
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)

    for cur in frames[1:]:
        cur_arr = _to_hwc_uint8(cur)
        cur_gray = cv2.cvtColor(cur_arr, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            cur_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        h, w = prev_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)

        warped_cur = cv2.remap(cur_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        prev_t = _to_chw_float(prev)
        warped_t = _to_chw_float(warped_cur)
        scores.append(compute_psnr(warped_t, prev_t))

        prev = cur_arr
        prev_gray = cur_gray

    if not scores:
        return None
    return float(np.mean(scores))


def compute_temporal_metrics(
    frames: List[torch.Tensor | np.ndarray],
    target_frames: List[torch.Tensor | np.ndarray] | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["diff_energy"] = frame_difference_energy(frames)
    tpsnr = temporal_psnr_flow_warp(frames)
    if tpsnr is not None:
        out["tpsnr"] = tpsnr
    if target_frames is not None:
        if len(frames) != len(target_frames):
            raise ValueError(f"Expected equal frame counts, got pred={len(frames)} target={len(target_frames)}")
        target_diff = frame_difference_energy(target_frames)
        out["target_diff_energy"] = target_diff
        out["diff_energy_delta"] = out["diff_energy"] - target_diff
        out["temporal_error_energy"] = temporal_error_energy(frames, target_frames)

        target_tpsnr = temporal_psnr_flow_warp(target_frames)
        if tpsnr is not None and target_tpsnr is not None:
            out["target_tpsnr"] = target_tpsnr
            out["tpsnr_delta"] = tpsnr - target_tpsnr
    return out
