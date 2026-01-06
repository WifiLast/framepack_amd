from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def _nearest_interp(src: Sequence[float], target_length: int) -> np.ndarray:
    """
    Nearest-neighbor interpolation used by the original MagCache implementation.
    Preserves the trend of the empirical magnitudes when we need to adapt to a
    different number of sampler steps.
    """
    src_array = np.asarray(src, dtype=np.float32)
    src_length = src_array.shape[0]
    if target_length <= 1 or src_length == target_length:
        if target_length == src_length:
            return src_array.copy()
        return np.ones(target_length, dtype=np.float32) * src_array[-1]

    scale = (src_length - 1) / max(target_length - 1, 1)
    mapped_indices = np.round(np.arange(target_length, dtype=np.float32) * scale).astype(int)
    mapped_indices = np.clip(mapped_indices, 0, src_length - 1)
    return src_array[mapped_indices]


_FRAMEPACK_BASE_RATIOS = np.array(
    [
        1.0,
        1.26562,
        1.23438,
        1.03125,
        1.02344,
        1.03906,
        1.01562,
        1.03906,
        1.05469,
        1.02344,
        1.03906,
        0.99609,
        1.03125,
        1.02344,
        1.01562,
        1.02344,
        1.00781,
        1.04688,
        0.98828,
        1.0,
        1.00781,
        0.98828,
        0.94141,
        0.94141,
        0.78906,
    ],
    dtype=np.float32,
)

DEFAULT_MAG_RATIOS: Dict[int, np.ndarray] = {
    640: _FRAMEPACK_BASE_RATIOS,
}


def get_default_mag_ratios(video_height: int) -> np.ndarray:
    """
    Returns the empirical magnitude ratios calibrated for the FramePack reference prompt.
    We reuse the closest known resolution (defaults to 640p) when heights differ.
    """
    if not DEFAULT_MAG_RATIOS:
        return np.ones(1, dtype=np.float32)

    closest_height = min(DEFAULT_MAG_RATIOS.keys(), key=lambda h: abs(h - video_height))
    return DEFAULT_MAG_RATIOS[closest_height]


@dataclass
class MagCacheConfig:
    enabled: bool = False
    threshold: float = 0.18
    max_skip_steps: int = 2
    retention_ratio: float = 0.2
    default_height: int = 640
    custom_ratios: Optional[Sequence[float]] = None
    verbose: bool = False
    cache_on_cpu: bool = False  # optional trade-off for memory vs. bandwidth
    max_ratio_deviation: float = 0.06


class MagCacheVideo:
    """
    Diffusers-friendly MagCache helper. It mirrors the logic from the original MagCache implementation
    (see cache/MagCache-main/MagCache4HunyuanVideo/magcache_sample_video.py) but exposes clean hooks
    for the FramePack transformer.

    Usage:
        cache = MagCacheVideo(MagCacheConfig(enabled=True))
        cache.configure(num_steps=25, video_height=720)
        hidden_states, encoder_states, skipped = cache.maybe_skip(hidden_states, encoder_hidden_states)
        if not skipped:
            ... run expensive transformer blocks ...
            hidden_states, encoder_states = cache.finalize(hidden_states, encoder_states)
    """

    def __init__(self, config: Optional[MagCacheConfig] = None):
        self.config = config or MagCacheConfig()
        self.num_steps: int = 0
        self.retention_steps: int = 0
        self.mag_ratios: Optional[torch.Tensor] = None

        self.step_index: int = 0
        self.accumulated_ratio: float = 1.0
        self.accumulated_err: float = 0.0
        self.accumulated_steps: int = 0

        self.reference_hidden: Optional[torch.Tensor] = None
        self.reference_encoder: Optional[torch.Tensor] = None
        self.residual_hidden: Optional[torch.Tensor] = None
        self.residual_encoder: Optional[torch.Tensor] = None
        self.skip_active: bool = False

        self.stats: Dict[str, int] = {"skips": 0, "full_pass": 0}
        self._last_error: float = 0.0

    def update_config(self, config: MagCacheConfig):
        self.config = config
        if not self.config.enabled:
            self.clear(runtime_only=False)

    def configure(
        self,
        num_steps: int,
        video_height: Optional[int] = None,
        custom_ratios: Optional[Sequence[float]] = None,
    ):
        """
        Must be called whenever the sampler length or resolution changes.
        """
        if not self.config.enabled:
            self.clear(runtime_only=False)
            return

        num_steps = max(1, int(num_steps))
        ratios = custom_ratios or self.config.custom_ratios
        if ratios is None:
            target_height = video_height or self.config.default_height
            ratios = get_default_mag_ratios(target_height)

        ratio_array = np.asarray(ratios, dtype=np.float32)
        if ratio_array.shape[0] != num_steps:
            ratio_array = _nearest_interp(ratio_array, num_steps)

        self.mag_ratios = torch.from_numpy(ratio_array)
        self.num_steps = num_steps
        self.retention_steps = max(0, int(self.config.retention_ratio * num_steps))
        self.reset_cycle()

    def reset_cycle(self):
        self.step_index = 0
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0
        self.skip_active = False
        self.reference_hidden = None
        self.reference_encoder = None
        self._last_error = 0.0

    def clear(self, runtime_only: bool = True):
        self.residual_hidden = None
        self.residual_encoder = None
        self.reference_hidden = None
        self.reference_encoder = None
        self.skip_active = False
        self.stats = {"skips": 0, "full_pass": 0}
        self._last_error = 0.0
        if not runtime_only:
            self.num_steps = 0
            self.retention_steps = 0
            self.mag_ratios = None

    def maybe_skip(
        self,
        hidden_states: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Checks whether the current denoising step can reuse the cached residual.
        Returns the (possibly adjusted) hidden states, encoder states, and a boolean flag
        indicating if the expensive transformer blocks should be skipped.
        """
        if (
            not self.config.enabled
            or hidden_states is None
            or self.mag_ratios is None
            or self.num_steps <= 0
        ):
            self.skip_active = False
            return hidden_states, encoder_hidden_states, False

        skip_forward = False
        if (
            self.step_index >= self.retention_steps
            and self.residual_hidden is not None
            and self.accumulated_steps < self.config.max_skip_steps
        ):
            ratio_idx = min(self.step_index, int(self.mag_ratios.shape[0] - 1))
            cur_mag_ratio = float(self.mag_ratios[ratio_idx].item())
            if abs(1.0 - cur_mag_ratio) <= self.config.max_ratio_deviation:
                self.accumulated_ratio *= cur_mag_ratio
                cur_error = abs(1.0 - self.accumulated_ratio)
                self.accumulated_err += cur_error
                self.accumulated_steps += 1
                self._last_error = cur_error

                if self.accumulated_err <= self.config.threshold:
                    skip_forward = True
                    hidden_states = hidden_states + self._to_like(self.residual_hidden, hidden_states)
                    if encoder_hidden_states is not None and self.residual_encoder is not None:
                        encoder_hidden_states = encoder_hidden_states + self._to_like(
                            self.residual_encoder,
                            encoder_hidden_states,
                        )
                else:
                    self._reset_error_trackers()
            else:
                self._reset_error_trackers()

        if skip_forward:
            self.stats["skips"] += 1
            self.skip_active = True
            self.reference_hidden = None
            self.reference_encoder = None
            if self.config.verbose:
                print(
                    f"[MagCache] Skip step={self.step_index}, "
                    f"acc_err={self.accumulated_err:.4f}, "
                    f"acc_ratio={self.accumulated_ratio:.4f}"
                )
            self._advance_step()
            self.skip_active = False
            return hidden_states, encoder_hidden_states, True

        # fall back to the full forward pass
        self.stats["full_pass"] += 1
        self.skip_active = False
        self.reference_hidden = hidden_states.detach()
        self.reference_encoder = (
            encoder_hidden_states.detach() if encoder_hidden_states is not None else None
        )
        return hidden_states, encoder_hidden_states, False

    def finalize(
        self,
        hidden_states: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Commits the residual produced by a full transformer pass.
        Must be called after maybe_skip() when skip=False.
        """
        if (
            not self.config.enabled
            or hidden_states is None
            or self.mag_ratios is None
            or self.num_steps <= 0
        ):
            return hidden_states, encoder_hidden_states

        if not self.skip_active and self.reference_hidden is not None:
            resid = (hidden_states - self.reference_hidden).detach()
            self.residual_hidden = self._maybe_move_to_cache_device(resid)
            if encoder_hidden_states is not None and self.reference_encoder is not None:
                enc_resid = (encoder_hidden_states - self.reference_encoder).detach()
                self.residual_encoder = self._maybe_move_to_cache_device(enc_resid)
            else:
                self.residual_encoder = None

        self._advance_step()
        self.reference_hidden = None
        self.reference_encoder = None
        self.skip_active = False
        return hidden_states, encoder_hidden_states

    def _advance_step(self):
        self.step_index += 1
        if self.step_index >= self.num_steps:
            self.step_index = 0
            self._reset_error_trackers()
        elif not self.skip_active:
            self.accumulated_steps = 0
            self.accumulated_ratio = 1.0
            self.accumulated_err = 0.0

    def _reset_error_trackers(self):
        self.accumulated_ratio = 1.0
        self.accumulated_err = 0.0
        self.accumulated_steps = 0

    def _to_like(self, residual: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if residual is None:
            raise RuntimeError("MagCache residual tensor is missing.")
        if residual.device == ref.device and residual.dtype == ref.dtype:
            return residual
        return residual.to(device=ref.device, dtype=ref.dtype)

    def _maybe_move_to_cache_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.cache_on_cpu:
            return tensor.to("cpu")
        return tensor

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "num_steps": self.num_steps,
            "retention_steps": self.retention_steps,
            "mag_ratios": None if self.mag_ratios is None else self.mag_ratios.clone(),
            "residual_hidden": None if self.residual_hidden is None else self.residual_hidden.clone(),
            "residual_encoder": None if self.residual_encoder is None else self.residual_encoder.clone(),
            "stats": dict(self.stats),
            "step_index": self.step_index,
            "accumulated_ratio": self.accumulated_ratio,
            "accumulated_err": self.accumulated_err,
            "accumulated_steps": self.accumulated_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        if not isinstance(state_dict, dict):
            return
        self.num_steps = int(state_dict.get("num_steps", 0))
        self.retention_steps = int(state_dict.get("retention_steps", 0))
        mag_ratios = state_dict.get("mag_ratios")
        if mag_ratios is not None:
            self.mag_ratios = mag_ratios.clone()
        self.residual_hidden = state_dict.get("residual_hidden")
        self.residual_encoder = state_dict.get("residual_encoder")
        self.stats = dict(state_dict.get("stats", {"skips": 0, "full_pass": 0}))
        self.step_index = int(state_dict.get("step_index", 0))
        self.accumulated_ratio = float(state_dict.get("accumulated_ratio", 1.0))
        self.accumulated_err = float(state_dict.get("accumulated_err", 0.0))
        self.accumulated_steps = int(state_dict.get("accumulated_steps", 0))

    @property
    def last_error(self) -> float:
        return self._last_error


__all__ = [
    "MagCacheConfig",
    "MagCacheVideo",
    "get_default_mag_ratios",
    "DEFAULT_MAG_RATIOS",
]
