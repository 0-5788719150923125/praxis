import math
from typing import Any, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.normalization import NORMALIZATION_REGISTRY
from praxis.orchestration import EXPERT_REGISTRY
from praxis.residuals import RESIDUAL_REGISTRY
from praxis.utils import norm_scaling

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _hadamard(n: int) -> Tensor:
    """Normalized Sylvester-Hadamard matrix; n must be a power of 2."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class HadamardMixer(nn.Module):
    """Orthogonal, self-inverse Walsh-Hadamard transform over the channel dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("h", _hadamard(dim), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.h)


class LiftingWavelet(nn.Module):
    """Causal learned lifting wavelet. predict/update are shared between
    analysis and synthesis, so reconstruction is exact by construction.
    Channel-wise (time-shared) maps; causality comes from the left-only shift."""

    def __init__(self, dim: int, max_levels: int) -> None:
        super().__init__()
        self.predict = nn.ModuleList(nn.Linear(dim, dim) for _ in range(max_levels))
        self.update = nn.ModuleList(nn.Linear(dim, dim) for _ in range(max_levels))
        # zero-init -> starts as a plain Haar-style average/difference
        for lin in (*self.predict, *self.update):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def decompose(self, x: Tensor, levels: int) -> Tuple[Tensor, List[Tensor]]:
        details: List[Tensor] = []
        cur = x
        for level in range(levels):
            dilation = 1 << level
            odd = F.pad(cur, (0, 0, dilation, 0))[:, :-dilation, :]  # past only
            detail = (odd - self.predict[level](cur)) * INV_SQRT2
            cur = (cur + self.update[level](detail)) * INV_SQRT2
            details.append(detail)
        return cur, details

    def reconstruct(self, approx: Tensor, details: List[Tensor]) -> Tensor:
        cur = approx
        for level in range(len(details) - 1, -1, -1):
            cur = cur * SQRT2 - self.update[level](details[level])
        return cur


class GatedSpectralMixer(nn.Module):
    """Per-scale SwiGLU mixer applied in Walsh-Hadamard space."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.signal = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.signal(x) * F.silu(self.gate(x))


class WaveletBlock(nn.Module):
    """Attention-free block: a causal lifting-wavelet mixer with per-scale gated
    spectral mixing, then a standard FFN. Ports the core ideas of WaveletLM
    (github.com/ramongougis/WaveletLM) onto Praxis primitives.

    Stateless: it returns no cache/state and recomputes over the full context.
    """

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cp = _next_pow2(self.hidden_size)  # FWHT needs a power-of-2 width
        self.use_scaler = config.scaled

        context = getattr(config, "block_size", None) or getattr(
            config, "max_position_embeddings", 1024
        )
        # fixed, model-agnostic depth (deeper levels reach 2^level into the past)
        self.max_levels = int(
            getattr(config, "wavelet_levels", max(1, min(int(math.log2(context)), 8)))
        )

        # --- wavelet mixer sublayer ---
        self.mix_res = RESIDUAL_REGISTRY.get(config.residual_type)(
            self.hidden_size, num_depths=config.depth
        )
        self.mix_norm = NORMALIZATION_REGISTRY[config.norm_type](
            self.hidden_size, eps=config.epsilon
        )
        self.lifting = LiftingWavelet(self.cp, self.max_levels)
        self.hadamard = HadamardMixer(self.cp)
        self.scale_mixers = nn.ModuleList(
            GatedSpectralMixer(self.cp) for _ in range(self.max_levels + 1)
        )
        self.scale_weights = nn.Parameter(torch.ones(self.max_levels + 1))
        self.proj_out = nn.Linear(self.cp, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # --- FFN sublayer (mirrors TransformerBlock) ---
        self.ffn_res = RESIDUAL_REGISTRY.get(config.residual_type)(
            self.hidden_size, num_depths=config.depth
        )
        self.ffn_norm = NORMALIZATION_REGISTRY[config.norm_type](
            self.hidden_size, eps=config.epsilon
        )
        self.ffn = EXPERT_REGISTRY[config.expert](config)

    def _mix(self, h: Tensor) -> Tensor:
        if self.cp != self.hidden_size:
            h = F.pad(h, (0, self.cp - self.hidden_size))

        levels = self.max_levels  # fixed depth; deep levels see zero-padded history
        approx, details = self.lifting.decompose(h, levels)
        coeffs = torch.stack([approx, *details], dim=2)  # [B, T, S, Cp]

        spec = self.hadamard(coeffs)
        spec = torch.stack(
            [
                spec[:, :, s, :] + self.scale_mixers[s](spec[:, :, s, :])
                for s in range(levels + 1)
            ],
            dim=2,
        )
        mixed = self.hadamard(spec)
        mixed = mixed * self.scale_weights.view(1, 1, levels + 1, 1)

        recon = self.lifting.reconstruct(
            mixed[:, :, 0, :], [mixed[:, :, s, :] for s in range(1, levels + 1)]
        )
        out = self.proj_out(recon)
        return self.dropout(out[..., : self.hidden_size])

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        current_state: Optional[Any] = None,
        current_depth: int = 0,
        block_ids: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None, float]:
        # =========== Wavelet Mixer ===========
        residual, beta = self.mix_res.connect_width(inputs)
        h = self.mix_norm(self.mix_res.format_state(residual), mode="pre")
        if self.use_scaler:
            h = norm_scaling(h, current_depth)
        h = self._mix(h)
        h = self.mix_norm(h, mode="post")
        merged = self.mix_res.connect_depth(residual, h, beta, current_depth=current_depth)

        # =========== FeedForward =============
        residual, beta_ffn = self.ffn_res.connect_width(
            self.ffn_res.format_state(merged)
        )
        f = self.ffn_norm(self.ffn_res.format_state(residual), mode="pre")
        if self.use_scaler:
            f = norm_scaling(f, current_depth)
        f = self.ffn(f, current_depth)
        f = self.ffn_norm(f, mode="post")
        out = self.ffn_res.connect_depth(
            residual, f, beta_ffn, current_depth=current_depth
        )

        return self.ffn_res.format_state(out), None, None, 0.0
