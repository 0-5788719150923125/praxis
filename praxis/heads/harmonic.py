"""Harmonic head: 2D irrational-rotation field, multiplicatively coupled.

The bias is a 2D standing wave over (position, feature) built from an
``F_t * F_d`` complex amplitude grid, IRFFT2'd to a ``[T_max, D]`` real field.
Phases are seeded by Weyl's theorem on the 2-torus: the cell ``(f_t, f_d)``
gets phase ``2*pi * frac(f_t * pi + f_d * e)``, equidistributed because
``{1, pi, e}`` are linearly independent over Q. Radial ``1/f^alpha`` decay
gives a 2D pink-noise prior on the spectrum.

The field is applied multiplicatively: ``h * (1 + b)``. Multiplicative coupling
forces the head into the gradient path - the upstream cannot cancel by emitting
``h(x) - b`` because the bias scales features rather than adding to them.
The lm_head is learnable - a frozen kernel projection cannot align with
content-dependent multiplicative shifts.

See ``proofs/harmonic_pi.md``.
"""

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead

IRR_T: float = math.pi
IRR_D: float = math.e
ALPHA: float = 1.0  # radial 1/f^alpha pink-noise decay
AMPLITUDE_INIT_STD: float = 1.0


def _spectrum_2d(
    F_t: int, F_d: int, irr_t: float, irr_d: float, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex unit-magnitude spectrum [F_t, F_d] with radial 1/f decay."""
    f_t = np.arange(1, F_t + 1, dtype=np.float64).reshape(-1, 1)
    f_d = np.arange(1, F_d + 1, dtype=np.float64).reshape(1, -1)
    raw = f_t * irr_t + f_d * irr_d
    phase = 2.0 * math.pi * (raw - np.floor(raw))
    decay = 1.0 / np.sqrt(f_t ** 2 + f_d ** 2) ** alpha
    real = np.cos(phase) * decay
    imag = np.sin(phase) * decay
    return (
        torch.from_numpy(real).to(torch.float32),
        torch.from_numpy(imag).to(torch.float32),
    )


class HarmonicField(nn.Module):
    """2D irrational-rotation field, applied multiplicatively to hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        max_positions: int,
        F_t: Optional[int] = None,
        F_d: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.T = max_positions
        self.D = hidden_dim
        self.F_t = F_t or min(hidden_dim, max_positions // 2)
        self.F_d = F_d or max(2, hidden_dim // 2)

        spec_real, spec_imag = _spectrum_2d(
            self.F_t, self.F_d, IRR_T, IRR_D, ALPHA
        )
        self.register_buffer("spec_real", spec_real, persistent=False)
        self.register_buffer("spec_imag", spec_imag, persistent=False)

        self.amplitudes = nn.Parameter(torch.empty(self.F_t, self.F_d))
        nn.init.normal_(self.amplitudes, mean=0.0, std=AMPLITUDE_INIT_STD)

    def _field(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> Tensor:
        rfft_D = self.D // 2 + 1
        spec = torch.zeros(
            self.T, rfft_D, dtype=torch.complex64, device=device
        )
        scaled = (
            torch.complex(self.spec_real.to(device), self.spec_imag.to(device))
            * self.amplitudes
        )
        spec[1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
        # Hermitian symmetry on the T axis so irfft2 yields a real field.
        spec[self.T - self.F_t : self.T, 1 : self.F_d + 1] = (
            scaled.flip(0).conj()
        )
        # ortho norm preserves spectral energy: field std stays ~ amp std
        # times sqrt(F_t * F_d / (T * D)), independent of T scale.
        field = torch.fft.irfft2(spec, s=(self.T, self.D), norm="ortho")
        idx = (torch.arange(seq_len, device=device) % self.T).long()
        b = field[idx]
        return b.to(dtype) if dtype is not None else b

    def forward(self, hidden_states: Tensor) -> Tensor:
        b = self._field(
            hidden_states.shape[-2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return hidden_states * (1.0 + b)


class HarmonicHead(BaseHead):
    """Learnable lm_head with a 2D harmonic field modulating features."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)

        max_positions = int(
            getattr(config, "max_position_embeddings", 32768) or 32768
        )
        if getattr(config, "encoder_type", None) and "byte" in str(
            config.encoder_type
        ):
            max_positions = max(max_positions, max_positions * 8)

        self.field = HarmonicField(
            hidden_dim=self.hidden_size, max_positions=max_positions
        )

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(self.field(hidden_states))

    @property
    def classifier(self) -> nn.Module:
        return self.lm_head
