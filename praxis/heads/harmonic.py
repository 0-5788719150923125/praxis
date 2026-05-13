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
# Forward-shift smoothness aux loss weight. The closed-form prior for
# "b_t predicts b_{t+1}" in our 2D Fourier basis with frozen Weyl phases
# reduces to a quadratic penalty on temporal frequency, normalized by
# amplitude norm. Replaces the prior Hoyer-sparsity loss, which rewarded
# sparsity but said nothing about which cells should win. Set to 0 to
# disable. See next/harmony.md for the derivation.
SMOOTHNESS_LAMBDA: float = 0.01
# Coupling depth: forward is h * (1 + COUPLING_DEPTH * tanh(b)), so the
# field range is (1 - COUPLING_DEPTH, 1 + COUPLING_DEPTH). Bounded away
# from 0 so saturating cells cannot zero out features.
COUPLING_DEPTH: float = 0.5
# Input-conditional amplitudes (deferred-item-2 promoted): a small MLP
# predicts a per-sequence delta on top of the static amplitude grid.
# Static-field baseline stays load-bearing; the delta is the path by which
# the field adapts to actual corpus content.
# Recency-biased multi-scale pooling fractions: full sequence, last half,
# last quarter, last eighth. Takens-flavored view of the conditioning input.
DILATIONS: list = [1, 2, 4, 8]
# Width of the amplitude-MLP hidden layer, as a multiplier of hidden_dim.
MLP_WIDTH_MULT: int = 1
# L2 regularization on the per-input delta. Keeps the static baseline
# load-bearing and prevents the MLP from becoming a content side-channel
# in its own right. Set to 0 to disable.
DELTA_LAMBDA: float = 0.001


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

    metric_descriptions = {
        "harmonic_amplitudes_norm": (
            "L2 norm of the 2D amplitude grid. Stable near init = no "
            "structure being learned; growing or rearranging = the field is "
            "shaping itself."
        ),
        "harmonic_grad_ratio": (
            "||grad(amplitudes)|| / ||grad(lm_head)||. Vanishing means the "
            "model is routing learning past the field rather than through it."
        ),
        "harmonic_spectrum": (
            "Live snapshot of |amp[f_t, f_d]|. Concentration in specific "
            "bands means corpus rhythms are being learned; uniform mass means "
            "the field is still noise."
        ),
        "harmonic_concentration": (
            "Hoyer sparsity of the amplitude grid in [0, 1]. 1 = all energy "
            "in a single (f_t, f_d) cell, 0 = perfectly uniform. Diagnostic "
            "only - no longer the loss target. Reading the rise here is "
            "evidence the field is committing to specific harmonics."
        ),
        "harmonic_smoothness": (
            "Forward-shift smoothness in [0, 1]. Closed-form expected "
            "(b_t - b_{t+1})^2 for the field, normalized by amplitude norm. "
            "Low = field varies predictably across positions; high = field "
            "is dominated by fast temporal modes. Pushed downward by the "
            "smoothness aux loss."
        ),
        "harmonic_delta_norm": (
            "RMS of the per-input amplitude delta, relative to the static "
            "baseline RMS. 0 = field is purely static; rising = the MLP is "
            "learning to adapt the field per input. Stays small under the "
            "delta L2 regularizer."
        ),
    }

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

        # Input-conditional delta MLP: pools the input at multiple scales,
        # produces a per-sequence amplitude delta. Output layer zeroed so the
        # field starts as the trained static baseline and anneals into
        # input dependence as the MLP learns.
        mlp_in = hidden_dim * len(DILATIONS)
        mlp_hidden = hidden_dim * MLP_WIDTH_MULT
        self.amplitude_mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.F_t * self.F_d),
        )
        nn.init.zeros_(self.amplitude_mlp[-1].weight)
        nn.init.zeros_(self.amplitude_mlp[-1].bias)

        # Stash the most recent delta (attached to graph) so aux_loss() can
        # regularize it after the forward pass completes.
        self._last_delta: Optional[Tensor] = None

    def _pool_with_delays(self, hidden_states: Tensor) -> Tensor:
        """[B, T, D] -> [B, D * len(DILATIONS)] via recency-biased multi-scale pooling.

        Each fraction f in DILATIONS yields the mean over the last T//f tokens.
        Concatenating gives the MLP four views of the sequence at increasing
        temporal resolution - the Takens-flavored "delayed observations at
        multiple scales" adapted to per-sequence pooling.
        """
        T = hidden_states.shape[-2]
        pooled = []
        for frac in DILATIONS:
            n = max(1, T // frac)
            pooled.append(hidden_states[..., -n:, :].mean(dim=-2))
        return torch.cat(pooled, dim=-1)

    def _field(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
        delta: Optional[Tensor] = None,
    ) -> Tensor:
        rfft_D = self.D // 2 + 1
        complex_spec = torch.complex(
            self.spec_real.to(device), self.spec_imag.to(device)
        )

        if delta is None:
            # Static path: single field for the whole batch.
            spec = torch.zeros(
                self.T, rfft_D, dtype=torch.complex64, device=device
            )
            scaled = complex_spec * self.amplitudes
            spec[1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
            spec[self.T - self.F_t : self.T, 1 : self.F_d + 1] = (
                scaled.flip(0).conj()
            )
            field = torch.fft.irfft2(spec, s=(self.T, self.D), norm="ortho")
            idx = (torch.arange(seq_len, device=device) % self.T).long()
            b = field[idx]
        else:
            # Input-conditional path: one field per batch element.
            B = delta.shape[0]
            amps = self.amplitudes.unsqueeze(0) + delta  # [B, F_t, F_d]
            spec = torch.zeros(
                B, self.T, rfft_D, dtype=torch.complex64, device=device
            )
            scaled = complex_spec.unsqueeze(0) * amps  # [B, F_t, F_d]
            spec[:, 1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
            spec[:, self.T - self.F_t : self.T, 1 : self.F_d + 1] = (
                scaled.flip(1).conj()
            )
            field = torch.fft.irfft2(
                spec, s=(self.T, self.D), norm="ortho"
            )  # [B, T, D]
            idx = (torch.arange(seq_len, device=device) % self.T).long()
            b = field[:, idx]

        return b.to(dtype) if dtype is not None else b

    def forward(self, hidden_states: Tensor) -> Tensor:
        ctx = self._pool_with_delays(hidden_states)
        delta = self.amplitude_mlp(ctx).view(-1, self.F_t, self.F_d)
        self._last_delta = delta
        b = self._field(
            hidden_states.shape[-2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            delta=delta,
        )
        # Bounded away from 0 so a saturating cell cannot zero features.
        return hidden_states * (1.0 + COUPLING_DEPTH * torch.tanh(b))

    def concentration(self) -> Tensor:
        """Hoyer sparsity of the amplitude grid in [0, 1].

        H = (sqrt(N) - ||a||_1 / ||a||_2) / (sqrt(N) - 1).
        1 = all energy in a single cell, 0 = perfectly uniform. Scale-invariant.
        Diagnostic only - no longer the aux-loss target.
        """
        a = self.amplitudes.abs()
        N = a.numel()
        sqrt_N = math.sqrt(N)
        l1 = a.sum()
        l2 = torch.sqrt((a * a).sum() + 1e-12)
        return (sqrt_N - l1 / l2) / (sqrt_N - 1)

    def smoothness(self) -> Tensor:
        """Forward-shift smoothness in [0, 1].

        Expected (b_t - b_{t+1})^2 for our 2D Fourier basis with frozen Weyl
        phases reduces, in the F_t << T regime, to a quadratic penalty on
        temporal frequency. Normalized by amplitude norm so the value is
        scale-invariant: it asks where the amplitude variance lives, not how
        much of it there is. Low = predictable across positions.
        """
        a2 = self.amplitudes.pow(2)
        f_t = torch.arange(
            1, self.F_t + 1, device=a2.device, dtype=a2.dtype
        ).view(-1, 1)
        w = (f_t / self.F_t).pow(2)
        return (a2 * w).sum() / (a2.sum() + 1e-12)

    def delta_norm(self) -> Tensor:
        """RMS of the most recent input-conditional delta, relative to baseline.

        Returns ``rms(delta) / rms(self.amplitudes)``. 0 = field is purely
        static; ~1 = delta is the same scale as the baseline. The L2
        regularizer in aux_loss should keep this small.
        """
        baseline_rms = (
            self.amplitudes.detach().pow(2).mean().sqrt().clamp_min(1e-12)
        )
        if self._last_delta is None:
            return torch.zeros((), device=baseline_rms.device)
        delta_rms = self._last_delta.detach().pow(2).mean().sqrt()
        return delta_rms / baseline_rms

    def aux_loss(self) -> Optional[Tensor]:
        """Combined smoothness + delta-L2 aux loss.

        Smoothness operates on the static baseline so the spectral prior
        keeps shaping the long-run shape of the field. Delta L2 keeps the
        per-input perturbation small so the static baseline stays
        load-bearing rather than the MLP becoming a content side-channel.
        """
        parts = []
        if SMOOTHNESS_LAMBDA > 0.0:
            parts.append(SMOOTHNESS_LAMBDA * self.smoothness())
        if DELTA_LAMBDA > 0.0 and self._last_delta is not None:
            parts.append(DELTA_LAMBDA * self._last_delta.pow(2).mean())
        if not parts:
            return None
        return sum(parts)


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
