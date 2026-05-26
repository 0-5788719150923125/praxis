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
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead

IRR_T: float = math.pi
IRR_D: float = math.e
ALPHA: float = 1.0  # radial 1/f^alpha pink-noise decay
AMPLITUDE_INIT_STD: float = 1.0
# Forward-shift smoothness aux loss weight. The closed-form prior for
# "b_t predicts b_{t+1}" in our 2D Fourier basis with frozen Weyl phases
# reduces to a quadratic penalty on temporal frequency, normalized by
# amplitude norm. Set to 0 to disable.
# See next/harmony.md for the derivation.
SMOOTHNESS_LAMBDA: float = 0.01


def _spectrum_2d(
    F_t: int, F_d: int, irr_t: float, irr_d: float, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex unit-magnitude spectrum [F_t, F_d] with radial 1/f decay."""
    f_t = np.arange(1, F_t + 1, dtype=np.float64).reshape(-1, 1)
    f_d = np.arange(1, F_d + 1, dtype=np.float64).reshape(1, -1)
    raw = f_t * irr_t + f_d * irr_d
    phase = 2.0 * math.pi * (raw - np.floor(raw))
    decay = 1.0 / np.sqrt(f_t**2 + f_d**2) ** alpha
    real = np.cos(phase) * decay
    imag = np.sin(phase) * decay
    return (
        torch.from_numpy(real).to(torch.float32),
        torch.from_numpy(imag).to(torch.float32),
    )


class HarmonicField(nn.Module):
    """2D irrational-rotation field, applied multiplicatively to hidden states."""

    metric_descriptions = {
        "harmonic_amplitudes_norm": {
            "description": (
                "L2 norm of the 2D amplitude grid. Stable near init = no "
                "structure being learned; growing or rearranging = the field "
                "is shaping itself."
            ),
            "chart": {
                "title": "Harmonic Field Amplitudes",
                "y_label": "Amplitudes ||L2||",
                "y_scale": "logarithmic",
                "group": "harmonic_head",
                "order": 10,
            },
        },
        "harmonic_grad_ratio": {
            "description": (
                "||grad(amplitudes)|| / ||grad(lm_head)||. Vanishing means "
                "the model is routing learning past the field rather than "
                "through it."
            ),
            "chart": {
                "title": "Harmonic Gradient Ratio",
                "y_label": "Grad Ratio (Log Scale)",
                "y_scale": "logarithmic",
                "group": "harmonic_head",
                "order": 20,
            },
        },
        "harmonic_concentration": {
            "description": (
                "Hoyer sparsity of the amplitude grid in [0, 1]. 1 = all "
                "energy in a single (f_t, f_d) cell, 0 = perfectly uniform. "
                "Diagnostic only - no longer the loss target. Reading the "
                "rise here is evidence the field is committing to specific "
                "harmonics."
            ),
            "chart": {
                "title": "Spectral Concentration",
                "y_label": "Hoyer Sparsity",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 30,
            },
        },
        "harmonic_smoothness": {
            "description": (
                "Forward-shift smoothness in [0, 1]. Closed-form expected "
                "(b_t - b_{t+1})^2 for the field, normalized by amplitude "
                "norm. Low = field varies predictably across positions; "
                "high = field is dominated by fast temporal modes. Pushed "
                "downward by the smoothness aux loss."
            ),
            "chart": {
                "title": "Forward-Shift Smoothness",
                "y_label": "Forward-Shift Smoothness",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 40,
            },
        },
        # Spectrum is a bespoke heatmap snapshot, not a scalar chart -
        # the snapshot hint routes it through the heatmap_2d renderer.
        "harmonic_spectrum": {
            "description": (
                "Live snapshot of |amp[f_t, f_d]|. Concentration in specific "
                "bands means corpus rhythms are being learned; uniform mass "
                "means the field is still noise."
            ),
            "snapshot": {
                "title": "Harmonic Spectrum",
                "renderer": "heatmap_2d",
                "color_scale": "linear",
                "group": "harmonic_head",
                "order": 100,
            },
        },
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

        spec_real, spec_imag = _spectrum_2d(self.F_t, self.F_d, IRR_T, IRR_D, ALPHA)
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
        spec = torch.zeros(self.T, rfft_D, dtype=torch.complex64, device=device)
        scaled = (
            torch.complex(self.spec_real.to(device), self.spec_imag.to(device))
            * self.amplitudes
        )
        spec[1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
        # Hermitian symmetry on the T axis so irfft2 yields a real field.
        spec[self.T - self.F_t : self.T, 1 : self.F_d + 1] = scaled.flip(0).conj()
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
        f_t = torch.arange(1, self.F_t + 1, device=a2.device, dtype=a2.dtype).view(
            -1, 1
        )
        w = (f_t / self.F_t).pow(2)
        return (a2 * w).sum() / (a2.sum() + 1e-12)

    def aux_loss(self) -> Optional[Tensor]:
        """Forward-shift smoothness loss: lambda * smoothness.

        CCA-flavored prior: ask the field at position t to be predictable
        from the field at t+1. For our basis, this reduces to "low temporal
        frequency mass." Replaces the prior Hoyer loss, which knew nothing
        about which cells should win.
        """
        if SMOOTHNESS_LAMBDA <= 0.0:
            return None
        return SMOOTHNESS_LAMBDA * self.smoothness()


class HarmonicHead(BaseHead):
    """Learnable lm_head with a 2D harmonic field modulating features.

    In encoder-attached mode the head owns a field sized to the
    encoder's classifier feature dim and modulates ``decoder_embeds``
    before re-projecting through the encoder's classifier weight.
    The encoder still owns the classifier itself.
    """

    def __init__(self, config: Any, encoder: Optional[nn.Module] = None) -> None:
        super().__init__(config, encoder)
        max_positions = int(getattr(config, "max_position_embeddings", 32768) or 32768)

        if self.has_encoder:
            classifier = getattr(encoder, "classifier", None)
            if classifier is not None and hasattr(classifier, "weight"):
                feature_dim = classifier.weight.shape[1]
            else:
                feature_dim = self.hidden_size
            if "byte" in str(getattr(config, "encoder_type", "")):
                max_positions = max(max_positions, max_positions * 8)
            self.field = HarmonicField(
                hidden_dim=feature_dim, max_positions=max_positions
            )
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
            self.field = HarmonicField(
                hidden_dim=self.hidden_size, max_positions=max_positions
            )

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        return self.lm_head(self.field(hidden_states))

    @property
    def classifier(self) -> Optional[nn.Module]:
        return self.lm_head

    def process_encoder_output(
        self,
        decoder_embeds: Tensor,
        encoder_logits: Tensor,
        encoder_classifier: nn.Module,
    ) -> Tuple[Tensor, Tensor, nn.Module]:
        # Modulate decoder_embeds *before* binding them as hidden_states
        # so that cut_cross_entropy (which projects embeddings @ classifier
        # internally and discards the materialized logits) sees the field.
        decoder_embeds = self.field(decoder_embeds)
        logits = F.linear(
            decoder_embeds,
            encoder_classifier.weight,
            getattr(encoder_classifier, "bias", None),
        ).to(encoder_logits.dtype)
        return logits, decoder_embeds, encoder_classifier

    def aux_losses(self) -> dict:
        aux = self.field.aux_loss()
        return {"harmonic_smoothness": aux} if aux is not None else {}

    def _downstream_classifier(self) -> Optional[nn.Module]:
        """The learnable projection the field's output feeds into.

        Encoder mode: the encoder's own classifier (the field modulates
        decoder_embeds, which then get re-projected through it).
        Standalone: our own lm_head.
        """
        if self._encoder is not None:
            return getattr(self._encoder, "classifier", None)
        return self.lm_head

    def dashboard_snapshots(self) -> dict:
        """Amplitude grid magnitudes for the spectrum heatmap.

        Returns the field's ``|amp[f_t, f_d]|`` matrix and the
        irrationals used to seed phases, packaged for the generic
        ``heatmap_2d`` renderer (grid + axis ranges + max).
        """
        amps = self.field.amplitudes.detach().abs().to("cpu", dtype=torch.float32)
        F_t, F_d = int(amps.shape[0]), int(amps.shape[1])
        return {
            "harmonic_spectrum": {
                "grid": amps.tolist(),
                "grid_rows": F_t,
                "grid_cols": F_d,
                "x_range": [1, F_d],
                "y_range": [1, F_t],
                "max_count": float(amps.max().item()) if amps.numel() else 0.0,
                "irrationals": {"t": float(IRR_T), "d": float(IRR_D)},
            }
        }

    def training_metrics(self) -> dict:
        amps = self.field.amplitudes
        out = {
            "harmonic_amplitudes_norm": float(amps.detach().norm().item()),
            "harmonic_concentration": float(self.field.concentration().item()),
            "harmonic_smoothness": float(self.field.smoothness().item()),
        }

        # grad_ratio reads whether learning is flowing into the field or
        # past it through the downstream classifier. Skip silently if
        # gradients aren't available yet (pre-first-step) or if the
        # downstream classifier doesn't expose a .weight.
        amps_grad = amps.grad
        classifier = self._downstream_classifier()
        head_weight = getattr(classifier, "weight", None) if classifier else None
        head_grad = head_weight.grad if head_weight is not None else None
        if amps_grad is not None and head_grad is not None:
            head_norm = float(head_grad.detach().norm().item())
            if head_norm > 0:
                out["harmonic_grad_ratio"] = (
                    float(amps_grad.detach().norm().item()) / head_norm
                )
        return out
