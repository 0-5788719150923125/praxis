"""Harmonic head: 2D irrational-rotation field, multiplicatively coupled.

The bias is a 2D standing wave over (position, feature) built from an
``F_t * F_d`` complex amplitude grid, evaluated separably at the positions in
use (equivalent to IRFFT2 over the full ``[T_max, D]`` period, never built).
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
# amplitude norm. Set to 0 to disable.
# See next/harmony.md for the derivation.
SMOOTHNESS_LAMBDA: float = 0.01

# Amplitude modulation: an envelope over the temporal-frequency (f_t) axis,
# applied as ``amp[f_t, :] *= env[f_t]``. "static" seeds a single mid-band
# oscillation; "learned" makes the envelope a handful of learnable cosine
# coefficients so the wave adapts. The envelope's basis is zero at f_t -> 0,
# so it cannot reintroduce the flat (constant-over-position) mode that the bare
# grid settles into. See ``HarmonicField`` and ``next/harmony.md``.
AMP_MOD_DEPTH: float = 0.5  # peak envelope modulation, tanh-bounded
AMP_MOD_BASIS_K: int = 6  # learned envelope = K low-frequency f_t modes


def _envelope_basis(F_t: int, K: int) -> torch.Tensor:
    """``[F_t, K]`` sine modes over the f_t axis.

    Mode ``k`` is ``sin(pi*k*f_t/F_t)`` - a smooth wave that vanishes at
    ``f_t -> 0``, so any combination of modes leaves the flat DC component
    untouched. Mode 1 is a single hump peaking mid-band.
    """
    ft = np.arange(1, F_t + 1, dtype=np.float64).reshape(-1, 1)
    k = np.arange(1, K + 1, dtype=np.float64).reshape(1, -1)
    return torch.from_numpy(np.sin(np.pi * ft * k / F_t)).to(torch.float32)


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
        "harmonic_env_depth": {
            "description": (
                "Peak-to-trough of the f_t amplitude envelope. 0 = no "
                "modulation (the flat grid); >0 = a wave over temporal "
                "frequency. With learned modulation this moves as the "
                "envelope adapts; static holds it fixed."
            ),
            "chart": {
                "title": "Amplitude Envelope Depth",
                "y_label": "Envelope Depth",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 45,
            },
        },
        # Capacity allocation: three shares (sum to 1) on one chart, showing
        # how the field's energy budget is split between static bias, learned
        # input-conditional variance, and unwritten headroom.
        "harmonic_capacity_bias": {
            "description": (
                "Share of field energy doing bias work - the static, "
                "population-average spectrum every input sees."
            ),
            "chart": {
                "title": "Capacity Allocation",
                "y_label": "Share of field energy",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 46,
                "series_group": "harmonic_capacity",
                "series_label": "bias (static)",
            },
        },
        "harmonic_capacity_variance": {
            "description": (
                "Share of field energy doing variance work - the "
                "input-conditional delta the envelope writes per sequence. "
                "Zero until an input-modulated field has trained."
            ),
            "chart": {
                "title": "Capacity Allocation",
                "y_label": "Share of field energy",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 47,
                "series_group": "harmonic_capacity",
                "series_label": "variance (input-conditional)",
            },
        },
        "harmonic_capacity_dormant": {
            "description": (
                "Share of spectral capacity sitting dormant - headroom "
                "relative to a saturated spectrum. This is the room left: a "
                "concentrated field leaves most features unwritten."
            ),
            "chart": {
                "title": "Capacity Allocation",
                "y_label": "Share of field energy",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 48,
                "series_group": "harmonic_capacity",
                "series_label": "dormant (headroom)",
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
        # The field's PCA cross-section unrolled along the position axis into a
        # rising spiral ribbon - the real signal behind the old fake
        # "correlation" animation. Deterministic given the frozen Weyl phases,
        # so its shape fingerprints the model.
        "harmonic_spiral": {
            "description": (
                "Top-2 PCA cross-section of the harmonic field unrolled along "
                "the sequence axis into a 3D spiral. Ribbon width is the field "
                "energy left outside the plane (what the flat view hides): a "
                "tight spiral = low effective dimension (consensus), a wide "
                "fuzzy ribbon = high dimension (interference)."
            ),
            "snapshot": {
                "title": "Harmonic Spiral",
                "renderer": "harmonic_spiral",
                "group": "harmonic_head",
                "order": 101,
            },
        },
        # Same PCA projection as the spiral, but kept as a closed planar loop
        # and drawn as a Fourier epicycle. A second lens on the same field.
        "harmonic_curve": {
            "description": (
                "Top-2 PCA trajectory of the harmonic field across one period, "
                "drawn as a Fourier epicycle: nested rotating vectors whose tip "
                "traces the loop. The arms are generic Fourier scaffolding - the "
                "real signal is the loop shape and how energy spreads across the "
                "harmonics. A tight loop = low effective dimension, a "
                "space-filling rosette = interference."
            ),
            "snapshot": {
                "title": "Harmonic Epicycle",
                "renderer": "harmonic_curve",
                "group": "harmonic_head",
                "order": 102,
            },
        },
        # Time domain: the raw field as per-feature traces over the period.
        # Complements the spectrum (frequency) and spiral/epicycle (PCA shape).
        "harmonic_traces": {
            "description": (
                "Raw field b(t, d) over one period: each line is one feature's "
                "value flowing timestep to timestep, sampled across evenly "
                "spaced features. The overlay is the harmonics interfering - a "
                "moiré of phase-shifted waves; the playhead reads the whole "
                "feature column at one position."
            ),
            "snapshot": {
                "title": "Harmonic Field Traces",
                "renderer": "field_traces",
                "group": "harmonic_head",
                "order": 103,
            },
        },
        # The real "correlation": cosine similarity between feature trajectories.
        # Amplitude-invariant, so it reads pure co-evolution - the honest
        # successor to the fake terminal "CORRELATION" panel.
        "harmonic_correlation": {
            "description": (
                "Cosine similarity between feature trajectories over one period "
                "(amplitude removed). Red = rise/fall together, blue = "
                "anti-correlated, white = unrelated. Blocks of warm cells are "
                "feature groups locked into the same harmonic rhythm."
            ),
            "snapshot": {
                "title": "Feature Correlation",
                "renderer": "corr_matrix",
                "group": "harmonic_head",
                "order": 104,
            },
        },
        # Frequency-domain sibling of the spiral: the harmonic ladder as a tower
        # of blocks, ranked by energy and placed by frozen Weyl phase.
        "harmonic_staircase": {
            "description": (
                "Each block is one harmonic: stacked by energy (biggest at the "
                "base, tapering into the sky), angled around the column by its "
                "frozen Weyl phase, sized by amplitude. Where the spiral walks "
                "position, this walks frequency - a tall narrow climb means a "
                "few harmonics dominate; a broad scattered one means many do."
            ),
            "snapshot": {
                "title": "Harmonic Staircase",
                "renderer": "harmonic_staircase",
                "group": "harmonic_head",
                "order": 105,
            },
        },
        # The bias/variance strands. Each feature is a particle; one cylinder end
        # arranges them by phase (the static field, pure bias), the other by
        # (static energy, input-conditional energy) - the orthogonal axes made
        # literal. With amp_modulation != "input" the variance axis is ~0 and the
        # plane stays collapsed: the split appearing is the trained result.
        "harmonic_strands": {
            "description": (
                "Bias and variance as a morphing cylinder. Particles are "
                "features: one end is the static field's phase ring (pure bias, "
                "all structure), the other is the (bias energy, variance energy) "
                "plane where the input-conditional envelope pulls features off "
                "the bias axis. A collapsed plane means the field is still pure "
                "bias; a split means structured variance has been learned."
            ),
            "snapshot": {
                "title": "Bias/Variance Strands",
                "renderer": "harmonic_strands",
                "group": "harmonic_head",
                "order": 106,
            },
        },
        "harmonic_snake": {
            "description": (
                "The field over a single sequence as a radial snake: 12 samples "
                "along the sequence, angle = the field's PCA phase, radius = time "
                "sinking toward the origin (newest at center). On a clock-face of "
                "four quadrants. 'Circles the origin' is the winding number of the "
                "phase - it reaches a full loop only when the dominant mode "
                "actually turns once over the sequence, so the badge reports "
                "whether the data produced the circle, not whether we wanted it."
            ),
            "snapshot": {
                "title": "Sequence Snake (on the dial)",
                "renderer": "harmonic_snake",
                "group": "harmonic_head",
                "order": 107,
            },
        },
    }

    def __init__(
        self,
        hidden_dim: int,
        max_positions: int,
        F_t: Optional[int] = None,
        F_d: Optional[int] = None,
        amp_modulation: str = "off",
    ) -> None:
        super().__init__()
        self.T = max_positions
        self.D = hidden_dim
        self.F_t = F_t or min(hidden_dim, max_positions // 2)
        self.F_d = F_d or max(2, hidden_dim // 2)

        spec_real, spec_imag = _spectrum_2d(self.F_t, self.F_d, IRR_T, IRR_D, ALPHA)
        self.register_buffer("spec_real", spec_real, persistent=False)
        self.register_buffer("spec_imag", spec_imag, persistent=False)

        # Feature-axis cosine basis for the separable field evaluation: bin
        # f_d contributes w * cos(2*pi*f_d*d/D), w=2 from Hermitian doubling
        # (w=1 at Nyquist), matching irfft over the D axis exactly.
        f_d = torch.arange(1, self.F_d + 1, dtype=torch.float32)
        d = torch.arange(self.D, dtype=torch.float32)
        w = torch.full((self.F_d, 1), 2.0)
        if self.D % 2 == 0 and self.F_d == self.D // 2:
            w[-1] = 1.0
        self.register_buffer(
            "basis_d",
            w * torch.cos(2 * math.pi * f_d.unsqueeze(1) * d / self.D),
            persistent=False,
        )

        self.amplitudes = nn.Parameter(torch.empty(self.F_t, self.F_d))
        nn.init.normal_(self.amplitudes, mean=0.0, std=AMPLITUDE_INIT_STD)

        # Amplitude envelope over f_t. "static" and "learned" share one formula
        # (so they are identical at init); only "learned" lets the coefficients
        # move. Init = a single mid-band oscillation (coeff 0 = 1, rest 0).
        if amp_modulation not in ("off", "static", "learned", "input", "pure"):
            raise ValueError(
                f"amp_modulation must be off|static|learned|input|pure, got {amp_modulation!r}"
            )
        self.amp_modulation = amp_modulation
        if amp_modulation != "off":
            self.register_buffer(
                "amp_basis",
                _envelope_basis(self.F_t, AMP_MOD_BASIS_K),
                persistent=False,
            )
            coeffs = torch.zeros(AMP_MOD_BASIS_K)
            if amp_modulation != "pure":
                coeffs[0] = 1.0  # "pure" has no static base envelope
            if amp_modulation in ("learned", "input"):
                self.amp_coeffs = nn.Parameter(coeffs)
            else:
                self.register_buffer("amp_coeffs", coeffs, persistent=False)
            if amp_modulation == "pure":
                # Variance-only field: no static spectrum reaches the output.
                # The field is the conditional delta alone, so it is exactly
                # zero at init (zero-init projection) and ramps only under
                # optimizer pressure; the per-band gain lets that ramp be
                # band-selective.
                self.amp_gain = nn.Parameter(torch.ones(self.F_t))
            if amp_modulation in ("input", "pure"):
                # Input-conditional envelope - the field's structured-variance
                # axis. A zero-init projection from pooled hidden states to
                # envelope coefficients: the field is exactly the static (bias)
                # field at init and learns its input-dependence, orthogonal to
                # the static spectrum. ``_last_input_coeffs`` keeps a
                # representative coeff set (mean over the last batch) so the
                # strands snapshot can rebuild the conditional field with no batch.
                self.amp_input = nn.Linear(self.D, AMP_MOD_BASIS_K, bias=False)
                nn.init.zeros_(self.amp_input.weight)
                self.register_buffer(
                    "_last_input_coeffs", coeffs.clone(), persistent=False
                )

    def _eval_field(self, scaled: Tensor, seq_len: int, device: torch.device) -> Tensor:
        """Band-limited field at positions ``0..seq_len-1`` via two small
        matmuls - exactly the ortho-normed irfft2 of the (Hermitian-extended)
        ``[T, D]`` spectrum, but never materializing it: only F_t * F_d bins
        are nonzero, so the transform is separable. ``scaled`` is the complex
        amplitude grid ``[..., F_t, F_d]``; returns ``[..., seq_len, D]``.
        Memory is O(seq_len * D) instead of O(T * D) - load-bearing when T
        spans the full context and seq_len is one block.

        The 2/sqrt(T*D) factor folds the Hermitian doubling on the T axis
        into irfft2's ortho norm, preserving spectral energy: field std stays
        ~ amp std * sqrt(F_t * F_d / (T * D)), independent of T scale.
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        f_t = torch.arange(1, self.F_t + 1, device=device, dtype=torch.float32)
        ang = 2 * math.pi * t.unsqueeze(1) * f_t / self.T  # [L, F_t]
        a = torch.cos(ang) @ scaled.real - torch.sin(ang) @ scaled.imag
        return (2.0 / math.sqrt(self.T * self.D)) * (a @ self.basis_d.to(device))

    def _field(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> Tensor:
        amps = self.amplitudes
        env = self._envelope()
        if env is not None:
            amps = amps * env.unsqueeze(1)  # modulate each f_t row by env[f_t]
        scaled = (
            torch.complex(self.spec_real.to(device), self.spec_imag.to(device)) * amps
        )
        b = self._eval_field(scaled, seq_len, device)
        return b.to(dtype) if dtype is not None else b

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.amp_modulation in ("input", "pure"):
            b = self._field_conditional(hidden_states)
        else:
            b = self._field(
                hidden_states.shape[-2],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return hidden_states * (1.0 + b)

    def _field_conditional(self, hidden_states: Tensor) -> Tensor:
        """Input-conditional field ``[B, seq_len, D]``: the static spectrum with
        an envelope whose coefficients carry a per-sequence delta from pooled
        hidden states. Zero-init projection means it is identical to the static
        field at init; the learned delta is the structured-variance axis."""
        b_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        pooled = hidden_states.mean(dim=-2).to(self.amp_basis.dtype)  # [B, D]
        if self.amp_modulation == "pure":
            coeffs = self.amp_input(pooled)  # [B, K] - no static base
        else:
            coeffs = self.amp_coeffs + self.amp_input(pooled)  # [B, K]
        self._last_input_coeffs = coeffs.detach().mean(0)
        env = self._env_from_coeffs(coeffs)  # [B, F_t]
        amps = self.amplitudes.unsqueeze(0) * env.unsqueeze(-1)  # [B, F_t, F_d]
        return self._build_field(amps, seq_len, device).to(hidden_states.dtype)

    def _build_field(self, amps: Tensor, seq_len: int, device: torch.device) -> Tensor:
        """Batched field from per-example amplitudes ``[B, F_t, F_d]`` -> ``[B,
        seq_len, D]``. The batched twin of :meth:`_field`; the separable
        evaluation broadcasts over the batch dim."""
        phase = torch.complex(self.spec_real.to(device), self.spec_imag.to(device))
        scaled = phase.unsqueeze(0) * amps  # [B, F_t, F_d]
        return self._eval_field(scaled, seq_len, device)

    def _env_from_coeffs(self, coeffs: Tensor) -> Tensor:
        """Envelope over f_t from coefficient rows ``[..., K]`` -> ``[..., F_t]``.
        ``1 + depth*tanh(...)`` stays positive and bounded; "pure" drops the
        base 1 (the field is the conditional delta alone) and applies the
        per-band gain."""
        mod = AMP_MOD_DEPTH * torch.tanh(coeffs @ self.amp_basis.T)
        if self.amp_modulation == "pure":
            return mod * self.amp_gain
        return 1.0 + mod

    def _envelope(self) -> Optional[Tensor]:
        """``[F_t]`` amplitude envelope over the temporal-frequency axis, or
        None when modulation is off. For "pure" this is the last batch's
        conditional envelope (zero before any forward)."""
        if self.amp_modulation == "off":
            return None
        if self.amp_modulation == "pure":
            return self._env_from_coeffs(self._last_input_coeffs)
        return self._env_from_coeffs(self.amp_coeffs)

    def effective_amplitudes(self) -> Tensor:
        """The amplitude grid after the envelope - what the spectrum heatmap
        should show so the modulation is visible (equals ``amplitudes`` when
        modulation is off)."""
        amps = self.amplitudes.detach()
        env = self._envelope()
        if env is not None:
            amps = amps * env.detach().unsqueeze(1)
        return amps

    def envelope_depth(self) -> float:
        """Peak-to-trough of the f_t envelope; 0 when modulation is off."""
        env = self._envelope()
        return 0.0 if env is None else float((env.max() - env.min()).detach().item())

    def _sample_field(self, Tp: int, coeffs: Optional[Tensor] = None) -> Tensor:
        """Real field [Tp, D] sampled over one period, mean-centered over time.

        Alias-free for Tp >= 2*F_t+1 (the field is band-limited to F_t temporal
        frequencies), and far cheaper than the full-T irfft. Shared by every
        snapshot view below. ``coeffs`` overrides the envelope coefficients (the
        input-conditional set, for the strands snapshot); default = the static
        base envelope.
        """
        rfft_D = self.D // 2 + 1
        spec = torch.zeros(Tp, rfft_D, dtype=torch.complex64)
        amps = self.amplitudes.detach().cpu()
        if coeffs is not None:
            env = self._env_from_coeffs(coeffs.to(self.amp_basis.device))
            amps = amps * env.detach().cpu().unsqueeze(1)
        else:
            env = self._envelope()
            if env is not None:
                amps = amps * env.detach().cpu().unsqueeze(1)
        scaled = torch.complex(self.spec_real.cpu(), self.spec_imag.cpu()) * amps
        spec[1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
        spec[Tp - self.F_t : Tp, 1 : self.F_d + 1] = scaled.flip(0).conj()
        field = torch.fft.irfft2(spec, s=(Tp, self.D), norm="ortho")
        return field - field.mean(dim=0, keepdim=True)

    def field_strands(self, n_points: int = 240) -> dict:
        """Per-feature bias/variance decomposition for the cylinder-morph card.

        Each feature is a particle with two embeddings: a phase angle from the
        static field (the bias ring) and a pair of energies - ``bias`` (static
        field) and ``var`` (the input-conditional delta) - the orthogonal axes.
        ``separated`` is the fraction of total energy that is input-conditional;
        it is ~0 until an ``amp_modulation="input"`` field has trained, so the
        plane stays collapsed until the variance axis is actually learned.
        """
        with torch.no_grad():
            Tp = max(int(n_points), 2 * self.F_t + 1)
            static = self._sample_field(Tp)  # [Tp, D] static (bias)
            cond_coeffs = getattr(self, "_last_input_coeffs", None)
            if self.amp_modulation == "pure":
                # No static spectrum reaches the output: the sampled field IS
                # the conditional delta, so every hair is pure variance.
                cond, static = static, torch.zeros_like(static)
            elif self.amp_modulation == "input" and cond_coeffs is not None:
                cond = self._sample_field(Tp, coeffs=cond_coeffs)
            else:
                cond = static  # no conditional field -> variance axis is zero
            delta = cond - static

            bias_e = (static * static).sum(dim=0)  # [D]
            var_e = (delta * delta).sum(dim=0)  # [D]
            # Energy reference: peak bias, or peak variance for a bias-free
            # ("pure") field so its hairs still span the unit geometry.
            ref = bias_e.max()
            if ref < 1e-12:
                ref = var_e.max()
            ref = ref.clamp_min(1e-12)
            # Fundamental temporal Fourier component per feature -> phase angle.
            ang_src = cond if self.amp_modulation == "pure" else static
            fund = torch.fft.rfft(ang_src, dim=0)[1]  # [D] complex
            angle = torch.atan2(fund.imag, fund.real)  # [D]

            total = (bias_e.sum() + var_e.sum()).clamp_min(1e-12)
            return {
                "angle": angle.to(torch.float32).tolist(),
                "bias_energy": (bias_e / ref).to(torch.float32).tolist(),
                "var_energy": (var_e / ref).to(torch.float32).tolist(),
                "n": int(self.D),
                "separated": float((var_e.sum() / total).item()),
            }

    def capacity_split(self) -> dict:
        """Three-way spectral capacity allocation, summing to 1.

        bias = static-spectrum energy, variance = input-conditional delta
        energy (the same per-feature decomposition the strands card reads),
        dormant = headroom. The ceiling is a saturated spectrum: if every
        feature carried the peak feature's energy the field would be full, so
        the gap between that ceiling and the energy actually present is
        capacity still unwritten. A concentrated field (few features doing the
        work) reads as large dormant - the empirical "we have room left".
        """
        with torch.no_grad():
            Tp = max(240, 2 * self.F_t + 1)
            static = self._sample_field(Tp)  # [Tp, D]
            cond_coeffs = getattr(self, "_last_input_coeffs", None)
            if self.amp_modulation == "pure":
                cond, static = static, torch.zeros_like(static)
            elif self.amp_modulation == "input" and cond_coeffs is not None:
                cond = self._sample_field(Tp, coeffs=cond_coeffs)
            else:
                cond = static  # no conditional field -> variance is zero
            bias_e = (static * static).sum(dim=0)  # [D]
            var_e = ((cond - static) ** 2).sum(dim=0)  # [D]

            peak = torch.maximum(bias_e.max(), var_e.max()).clamp_min(1e-12)
            ceiling = peak * self.D  # every feature at peak = saturated
            bias, var = bias_e.sum(), var_e.sum()
            dormant = (ceiling - bias - var).clamp_min(0.0)
            total = (bias + var + dormant).clamp_min(1e-12)
            return {
                "harmonic_capacity_bias": float((bias / total).item()),
                "harmonic_capacity_variance": float((var / total).item()),
                "harmonic_capacity_dormant": float((dormant / total).item()),
            }

    def spiral(self, n_points: int = 720) -> dict:
        """The field's top-2 PCA cross-section unrolled along the position axis.

        The field is band-limited to F_t temporal frequencies, so sampling at
        Tp >= 2*F_t+1 points is exact (no aliasing) and far cheaper than the
        full-T irfft. We project each position's feature vector onto the top-2
        principal axes (``x``, ``y``) and let position itself be the third axis
        (``z``) - so the periodic loop unrolls into a rising spiral. ``band`` is
        the field energy left outside that plane (what the flat shadow hides),
        the analogue of the activation-curve percentile band. The Weyl phases
        are frozen, so the shape is a deterministic fingerprint of the learned
        amplitudes; ``participation_ratio`` reads effective dimensionality (~1 =
        one mode wins / consensus, high = spread / interference).
        """
        with torch.no_grad():
            Tp = max(int(n_points), 2 * self.F_t + 1)
            field = self._sample_field(Tp)

            _, S, Vh = torch.linalg.svd(field, full_matrices=False)
            xy = field @ Vh[:2].T  # [Tp, 2] in-plane shape
            row_sq = (field * field).sum(dim=1)
            resid = (
                (row_sq - (xy * xy).sum(dim=1)).clamp_min(0.0).sqrt()
            )  # off-plane spread

            scale = xy.abs().max().clamp_min(1e-8)  # scale is arbitrary post-PCA
            xy = xy / scale
            band = resid / scale

            s2 = S * S
            part = float((s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item())

            step = max(1, Tp // n_points)
            xy = xy[::step]
            band = band[::step]
            n = xy.shape[0]
            z = torch.linspace(0.0, 1.0, n)
            path = (
                torch.stack([xy[:, 0], xy[:, 1], z], dim=1).to(torch.float32).tolist()
            )
            band = band.to(torch.float32).tolist()
        return {
            "path": path,
            "band": band,
            "n": int(n),
            "participation_ratio": part,
        }

    def curve(self, n_points: int = 720, n_modes: int = 32) -> dict:
        """Top-2 PCA trajectory of the field over one period, as epicycle modes.

        Companion to :meth:`spiral`: same projection, but the period stays a
        closed planar loop and we return its dominant Fourier components so the
        dashboard can redraw it as a classic epicycle (nested rotating vectors
        whose tip traces the curve). Frozen Weyl phases make the shape a
        deterministic fingerprint of the learned amplitudes.
        """
        with torch.no_grad():
            Tp = max(int(n_points), 2 * self.F_t + 1)
            field = self._sample_field(Tp)

            _, S, Vh = torch.linalg.svd(field, full_matrices=False)
            traj = field @ Vh[:2].T  # [Tp, 2]

            s2 = S * S
            part = float((s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item())

            traj = traj / traj.abs().max().clamp_min(
                1e-8
            )  # scale is arbitrary post-PCA

            # Epicycle decomposition: dominant Fourier modes of the complex curve.
            z = torch.complex(traj[:, 0].contiguous(), traj[:, 1].contiguous())
            Z = torch.fft.fft(z) / Tp
            k = torch.arange(Tp)
            signed = torch.where(k <= Tp // 2, k, k - Tp)  # signed integer harmonics
            order = torch.argsort(Z.abs(), descending=True)[: int(n_modes)]
            modes = [
                {
                    "f": int(signed[i].item()),
                    "re": float(Z[i].real.item()),
                    "im": float(Z[i].imag.item()),
                }
                for i in order
            ]
            step = max(1, Tp // n_points)
            points = traj[::step].to(torch.float32).tolist()
        return {
            "modes": modes,
            "points": points,
            "n_points": int(len(points)),
            "participation_ratio": part,
        }

    def traces(self, n_time: int = 192, n_feat: int = 64) -> dict:
        """Per-feature temporal traces of the field b(t, d), normalized + ordered.

        Time-domain companion to the spectrum (frequency domain) and the spiral
        (PCA shape). Amplitude carries little here, so each trace is scaled to
        unit range and features are ordered by their phase at the dominant
        temporal frequency - turning a chaotic overlay into a traveling
        wavefront where the harmonics' interference reads as a clean moiré.
        """
        with torch.no_grad():
            Tp = max(int(n_time), 2 * self.F_t + 1)
            field = self._sample_field(Tp)

            n_feat = min(int(n_feat), self.D)
            f_idx = torch.linspace(0, self.D - 1, n_feat).round().long()
            sub = field[:, f_idx]  # [Tp, n_feat]

            # Order features by phase at the dominant shared temporal frequency.
            spec = torch.fft.rfft(sub, dim=0)  # [Tp//2+1, n_feat]
            mag = spec.abs().sum(dim=1)
            mag[0] = 0.0  # ignore DC
            f0 = int(torch.argmax(mag).item())
            order = torch.argsort(torch.angle(spec[f0]))
            sub = sub[:, order]

            sub = sub / sub.abs().amax(dim=0, keepdim=True).clamp_min(
                1e-8
            )  # amplitude out
            t_idx = torch.linspace(0, Tp - 1, int(n_time)).round().long()
            series = sub[t_idx].t().to(torch.float32).tolist()  # [n_feat][n_time]
        return {
            "traces": series,
            "n_time": int(len(t_idx)),
            "n_feat": int(n_feat),
        }

    def snake(self, n_points: int = 12) -> dict:
        """The field over a single sequence as a radial snake of ``n_points``.

        Angle is the field's PCA-2D phase at each sampled position; radius is
        *time*, collapsing toward the origin (newest sample at the center) so the
        sequence sinks inward as discrete blocks. "Circles the origin" is made
        falsifiable: the winding number is the full-resolution phase's signed
        total turn around the origin divided by 2*pi, and it reaches 1 only when
        the field's dominant mode completes a real loop over the sequence - so we
        report the number and only claim the loop when the data produces it.
        """
        import math

        with torch.no_grad():
            Tp = max(2 * self.F_t + 1, 64)
            field = self._sample_field(Tp)  # [Tp, D]
            fc = field - field.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(fc, full_matrices=False)
            xy = fc @ Vh[:2].T  # [Tp, 2] PCA trajectory over position
            ang = torch.atan2(xy[:, 1], xy[:, 0])  # [Tp] phase per position
            # Winding number: sum of wrapped phase steps over the full sequence.
            step = torch.diff(ang)
            step = (step + math.pi) % (2 * math.pi) - math.pi
            winding = float(step.sum().item() / (2 * math.pi))
            mag = xy.norm(dim=1)
            mag = mag / mag.max().clamp_min(1e-8)

            n = int(n_points)
            idx = torch.linspace(0, Tp - 1, n).round().long()
            points = [
                {
                    "angle": float(ang[int(idx[k])].item()),
                    "radius": (n - 1 - k)
                    / max(n - 1, 1),  # time -> radius, newest centered
                    "mag": float(mag[int(idx[k])].item()),
                }
                for k in range(n)
            ]
        return {
            "points": points,
            "winding": round(winding, 3),
            "circles_origin": abs(winding) >= 1.0,
            "n": n,
        }

    def correlation(self, n_feat: int = 64) -> dict:
        """Cosine similarity between feature trajectories over one period.

        ``C[i,j] = cos(b(:,i), b(:,j))`` across position - amplitude-invariant,
        so it reads pure co-evolution structure. This is the honest version of
        the old fake "correlation" panel: block/diagonal structure marks groups
        of features that rise and fall together. Packaged for the diverging
        ``corr_matrix`` renderer (values in [-1, 1]).
        """
        with torch.no_grad():
            Tp = max(2 * self.F_t + 1, int(n_feat))
            field = self._sample_field(Tp)
            n_feat = min(int(n_feat), self.D)
            f_idx = torch.linspace(0, self.D - 1, n_feat).round().long()
            sub = field[:, f_idx]  # [Tp, n_feat]
            sub = sub / sub.norm(dim=0, keepdim=True).clamp_min(1e-8)
            corr = (sub.t() @ sub).clamp(-1.0, 1.0)  # [n_feat, n_feat] cosine sim
        return {
            "grid": corr.to(torch.float32).tolist(),
            "grid_rows": int(n_feat),
            "grid_cols": int(n_feat),
            "x_range": [0, int(n_feat)],
            "y_range": [0, int(n_feat)],
            "max_count": 1.0,
        }

    def staircase(self, n_steps: int = 48) -> dict:
        """The harmonic ladder as ascending planks: each step is one harmonic.

        Where the spiral walks position, this walks the frequency/phase
        structure. Harmonics are ranked by energy (biggest at the base, tapering
        upward) and placed around a column by their frozen Weyl ``phase``. Each
        plank is oriented (``yaw``) along its own frequency direction
        ``atan2(f_d, f_t)`` rather than toward the center, and ``fnorm`` (radial
        frequency) plus amplitude drive its cross-section. Deterministic, and
        distinctly non-spiral.
        """
        with torch.no_grad():
            amps = self.amplitudes.detach().cpu()
            env = self._envelope()
            if env is not None:
                amps = amps * env.detach().cpu().unsqueeze(1)
            c = torch.complex(self.spec_real.cpu(), self.spec_imag.cpu()) * amps
            mag = c.abs().flatten()
            phase = torch.angle(c).flatten()
            n = min(int(n_steps), mag.numel())
            order = torch.argsort(mag, descending=True)[:n]
            sel = mag[order] / mag[order].max().clamp_min(1e-8)
            sel_phase = phase[order]
            # recover (f_t, f_d) per harmonic: orientation along its own
            # frequency direction, thickness from its radial frequency.
            ft = (order // self.F_d + 1).float()
            fd = (order % self.F_d + 1).float()
            yaw = 2.0 * torch.atan2(fd, ft)  # x2 so planks span a full half-turn
            # radial frequency, min-maxed over the shown set so thickness varies
            # (the top harmonics are all low-freq; a global norm would be flat).
            r = torch.sqrt(ft * ft + fd * fd)
            fnorm = (r - r.min()) / (r.max() - r.min()).clamp_min(1e-8)
            steps = [
                {
                    "a": float(sel[i].item()),
                    "phase": float(sel_phase[i].item()),
                    "yaw": float(yaw[i].item()),
                    "fnorm": float(fnorm[i].item()),
                }
                for i in range(n)
            ]
        return {"steps": steps, "n": int(n)}

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

    Owns both the field and the classifier, sized to :meth:`output_dims`
    (the encoder's declared byte-output layout in encoder mode, else
    ``(hidden_size, vocab_size)``). ``forward`` modulates the features
    with the field, then projects through ``lm_head`` - identical in
    standalone and encoder modes.
    """

    def __init__(
        self,
        config: Any,
        encoder: Optional[nn.Module] = None,
        amp_modulation: str = "off",
        build_classifier: bool = True,
    ) -> None:
        super().__init__(config, encoder)
        self._downstream = None  # injectable downstream classifier (grad-ratio)
        # Field period = the training window, so the F_t frequencies actually
        # oscillate within a sequence (the old max_position_embeddings sizing,
        # x8 for byte encoders, left the fastest component slower than one
        # block - a near-DC field). Positions past T wrap (the field is
        # periodic); block_size is in the encoder's own units (bytes for
        # byte-level tokenizers) since that is what the head sees.
        max_positions = int(
            getattr(config, "block_size", 0)
            or getattr(config, "max_position_embeddings", 32768)
            or 32768
        )

        dims = self.output_dims()
        if dims is None:
            # Encoder owns its full output pipeline (handles_loss, e.g. CALM):
            # nothing for this head to build.
            self.field = None
            self.lm_head = None
            return

        feature_dim, vocab_size = dims
        self.field = HarmonicField(
            hidden_dim=feature_dim,
            max_positions=max_positions,
            amp_modulation=amp_modulation,
        )
        # Transform-only stages (in a SequentialHead) skip the classifier - the
        # terminal head classifies, so the vocab projection would be dead.
        if build_classifier:
            self.lm_head = nn.Linear(feature_dim, vocab_size, bias=False)
            self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        else:
            self.lm_head = None

    def compose_repr(self) -> str:
        return "HarmonicField"

    def transform(self, hidden_states: Tensor) -> Tensor:
        """The field modulation - this head's contribution as a non-terminal
        SequentialHead stage."""
        return self.field(hidden_states) if self.field is not None else hidden_states

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        h = self.transform(hidden_states)
        return self.lm_head(h) if self.lm_head is not None else h

    @property
    def classifier(self) -> Optional[nn.Module]:
        return self.lm_head

    def set_downstream(self, classifier: Optional[nn.Module]) -> None:
        """Point grad-ratio at the classifier this field actually feeds, when
        used transform-only in a SequentialHead (else it has none of its own).

        Held in a tuple so it is *not* registered as a submodule: the
        downstream is owned elsewhere (the terminal head), and registering it
        would duplicate its params in our state_dict and leak its metric
        descriptions into this stage's module walk."""
        self._downstream = (classifier,) if classifier is not None else None

    def aux_losses(self) -> dict:
        if self.field is None:
            return {}
        aux = self.field.aux_loss()
        return {"harmonic_smoothness": aux} if aux is not None else {}

    def _downstream_classifier(self) -> Optional[nn.Module]:
        """The learnable projection the field feeds into: our own ``lm_head``
        when terminal, else the injected downstream classifier."""
        if self.lm_head is not None:
            return self.lm_head
        return self._downstream[0] if self._downstream else None

    def dashboard_snapshots(self) -> dict:
        """Amplitude grid magnitudes for the spectrum heatmap.

        Returns the field's ``|amp[f_t, f_d]|`` matrix and the
        irrationals used to seed phases, packaged for the generic
        ``heatmap_2d`` renderer (grid + axis ranges + max).
        """
        if self.field is None:
            return {}
        amps = self.field.effective_amplitudes().abs().to("cpu", dtype=torch.float32)
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
            },
            "harmonic_spiral": self.field.spiral(),
            "harmonic_curve": self.field.curve(),
            "harmonic_traces": self.field.traces(),
            "harmonic_correlation": self.field.correlation(),
            "harmonic_staircase": self.field.staircase(),
            "harmonic_strands": self.field.field_strands(),
            "harmonic_snake": self.field.snake(),
        }

    def training_metrics(self) -> dict:
        if self.field is None:
            return {}
        amps = self.field.amplitudes
        out = {
            "harmonic_amplitudes_norm": float(amps.detach().norm().item()),
            "harmonic_concentration": float(self.field.concentration().item()),
            "harmonic_smoothness": float(self.field.smoothness().item()),
            "harmonic_env_depth": self.field.envelope_depth(),
            **self.field.capacity_split(),
        }

        # grad_ratio reads whether learning is flowing into the field or
        # past it through the downstream classifier. Skip silently if
        # gradients aren't available yet (pre-first-step) or the classifier
        # exposes no readable weight tensor.
        amps_grad = amps.grad
        head_weight = _classifier_weight(self._downstream_classifier())
        head_grad = head_weight.grad if head_weight is not None else None
        if amps_grad is not None and head_grad is not None:
            head_norm = float(head_grad.detach().norm().item())
            if head_norm > 0:
                out["harmonic_grad_ratio"] = (
                    float(amps_grad.detach().norm().item()) / head_norm
                )
        return out


def _classifier_weight(mod: Optional[nn.Module]) -> Optional[Tensor]:
    """Primary weight tensor of a downstream classifier: ``weight`` for a
    Linear, ``centers`` for the crystal classifier."""
    if mod is None:
        return None
    for attr in ("weight", "centers"):
        w = getattr(mod, attr, None)
        if isinstance(w, torch.Tensor):
            return w
    return None
