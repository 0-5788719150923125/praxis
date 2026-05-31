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
# amplitude norm. Set to 0 to disable.
# See next/harmony.md for the derivation.
SMOOTHNESS_LAMBDA: float = 0.01

# Amplitude modulation: an envelope over the temporal-frequency (f_t) axis,
# applied as ``amp[f_t, :] *= env[f_t]``. "static" seeds a single mid-band
# oscillation; "learned" makes the envelope a handful of learnable cosine
# coefficients so the wave adapts. The envelope's basis is zero at f_t -> 0,
# so it cannot reintroduce the flat (constant-over-position) mode that the bare
# grid settles into. See ``HarmonicField`` and ``next/harmony.md``.
AMP_MOD_DEPTH: float = 0.5   # peak envelope modulation, tanh-bounded
AMP_MOD_BASIS_K: int = 6     # learned envelope = K low-frequency f_t modes


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

        self.amplitudes = nn.Parameter(torch.empty(self.F_t, self.F_d))
        nn.init.normal_(self.amplitudes, mean=0.0, std=AMPLITUDE_INIT_STD)

        # Amplitude envelope over f_t. "static" and "learned" share one formula
        # (so they are identical at init); only "learned" lets the coefficients
        # move. Init = a single mid-band oscillation (coeff 0 = 1, rest 0).
        if amp_modulation not in ("off", "static", "learned"):
            raise ValueError(
                f"amp_modulation must be off|static|learned, got {amp_modulation!r}"
            )
        self.amp_modulation = amp_modulation
        if amp_modulation != "off":
            self.register_buffer(
                "amp_basis", _envelope_basis(self.F_t, AMP_MOD_BASIS_K), persistent=False
            )
            coeffs = torch.zeros(AMP_MOD_BASIS_K)
            coeffs[0] = 1.0
            if amp_modulation == "learned":
                self.amp_coeffs = nn.Parameter(coeffs)
            else:
                self.register_buffer("amp_coeffs", coeffs, persistent=False)

    def _field(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> Tensor:
        rfft_D = self.D // 2 + 1
        spec = torch.zeros(self.T, rfft_D, dtype=torch.complex64, device=device)
        amps = self.amplitudes
        env = self._envelope()
        if env is not None:
            amps = amps * env.unsqueeze(1)  # modulate each f_t row by env[f_t]
        scaled = (
            torch.complex(self.spec_real.to(device), self.spec_imag.to(device))
            * amps
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

    def _envelope(self) -> Optional[Tensor]:
        """``[F_t]`` amplitude envelope over the temporal-frequency axis, or
        None when modulation is off. ``1 + depth*tanh(basis @ coeffs)`` stays
        positive (a true amplitude envelope) and bounded."""
        if self.amp_modulation == "off":
            return None
        return 1.0 + AMP_MOD_DEPTH * torch.tanh(self.amp_basis @ self.amp_coeffs)

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

    def _sample_field(self, Tp: int) -> Tensor:
        """Real field [Tp, D] sampled over one period, mean-centered over time.

        Alias-free for Tp >= 2*F_t+1 (the field is band-limited to F_t temporal
        frequencies), and far cheaper than the full-T irfft. Shared by every
        snapshot view below.
        """
        rfft_D = self.D // 2 + 1
        spec = torch.zeros(Tp, rfft_D, dtype=torch.complex64)
        amps = self.amplitudes.detach().cpu()
        env = self._envelope()
        if env is not None:
            amps = amps * env.detach().cpu().unsqueeze(1)
        scaled = torch.complex(self.spec_real.cpu(), self.spec_imag.cpu()) * amps
        spec[1 : self.F_t + 1, 1 : self.F_d + 1] = scaled
        spec[Tp - self.F_t : Tp, 1 : self.F_d + 1] = scaled.flip(0).conj()
        field = torch.fft.irfft2(spec, s=(Tp, self.D), norm="ortho")
        return field - field.mean(dim=0, keepdim=True)

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
            resid = (row_sq - (xy * xy).sum(dim=1)).clamp_min(0.0).sqrt()  # off-plane spread

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
            path = torch.stack([xy[:, 0], xy[:, 1], z], dim=1).to(torch.float32).tolist()
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

            traj = traj / traj.abs().max().clamp_min(1e-8)  # scale is arbitrary post-PCA

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

            sub = sub / sub.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)  # amplitude out
            t_idx = torch.linspace(0, Tp - 1, int(n_time)).round().long()
            series = sub[t_idx].t().to(torch.float32).tolist()  # [n_feat][n_time]
        return {
            "traces": series,
            "n_time": int(len(t_idx)),
            "n_feat": int(n_feat),
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
        max_positions = int(getattr(config, "max_position_embeddings", 32768) or 32768)
        if "byte" in str(getattr(config, "encoder_type", "")):
            max_positions = max(max_positions, max_positions * 8)

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
        used transform-only in a SequentialHead (else it has none of its own)."""
        self._downstream = classifier

    def aux_losses(self) -> dict:
        if self.field is None:
            return {}
        aux = self.field.aux_loss()
        return {"harmonic_smoothness": aux} if aux is not None else {}

    def _downstream_classifier(self) -> Optional[nn.Module]:
        """The learnable projection the field feeds into: our own ``lm_head``
        when terminal, else the injected downstream classifier."""
        return self.lm_head if self.lm_head is not None else self._downstream

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
