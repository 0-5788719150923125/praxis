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
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead, decode_context

IRR_T: float = math.pi
IRR_D: float = math.e
ALPHA: float = 1.0  # radial 1/f^alpha pink-noise decay
AMPLITUDE_INIT_STD: float = 1.0
# Forward-shift smoothness prior. The closed-form for "b_t predicts b_{t+1}" in
# our 2D Fourier basis with frozen Weyl phases reduces to a quadratic penalty on
# temporal frequency, normalized by amplitude norm (see next/harmony.md), giving
# a scale-free S in [0, 1]: 0 = all mass at f_t=1, ~1/3 = isotropic, 1 = all mass
# at f_t=F_t.
#
# The strength is no longer a fixed weight. A fixed lambda is the wrong shape of
# constant: it sets the EQUILIBRIUM the prior settles at, but nothing about the
# problem tells you what value produces the smoothness you want, and if the prior
# is losing its fight against NLL you cannot tell without re-tuning. So lambda is
# a Lagrange multiplier on a constraint whose TARGET is measured from the data:
#
#   S_target = normalized 2nd spectral moment of the hidden states the field
#              multiplies - the same statistic S, computed on the signal instead
#              of the grid. "The field should vary across position at the rate
#              the signal it multiplies does."
#   lambda  <- softplus(rho),  rho += eta * (S - S_target)     [dual ascent]
#
# The two quantities are on one scale by construction, and measurably so: white
# hidden states give 0.340 against the grid's isotropic 0.341. lambda now rises
# on its own when the prior is losing and relaxes when it has won, so the run
# reports where the prior settled rather than being told.
#
# What is left is honestly still three constants, but they are a different KIND
# of constant: an initial condition (continuity with the fixed-lambda runs), an
# approach rate, and a safety cap. None of them sets the equilibrium - the data
# does. The rate is the one that needed care: a controller must be SLOWER than
# the plant it steers, and an amplitude grid being dragged by a regularizer
# against NLL is a slow plant. A sweep over dual rate x grid-response time (100x
# range) puts 0.003 as the only value that never pins against the cap and still
# reaches the target; 0.3 saturates the cap on half of all steps, and 0.001
# leaves the constraint unmet. Below 0.003 the prior is simply weak, above it
# the multiplier outruns the grid and slams the cap - which is a visible
# failure, not a silent one, because lambda is logged.
SMOOTHNESS_LAMBDA_INIT: float = 0.01  # lambda at step 0 = the old fixed value
SMOOTHNESS_DUAL_ETA: float = 0.003  # dual ascent rate on rho; see note below
SMOOTHNESS_LAMBDA_MAX: float = 1.0  # cap: aux = lambda*S <= 1, under byte NLL
SMOOTHNESS_TARGET_EMA: float = 0.99  # target is a batch estimate; smooth it
SMOOTHNESS_PROBE_ROWS: int = 2  # sequences sampled for the target measurement
SMOOTHNESS_PROBE_LEN: int = 256  # positions sampled for the target measurement

# Amplitude modulation: a separable envelope over the frequency grid, applied
# as ``amp[f_t, f_d] *= env[f_t, f_d]``. "static" seeds a single mid-band
# oscillation; "learned" lets the coefficients adapt. The envelope's basis is
# zero at f -> 0 on both axes, so it cannot reintroduce the flat
# (constant-over-position) mode that the bare grid settles into. See
# ``HarmonicField`` and ``next/harmony.md``.
#
# The mode counts are COMPLETE (K_t = F_t, K_d = F_d), not truncated, and so are
# derived from the grid rather than set by hand. The previous truncation to six
# f_t modes was a hard cap doing the job of a prior: it bounded the
# input-conditional delta - the variance axis - to six degrees of freedom on one
# axis, while the static grid it modulates has F_t*F_d. By the interference-
# capacity proposition (research/body.tex, Sec. 5) the configurations a harmonic
# field can distinguish are counted in the SPREAD of its spectrum, so a delta
# capped at six coefficients cannot carry them however large the grid grows.
# Completeness costs O(F_t + F_d) coefficients and is a strict generalisation:
# the f_d coefficients are zero-initialised, so at init the envelope is exactly
# the old f_t profile broadcast over f_d, bit for bit. The smoothness prior and
# the tanh depth remain the (soft) controls on how much of the new room is used.
AMP_MOD_DEPTH: float = 0.5  # peak envelope modulation, tanh-bounded

# Fast weights: a per-token, delta-rule recurrent overlay on the spectrum. A small
# linear-attention memory (ELU+1 kernel, delta write, z-normalized - the
# Infini-Attention rule) reads a per-token vector from the causal context; that
# read drives a bounded rank-r factoring ``u_t (x) v_t`` added to the amplitude
# grid. The slow grid is the foundation; this is the secondary, test-time,
# surprise-stabilized modulation. The field is linear in the grid, so the overlay
# is just the field of the per-token delta, added to the base field.
FAST_WEIGHT_RANK: int = 2  # rank of the per-token grid delta
FAST_MEM_DIM: int = 32  # key/value width of the delta-rule memory
# How often the compressed bank refreshes. The read is bank + within-segment
# causal prefix, so this is a compression granularity, NOT a blind spot: every
# token still sees every earlier token, whatever the segment size. Bigger
# segments mean fewer steps of the sequential bank recurrence (cheaper) and more
# of the context served exactly rather than through the compressed bank.
FAST_SEGMENT: int = 64
FAST_WEIGHT_SCALE: float = 0.25  # per-cell cap; keeps the slow grid foundational
# Forwards between refreshes of the snapshot-only `_fast_repr` readout. Nothing
# in training reads it; the dashboard samples it far slower than every step.
FAST_REPR_INTERVAL: int = 25
FAST_EPS: float = 1e-6
# Smoothing for the reported input-conditional envelope (the "input" arm's separate
# pooled-envelope pathway). The raw per-batch coeffs swing hard step to step; the
# snapshots want a representative, not one draw. The fast-weight overlay no longer
# needs this - its delta-rule state is already a stable, surprise-filtered read.
COEFF_EMA: float = 0.9
# Step 3 (deferred): a per-depth learned spectral temperature - |a_k|^(1/T(depth))
# via an nn.Embedding(depth, 1) delta - would let each decoder pass sharpen or
# release the harmonics. Per-depth Serpent activations likely already cover much
# of this conditional shaping; revisit if they do not.


def _envelope_basis(n: int, K: int) -> torch.Tensor:
    """``[n, K]`` sine modes over one frequency axis of length ``n``.

    Mode ``k`` is ``sin(pi*k*f/n)`` - a smooth wave that vanishes at ``f -> 0``,
    so any combination of modes leaves the flat DC component untouched. Mode 1
    is a single hump peaking mid-band. Used for both the f_t and f_d axes; at
    ``K = n`` the modes form a complete basis for that axis.
    """
    f = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    k = np.arange(1, K + 1, dtype=np.float64).reshape(1, -1)
    return torch.from_numpy(np.sin(np.pi * f * k / n)).to(torch.float32)


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
                "group_order": 40,
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
                "series_group": "harmonic_smooth_pair",
                "series_label": "field (grid)",
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
        "harmonic_smooth_lambda": {
            "description": (
                "The smoothness prior's Lagrange multiplier. Starts at 0.01 - "
                "the old fixed weight - and then moves on its own: it rises "
                "while the field is rougher than the signal it multiplies and "
                "relaxes once it is smoother. Reading it tells you whether the "
                "prior is winning its fight with NLL, which a fixed weight "
                "never could. Pinned at the 1.0 cap means the constraint is "
                "unreachable and the target is wrong, not the multiplier."
            ),
            "chart": {
                "title": "Smoothness Multiplier",
                "y_label": "lambda",
                "y_scale": "logarithmic",
                "group": "harmonic_head",
                "order": 48,
            },
        },
        "harmonic_smooth_target": {
            "description": (
                "The measured smoothness target: the normalized second "
                "spectral moment of the hidden states the field multiplies, "
                "EMA-smoothed. Same statistic and same scale as "
                "harmonic_smoothness, so the two are directly comparable - the "
                "gap between them is the constraint violation driving the "
                "multiplier. Near 1/3 means the signal still looks like noise; "
                "falling means the trunk is developing temporal structure."
            ),
            "chart": {
                "title": "Smoothness vs Target",
                "y_label": "normalized 2nd spectral moment",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 49,
                "series_group": "harmonic_smooth_pair",
                "series_label": "target (signal)",
            },
        },
        "harmonic_env_fd_share": {
            "description": (
                "Share of envelope coefficient energy sitting on the FEATURE "
                "axis (f_d) rather than the temporal axis (f_t). Exactly 0 at "
                "init, because the f_d coefficients are zero-seeded and the "
                "envelope starts as a pure f_t profile broadcast across "
                "features. A value that stays at 0 means the model never used "
                "the feature axis and the envelope is still effectively "
                "row-wise - the direct falsifier for completing the basis."
            ),
            "chart": {
                "title": "Envelope Feature-Axis Share",
                "y_label": "f_d share of coeff energy",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 46,
            },
        },
        "harmonic_env_modes": {
            "description": (
                "Effective number of active envelope modes - the participation "
                "ratio of the coefficient vector, so a value of 1 means one "
                "mode carries everything and F_t+F_d means all are equally "
                "used. 1.0 at init (a single mid-band hump). This counts the "
                "degrees of freedom the variance axis is actually spending; "
                "the interference-capacity proposition says configurations are "
                "carried by spread, so a value pinned near 1 is a variance "
                "axis with nothing to spend."
            ),
            "chart": {
                "title": "Envelope Effective Modes",
                "y_label": "Effective modes",
                "y_scale": "logarithmic",
                "group": "harmonic_head",
                "order": 47,
            },
        },
        "harmonic_fast_norm": {
            "description": (
                "L2 norm of the fast-weight overlay (EMA representative). The "
                "secondary, context-written delta on the spectrum - should stay "
                "small relative to the grid; growth means the model leans on "
                "test-time modulation, a spike means it is swamping the foundation."
            ),
            "chart": {
                "title": "Fast-Weight Magnitude",
                "y_label": "||fast overlay||",
                "y_scale": "linear",
                "group": "harmonic_head",
                "order": 49,
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
                "input-conditional delta the envelope writes from each "
                "position's causal prefix. "
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
    }

    def __init__(
        self,
        hidden_dim: int,
        max_positions: int,
        F_t: Optional[int] = None,
        F_d: Optional[int] = None,
        amp_modulation: str = "off",
        fast_weights: bool = False,
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

        # Position x temporal-frequency phase table, [T, F_t]. Depends only on
        # (T, F_t), never on the input, but was rebuilt on every forward - twice,
        # once in _eval_field and again in _field_fast - as arange + arange +
        # mul + cos + sin. Precomputed here and sliced to seq_len instead;
        # sequences longer than T fall back to computing (see _phase_table).
        pos = torch.arange(self.T, dtype=torch.float32).unsqueeze(1)
        freq = torch.arange(1, self.F_t + 1, dtype=torch.float32)
        pos_ang = 2 * math.pi * pos * freq / self.T
        self.register_buffer("pos_cos", torch.cos(pos_ang), persistent=False)
        self.register_buffer("pos_sin", torch.sin(pos_ang), persistent=False)

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
            # Complete sine bases on both grid axes; the coefficient vector is
            # the concatenation [c_t (F_t), c_d (F_d)], so K is derived from the
            # grid rather than chosen.
            self.register_buffer(
                "amp_basis",
                _envelope_basis(self.F_t, self.F_t),
                persistent=False,
            )
            self.register_buffer(
                "amp_basis_d",
                _envelope_basis(self.F_d, self.F_d),
                persistent=False,
            )
            self.amp_K = self.F_t + self.F_d
            coeffs = torch.zeros(self.amp_K)
            if amp_modulation != "pure":
                # Coefficient 0 is the first f_t mode: a single mid-band hump,
                # exactly the old init. Every f_d coefficient starts at zero, so
                # the envelope is constant along f_d until the model learns
                # otherwise - identical to the previous row-wise envelope.
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
                # axis. A zero-init projection from pooled hidden states (the
                # causal prefix at every position - see _field_conditional) to
                # envelope coefficients: the field is
                # exactly the static (bias) field at init and learns its
                # input-dependence, orthogonal to the static spectrum. ``_last_input_coeffs`` keeps a
                # representative coeff set (mean over the last batch) so the
                # strands snapshot can rebuild the conditional field with no batch.
                self.amp_input = nn.Linear(self.D, self.amp_K, bias=False)
                nn.init.zeros_(self.amp_input.weight)
                self.register_buffer(
                    "_last_input_coeffs", coeffs.clone(), persistent=False
                )

        # Fast weights: slow ``amplitudes`` is the foundation, this is the
        # secondary, per-token overlay. ``fast_qkv`` feeds a delta-rule memory;
        # its per-token read drives rank-r grid factors. ``fast_u`` zero-init (so
        # the overlay is exactly zero at init, identity start) while ``fast_v``
        # seeds the other factor so gradients still flow.
        # Smoothness constraint state. ``smooth_rho`` is the dual variable
        # (lambda = softplus(rho)); ``smooth_target`` is the EMA of the measured
        # hidden-state roughness. Both are persistent - they are learned state,
        # not derived, so a resumed run must not reset them to the cold start.
        # The dual step is taken here rather than by the model optimizer on
        # purpose: LionGeo routes scalars to a SIGN-based Lion secondary, which
        # discards the violation magnitude and (at its lr=3e-4) moves the
        # multiplier far too slowly to reach equilibrium inside a run.
        self._smooth_rho = math.log(math.expm1(SMOOTHNESS_LAMBDA_INIT))
        self._smooth_target = -1.0
        self.register_buffer(
            "_smooth_rho_buf", torch.tensor(self._smooth_rho), persistent=True
        )
        self.register_buffer("_smooth_target_buf", torch.tensor(-1.0), persistent=True)

        self.fast_weights = bool(fast_weights)
        if self.fast_weights:
            self.fast_rank = FAST_WEIGHT_RANK
            self.fast_mem = FAST_MEM_DIM
            self.fast_qkv = nn.Linear(self.D, 3 * self.fast_mem, bias=False)
            self.fast_u = nn.Linear(
                self.fast_mem, self.fast_rank * self.F_t, bias=False
            )
            self.fast_v = nn.Linear(
                self.fast_mem, self.fast_rank * self.F_d, bias=False
            )
            nn.init.zeros_(self.fast_u.weight)
            nn.init.normal_(self.fast_v.weight, std=0.02)
            self.register_buffer(
                "_fast_repr", torch.zeros(self.F_t, self.F_d), persistent=False
            )
            # Lower-triangular mask for the WITHIN-segment causal read, so a
            # token attends to its own segment's prefix (itself included) and
            # nothing later. Without this half, a token sees only whole prior
            # segments and is blind to everything since the last boundary.
            self.register_buffer(
                "fast_causal",
                torch.ones(FAST_SEGMENT, FAST_SEGMENT).tril().bool(),
                persistent=False,
            )

    def _fast_retrieve(
        self,
        hidden_states: Tensor,
        state: Optional[dict] = None,
        new_state: Optional[dict] = None,
    ) -> Tensor:
        """Per-token read ``[B, L, d]`` from a delta-rule linear-attention memory
        over the causal context. ELU+1 kernel, delta write (value minus what the
        memory already predicts), z-normalized retrieval - the Infini-Attention
        rule. The bank refreshes once per segment; queries are per token, so the
        read varies token to token. Strictly causal: a segment reads the bank
        built from prior segments, then writes itself.

        The delta correction is against the prior bank, so the write is closed
        form: ``phi_k^T (phi_k mem / z) = (phi_k^T (phi_k / z)) mem = A_n mem``.
        The per-segment ``A_n``, write matrix ``B_n = phi_k^T v`` and the ``z``
        normalizer are all computed batched; only the affine matrix recurrence
        ``mem_{n+1} = (I - A_n) mem_n + B_n`` stays sequential, over the handful
        of segments (no per-token loop, no ``S`` dim in the loop body).

        Under cached decode ``state`` carries the bank and normalizer after the
        last CLOSED segment plus the raw states of the open segment (its
        ``fast_tail``); the tail is re-read in front of the new chunk so the
        within-segment causal half stays exact, and ``new_state`` receives the
        bank advanced through every segment the chunk closed and the new tail.
        A one-token chunk therefore reads exactly what it would have read as
        the last position of a full-sequence forward.
        """
        d, seg = self.fast_mem, FAST_SEGMENT
        tail = state.get("fast_tail") if state is not None else None
        n_tail = 0
        if tail is not None and tail.shape[0] == hidden_states.shape[0]:
            n_tail = tail.shape[1]
            hidden_states = torch.cat([tail.to(hidden_states.dtype), hidden_states], 1)
        # Project in the parameters' own dtype - a hardcoded ``.float()`` faults
        # against any run whose weights are not fp32 - then give the recurrence
        # at least fp32 headroom, which is what that cast was there for. An
        # fp32 run gets exactly what it got before; fp64 keeps its width.
        out_dtype = hidden_states.dtype
        q, k, v = self.fast_qkv(hidden_states).split(d, dim=-1)
        work = torch.promote_types(q.dtype, torch.float32)
        if q.dtype != work:
            q, k, v = q.to(work), k.to(work), v.to(work)
        sig_q, sig_k = F.elu(q) + 1.0, F.elu(k) + 1.0
        b_size, seq_len, _ = q.shape

        n_seg = (seq_len + seg - 1) // seg
        pad = n_seg * seg - seq_len
        if pad:  # zero-pad post-kernel so the tail contributes nothing to any bank
            zr = sig_k.new_zeros(b_size, pad, d)
            sig_q, sig_k, v = (
                torch.cat([sig_q, zr], 1),
                torch.cat([sig_k, zr], 1),
                torch.cat([v, zr], 1),
            )
        qk = sig_q.view(b_size, n_seg, seg, d)
        kk = sig_k.view(b_size, n_seg, seg, d)
        vv = v.view(b_size, n_seg, seg, d)

        mem = q.new_zeros(b_size, d, d)
        z0 = q.new_zeros(b_size, d)
        if state is not None and "fast_mem" in state:
            mem = state["fast_mem"].to(q.device, q.dtype)
            z0 = state["fast_z"].to(q.device, q.dtype)

        dz = kk.sum(dim=2)  # [B, N, d] per-segment z increment
        # exclusive cumsum = bank from prior segments (plus the carried bank)
        z_prior = z0.unsqueeze(1) + dz.cumsum(dim=1) - dz
        dk = torch.einsum("bnsd,bnd->bns", kk, z_prior) + FAST_EPS  # [B, N, S]
        a_mat = torch.einsum("bnsd,bnse->bnde", kk, kk / dk.unsqueeze(-1))  # A_n
        b_mat = torch.einsum("bnsd,bnse->bnde", kk, vv)  # B_n = phi_k^T v

        mems = []
        for n in range(n_seg):
            mems.append(mem)  # bank from prior segments (mem_0 = carried or 0)
            mem = mem + b_mat[:, n] - a_mat[:, n] @ mem
        mem_stack = torch.stack(mems, dim=1)  # [B, N, d, d]

        if new_state is not None:
            # Commit through the segments this chunk CLOSED; the open remainder
            # is carried raw and re-read next call.
            n_full = seq_len // seg
            new_state["fast_mem"] = (mems[n_full] if n_full < n_seg else mem).detach()
            new_state["fast_z"] = (z0 + dz[:, :n_full].sum(dim=1)).detach()
            new_state["fast_tail"] = hidden_states[:, n_full * seg :].detach()

        # Cross-segment: the compressed bank of everything before this segment.
        num = torch.einsum("bnsd,bnde->bnse", qk, mem_stack)  # [B, N, S, d]
        den = torch.einsum("bnsd,bnd->bns", qk, z_prior)  # [B, N, S]

        # Within-segment: the causal prefix the bank cannot hold yet. Without
        # this the read is blind from the last segment boundary to the token
        # itself - a whole segment's worth, for every token - and segment 0 is
        # blind entirely, since its bank is the initial zeros. Standard chunked
        # linear attention: score against the segment's own keys, mask to the
        # causal prefix, and extend the SAME z normalizer so the two halves are
        # one weighted average rather than two scales glued together.
        scores = torch.einsum("bnsd,bntd->bnst", qk, kk)  # [B, N, S, S]
        scores = scores.masked_fill(~self.fast_causal[:seg, :seg], 0.0)
        num = num + torch.einsum("bnst,bnte->bnse", scores, vv)
        den = den + scores.sum(dim=-1)

        reads = num / (den + FAST_EPS).unsqueeze(-1)
        reads = reads.reshape(b_size, n_seg * seg, d)[:, n_tail:seq_len]
        return reads.to(out_dtype)

    def _field_fast(
        self,
        hidden_states: Tensor,
        offset: int = 0,
        state: Optional[dict] = None,
        new_state: Optional[dict] = None,
    ) -> Tensor:
        """Per-token bounded rank-r overlay field ``[B, L, D]``. The retrieved
        context vector drives factors ``u_t (x) v_t``; since the field is linear
        in the grid this is the field of the per-token delta alone, summed into
        the base field upstream. ``f_t`` is contracted against the frozen phase
        inside the matmul, so the ``[B, L, F_t, F_d]`` grid is never built."""
        r = self._fast_retrieve(hidden_states, state, new_state)
        b_size, seq_len, _ = r.shape
        u = torch.tanh(self.fast_u(r)).view(b_size, seq_len, self.fast_rank, self.F_t)
        v = torch.tanh(self.fast_v(r)).view(b_size, seq_len, self.fast_rank, self.F_d)
        self._update_fast_repr(u, v)

        device = hidden_states.device
        ca, sa = self._phase_table(seq_len, device, offset)  # shared with _eval_field
        cos_a = ca.view(1, seq_len, 1, self.F_t)
        sin_a = sa.view(1, seq_len, 1, self.F_t)
        p_re, p_im = self.spec_real.to(device), self.spec_imag.to(device)
        c = torch.einsum("blrf,fd->blrd", u * cos_a, p_re) - torch.einsum(
            "blrf,fd->blrd", u * sin_a, p_im
        )
        w = (v * c).sum(dim=2)  # [B, L, F_d]
        scale = FAST_WEIGHT_SCALE / self.fast_rank
        b = scale * (2.0 / math.sqrt(self.T * self.D)) * (w @ self.basis_d.to(device))
        return b.to(hidden_states.dtype)

    def _update_fast_repr(self, u: Tensor, v: Tensor) -> None:
        """[F_t, F_d] representative of the overlay for the snapshots: the
        rank-factored outer product of the batch-time-mean factors. No EMA - the
        mean over B*L tokens is already a stable readout of a stable state.

        Refreshed on a cadence, not every forward: nothing in training reads
        ``_fast_repr`` - it exists for the dashboard snapshot - and recomputing
        it per call was ~9% of the overlay for a number sampled far more slowly
        than that. The last value stands in between refreshes.
        """
        self._repr_tick = (getattr(self, "_repr_tick", 0) + 1) % FAST_REPR_INTERVAL
        if self._repr_tick != 1:
            return
        with torch.no_grad():
            scale = FAST_WEIGHT_SCALE / self.fast_rank
            u_m, v_m = u.mean(dim=(0, 1)), v.mean(dim=(0, 1))  # [r, F_t], [r, F_d]
            self._fast_repr = scale * torch.einsum("rf,rd->fd", u_m, v_m).to(
                self._fast_repr.dtype
            )

    def _eval_field(
        self, scaled: Tensor, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tensor:
        """Band-limited field at positions ``0..seq_len-1`` via two small
        matmuls - exactly the ortho-normed irfft2 of the (Hermitian-extended)
        ``[T, D]`` spectrum, but never materializing it: only F_t * F_d bins
        are nonzero, so the transform is separable. ``scaled`` is the complex
        amplitude grid ``[..., F_t, F_d]``; returns ``[..., seq_len, D]``.
        Memory is O(seq_len * D) instead of O(T * D), which is what makes this
        affordable when T spans the full context and seq_len is one block.

        The 2/sqrt(T*D) factor folds the Hermitian doubling on the T axis
        into irfft2's ortho norm, preserving spectral energy: field std stays
        ~ amp std * sqrt(F_t * F_d / (T * D)), independent of T scale.
        """
        cos_a, sin_a = self._phase_table(seq_len, device, offset)
        a = cos_a @ scaled.real - sin_a @ scaled.imag
        return (2.0 / math.sqrt(self.T * self.D)) * (a @ self.basis_d.to(device))

    def _phase_table(self, seq_len: int, device: torch.device, offset: int = 0):
        """``(cos, sin)`` of ``2*pi*t*f_t/T`` for ``t`` in
        ``offset .. offset+seq_len-1``.

        Sliced from the precomputed ``[T, F_t]`` buffers - these are constants
        of the module, not of the input. Positions past ``T`` are computed on
        the spot; the field is band-limited to period ``T``, so that path is
        only reachable if a config asks for positions past one period. The
        offset is the cached-decode continuation: the trunk hands the head only
        the new suffix, and the field is anchored to ABSOLUTE position.
        """
        if offset + seq_len <= self.T:
            return (
                self.pos_cos[offset : offset + seq_len].to(device),
                self.pos_sin[offset : offset + seq_len].to(device),
            )
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        f_t = torch.arange(1, self.F_t + 1, device=device, dtype=torch.float32)
        ang = 2 * math.pi * t.unsqueeze(1) * f_t / self.T
        return torch.cos(ang), torch.sin(ang)

    def _field(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype],
        offset: int = 0,
    ) -> Tensor:
        amps = self.amplitudes
        env = self._envelope()
        if env is not None:
            amps = amps * env  # per-cell envelope over the (f_t, f_d) grid
        scaled = (
            torch.complex(self.spec_real.to(device), self.spec_imag.to(device)) * amps
        )
        b = self._eval_field(scaled, seq_len, device, offset)
        return b.to(dtype) if dtype is not None else b

    @torch._dynamo.disable
    @torch.no_grad()
    def _update_smoothness_dual(self, hidden_states: Tensor) -> None:
        """Measure the target from the incoming signal and take one dual step.

        Runs on the field's INPUT, which is upstream of its own output, so the
        measurement is not reading back the field it is about to constrain. The
        trunk does adapt to the field across steps - that is true of any
        regularizer - but within a forward there is no loop.

        The dual state lives in PYTHON FLOATS, not in the buffers, and this
        method is hidden from Dynamo. Both are deliberate. A buffer mutated
        in-place inside a compiled forward bumps its autograd version counter
        between the point AOTAutograd saves tensors and the point backward runs
        them, which aborts the step with "variable needed for gradient
        computation has been modified by an inplace operation". Keeping the
        controller in plain Python takes it out of the traced graph entirely:
        nothing autograd saved can be mutated, because the controller never
        touches a tensor autograd knows about. The buffers still exist, and are
        synced at checkpoint time only (see ``_save_to_state_dict``).
        """
        if not self.training or SMOOTHNESS_DUAL_ETA <= 0.0:
            return
        h = hidden_states
        if h.dim() != 3 or h.shape[-2] < 4:
            return  # too short for a meaningful temporal statistic
        # The target is an EMA-smoothed setpoint, not a precision measurement,
        # so it is read from a slice: a couple of sequences and a bounded window
        # keep the cost off the step time regardless of batch or context size.
        h = h[:SMOOTHNESS_PROBE_ROWS, :SMOOTHNESS_PROBE_LEN].detach().float()
        c = h - h.mean(dim=-2, keepdim=True)
        # Forward-shift energy, which is what "b_t predicts b_{t+1}" literally
        # asks for - and cheaper than the spectrum, since no transform is
        # needed. It is the same quantity smoothness() approximates: the DFT of
        # a first difference scales |X(f)| by 4 sin^2(pi f / N), whose small-f
        # limit is the quadratic weight the grid statistic uses. The 1/6 is
        # derived, not fitted: white noise has spectral second moment 1/3 and
        # difference ratio E|x_t - x_{t+1}|^2 / E|x_t|^2 = 2, so 1/6 puts the
        # two statistics on one scale (checked: 0.333 vs 0.335 on white input).
        denom = c.pow(2).sum()
        if not torch.isfinite(denom) or denom <= 0:
            return
        diff = c[:, 1:] - c[:, :-1]
        target = float((diff.pow(2).sum() / (6.0 * denom)).clamp(0.0, 1.0))
        if not math.isfinite(target):
            return

        prev = self._smooth_target
        self._smooth_target = (
            target
            if prev < 0.0  # first observation seeds the EMA
            else SMOOTHNESS_TARGET_EMA * prev + (1.0 - SMOOTHNESS_TARGET_EMA) * target
        )
        violation = float(self.smoothness()) - self._smooth_target
        if not math.isfinite(violation):
            return
        rho_max = math.log(math.expm1(SMOOTHNESS_LAMBDA_MAX))
        self._smooth_rho = min(
            max(self._smooth_rho + SMOOTHNESS_DUAL_ETA * violation, -20.0), rho_max
        )

    def smooth_lambda(self) -> float:
        """The current multiplier, softplus(rho), bounded by the cap.

        A plain float on purpose: it enters :meth:`aux_loss` as a scalar
        coefficient, so no tensor owned by the controller is ever part of the
        autograd graph.
        """
        rho = self._smooth_rho
        lam = rho + math.log1p(math.exp(-rho)) if rho > 0 else math.log1p(math.exp(rho))
        return min(lam, SMOOTHNESS_LAMBDA_MAX)

    @property
    def smooth_target(self) -> float:
        """The EMA-smoothed measured target, or -1 before the first batch."""
        return self._smooth_target

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Sync the Python-float controller state into its buffers, so the
        checkpoint carries where the multiplier had settled."""
        self._smooth_rho_buf.fill_(self._smooth_rho)
        self._smooth_target_buf.fill_(self._smooth_target)
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Restore the dual state, tolerating checkpoints written before it
        existed. Same reasoning as SMEAR._load_from_state_dict: Lightning loads
        strictly, so an added persistent buffer must seed its own default or it
        makes every earlier checkpoint unloadable."""
        for name in ("_smooth_rho_buf", "_smooth_target_buf"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._smooth_rho = float(self._smooth_rho_buf)
        self._smooth_target = float(self._smooth_target_buf)

    # Cached decode. The model binds the live ``PraxisCache`` here around the
    # head call (modeling._bind_head_cache) and unbinds it after; training and
    # any cache-less forward see None. See ``praxis.heads.base.decode_context``.
    accepts_decode_cache: bool = True
    decode_cache: Any = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        seq_len = hidden_states.shape[-2]
        device = hidden_states.device
        dtype = hidden_states.dtype
        self._update_smoothness_dual(hidden_states)
        offset, state, commit = decode_context(self, hidden_states)
        new_state: dict = {}
        if self.amp_modulation in ("input", "pure"):
            b = self._field_conditional(hidden_states, offset, state, new_state)
        else:
            b = self._field(seq_len, device=device, dtype=dtype, offset=offset)
        if self.fast_weights:
            b = b + self._field_fast(hidden_states, offset, state, new_state)
        if commit is not None:
            commit(new_state)
        return hidden_states * (1.0 + b)

    def _field_conditional(
        self,
        hidden_states: Tensor,
        offset: int = 0,
        state: Optional[dict] = None,
        new_state: Optional[dict] = None,
    ) -> Tensor:
        """Input-conditional field ``[B, seq_len, D]``: the static spectrum with
        an envelope whose coefficients carry a delta from pooled hidden states.
        Zero-init projection means it is identical to the static field at init;
        the learned delta is the structured-variance axis.

        CAUSAL, PER POSITION. The pool used to be the mean over the WHOLE
        window, so at train time every position's field carried a summary of
        the future - a leak, and a train/inference mismatch (generation only
        ever has the prefix). Now position ``t`` pools the causal prefix
        ``h[:, :t+1]`` (inclusive: its own state is not the future) and gets
        its own coefficient set. That is affordable because the envelope
        FACTORIZES across the two grid axes (see :meth:`_env_factors`): the
        field is ``((phase_t * e_t) @ (spec * amps)) * e_d`` per position, two
        ``[B*T, F_t] @ [F_t, F_d]`` matmuls and no per-position grid, so the
        cost is a constant factor over the static field whatever the sequence
        length, and a 64-token window is conditioned as fully as a 512-token
        one.

        Under cached decode the prefix continues from the carried
        ``prefix_sum``/``prefix_count`` and positions from ``offset``, so a
        one-token chunk reads exactly the coefficients it would have read as
        the last position of a full-sequence forward.
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        cum = torch.cumsum(hidden_states.float(), dim=1)  # [B, T, D]
        counts = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
        if state is not None and "prefix_sum" in state:
            cum = cum + state["prefix_sum"].to(device).unsqueeze(1)
            counts = counts + float(state["prefix_count"])
        if new_state is not None:
            new_state["prefix_sum"] = cum[:, -1].detach()
            new_state["prefix_count"] = float(counts[-1])
        p = (cum / counts.view(1, -1, 1)).to(self.amp_basis.dtype)  # [B, T, D]
        if self.amp_modulation == "pure":
            coeffs = self.amp_input(p)  # [B, T, K] - no static base
        else:
            coeffs = self.amp_coeffs + self.amp_input(p)  # [B, T, K]
        # Representative coefficient set for the snapshots: the batch mean of
        # the last position's, the one conditioned on the most context.
        rep = coeffs[:, -1].detach().mean(0)
        self._last_input_coeffs = (
            COEFF_EMA * self._last_input_coeffs + (1.0 - COEFF_EMA) * rep
        )
        e_t, e_d = self._env_factors(coeffs)  # [B, T, F_t], [B, T, F_d]
        cos_a, sin_a = self._phase_table(seq_len, device, offset)  # [T, F_t]
        real = self.spec_real.to(device) * self.amplitudes  # [F_t, F_d]
        imag = self.spec_imag.to(device) * self.amplitudes
        a = (cos_a * e_t) @ real - (sin_a * e_t) @ imag  # [B, T, F_d]
        a = a * e_d
        field = (2.0 / math.sqrt(self.T * self.D)) * (a @ self.basis_d.to(device))
        return field.to(hidden_states.dtype)

    def _build_field(self, amps: Tensor, seq_len: int, device: torch.device) -> Tensor:
        """Batched field from per-example amplitudes ``[B, F_t, F_d]`` -> ``[B,
        seq_len, D]``. The batched twin of :meth:`_field`; the separable
        evaluation broadcasts over the batch dim."""
        phase = torch.complex(self.spec_real.to(device), self.spec_imag.to(device))
        scaled = phase.unsqueeze(0) * amps  # [B, F_t, F_d]
        return self._eval_field(scaled, seq_len, device)

    def _env_factors(self, coeffs: Tensor) -> tuple:
        """Envelope factors from coefficient rows ``[..., K]`` ->
        (``[..., F_t]``, ``[..., F_d]``); the envelope over the grid is their
        outer product.

        The coefficient vector splits into an f_t block and an f_d block; each
        maps through its own complete sine basis into a bounded factor
        ``1 + depth*tanh(.)``, and the grid envelope is the PRODUCT of the
        two. Factorizing (rather than summing the two profiles inside one tanh)
        is what lets the input-conditional path evaluate a distinct envelope at
        every position without ever forming a per-position ``[F_t, F_d]`` grid.
        With the f_d block at zero the f_d factor is identically one and this
        is the old f_t-only envelope, so it is still a strict generalisation and
        identical at init. "pure" drops the base 1 from the f_t factor (the
        field is the conditional delta alone, exactly zero at init) and applies
        the per-f_t-band gain there.
        """
        c_t, c_d = coeffs[..., : self.F_t], coeffs[..., self.F_t :]
        mod_t = AMP_MOD_DEPTH * torch.tanh(c_t @ self.amp_basis.T)
        mod_d = AMP_MOD_DEPTH * torch.tanh(c_d @ self.amp_basis_d.T)
        if self.amp_modulation == "pure":
            e_t = mod_t * self.amp_gain
        else:
            e_t = 1.0 + mod_t
        return e_t, 1.0 + mod_d

    def _env_from_coeffs(self, coeffs: Tensor) -> Tensor:
        """Envelope over the grid from coefficient rows ``[..., K]`` ->
        ``[..., F_t, F_d]``: the outer product of :meth:`_env_factors`."""
        e_t, e_d = self._env_factors(coeffs)
        return e_t.unsqueeze(-1) * e_d.unsqueeze(-2)

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
            amps = amps * env.detach()
        if self.fast_weights:
            amps = amps + self._fast_repr.detach()
        return amps

    def envelope_depth(self) -> float:
        """Peak-to-trough of the f_t envelope; 0 when modulation is off."""
        env = self._envelope()
        return 0.0 if env is None else float((env.max() - env.min()).detach().item())

    def _sample_field(
        self,
        Tp: int,
        coeffs: Optional[Tensor] = None,
        grid_delta: Optional[Tensor] = None,
    ) -> Tensor:
        """Real field [Tp, D] sampled over one period, mean-centered over time.

        Alias-free for Tp >= 2*F_t+1 (the field is band-limited to F_t temporal
        frequencies), and far cheaper than the full-T irfft. Shared by every
        snapshot view below. ``coeffs`` overrides the envelope coefficients (the
        input-conditional set, for the strands snapshot); default = the static
        base envelope. ``grid_delta`` adds the fast-weight overlay on the grid.
        """
        rfft_D = self.D // 2 + 1
        spec = torch.zeros(Tp, rfft_D, dtype=torch.complex64)
        amps = self.amplitudes.detach().cpu()
        if coeffs is not None:
            env = self._env_from_coeffs(coeffs.to(self.amp_basis.device))
            amps = amps * env.detach().cpu()
        else:
            env = self._envelope()
            if env is not None:
                amps = amps * env.detach().cpu()
        if grid_delta is not None:
            amps = amps + grid_delta.detach().cpu()
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
            grid_delta = self._fast_repr if self.fast_weights else None
            static = self._sample_field(Tp)  # [Tp, D] static (bias)
            cond_coeffs = getattr(self, "_last_input_coeffs", None)
            if self.amp_modulation == "pure":
                # No static spectrum reaches the output: the sampled field (plus
                # the fast overlay) IS the conditional delta, so all variance.
                cond = self._sample_field(Tp, grid_delta=grid_delta)
                static = torch.zeros_like(static)
            elif self.amp_modulation == "input" and cond_coeffs is not None:
                cond = self._sample_field(Tp, coeffs=cond_coeffs, grid_delta=grid_delta)
            elif grid_delta is not None:
                cond = self._sample_field(Tp, grid_delta=grid_delta)  # fast = variance
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

    def envelope_split(self) -> dict:
        """How many envelope degrees of freedom the field is actually using.

        :meth:`capacity_split` reads the *energy* the variance axis carries;
        this reads its *dimensionality*, which is the quantity the
        interference-capacity proposition counts. Two numbers, both from the
        coefficient vector currently driving the envelope:

        - ``harmonic_env_fd_share`` - fraction of coefficient energy on the
          feature axis. Zero at init by construction.
        - ``harmonic_env_modes`` - participation ratio ``(sum c^2)^2 / sum c^4``,
          the effective count of active modes. One at init.

        Both are scale-free, so neither moves merely because the envelope grew.
        """
        if self.amp_modulation == "off":
            return {}
        with torch.no_grad():
            coeffs = (
                (
                    self._last_input_coeffs
                    if self.amp_modulation in ("input", "pure")
                    else self.amp_coeffs
                )
                .detach()
                .float()
            )
            c2 = coeffs.pow(2)
            total = c2.sum().clamp_min(1e-12)
            fd_share = c2[self.F_t :].sum() / total
            modes = total.pow(2) / c2.pow(2).sum().clamp_min(1e-24)
            return {
                "harmonic_env_fd_share": float(fd_share.item()),
                "harmonic_env_modes": float(modes.item()),
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
            grid_delta = self._fast_repr if self.fast_weights else None
            static = self._sample_field(Tp)  # [Tp, D]
            cond_coeffs = getattr(self, "_last_input_coeffs", None)
            if self.amp_modulation == "pure":
                cond = self._sample_field(Tp, grid_delta=grid_delta)
                static = torch.zeros_like(static)
            elif self.amp_modulation == "input" and cond_coeffs is not None:
                cond = self._sample_field(Tp, coeffs=cond_coeffs, grid_delta=grid_delta)
            elif grid_delta is not None:
                cond = self._sample_field(Tp, grid_delta=grid_delta)  # fast = variance
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
                amps = amps * env.detach().cpu()
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

        ``lambda`` is the dual variable maintained by
        :meth:`_update_smoothness_dual` and is detached here: the multiplier is
        a coefficient on this term, not something autograd should push around.
        It is only ever moved by the constraint violation.
        """
        if SMOOTHNESS_DUAL_ETA <= 0.0 and SMOOTHNESS_LAMBDA_INIT <= 0.0:
            return None
        return self.smooth_lambda() * self.smoothness()


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
        fast_weights: bool = False,
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
            fast_weights=fast_weights,
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
            "harmonic_smooth_lambda": float(self.field.smooth_lambda()),
            **(
                {"harmonic_smooth_target": float(self.field.smooth_target)}
                if self.field.smooth_target >= 0.0
                else {}
            ),
            **self.field.envelope_split(),
            **self.field.capacity_split(),
        }
        if self.field.fast_weights:
            out["harmonic_fast_norm"] = float(self.field._fast_repr.norm().item())

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
