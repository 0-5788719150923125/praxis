"""ArcSSOG: the Gaussian field, given a depth axis, a warm gate and a real bank.

``SSOGAttention`` next door is the faithful port and stays that way. This is the
variant we hack on. Three deviations, each with a reason that is a measurement
from ``abstractinator-r`` or a theorem, not a preference:

1. THE FIELD IS PER-DEPTH. ``raw_mu``, ``raw_sigma``, ``log_lambda``,
   ``raw_gate`` and the temperature all gain a leading depth axis, indexed by
   ``current_depth``. This is the FAITHFUL direction, not a liberty: the
   reference's field is per-LAYER and its README describes the model opening the
   taps "layer by layer, exactly where steering pays off". A depth-SHARED field
   was ``-r``'s deviation, and it asks one gate scalar to be simultaneously
   right for pass 0 (where steering is noise on top of local structure) and pass
   5 (where it would pay). ``-r`` answered that question with the average, which
   is shut.

   It also decouples REACH from SMEARING, which is the whole game for retrieval.
   A shared field can only reach far by cascading, and h hops give mean lag
   ``h*mu`` with width ``sqrt(h)*sigma`` - ``-r``'s hop 6 lands near lag 51 with
   a width around 29, which returns the haystack. A depth-5 atom with its own
   ``mu = 200, sigma = 3`` is a SHARP far read. Diffuse far reads are not
   retrieval however far they reach.

2. THE GATE STARTS WARM (softplus(-2) ~ 0.13, against the reference's
   softplus(-8) ~ 3e-4). The cold start is doubled in our port: ``steer`` is
   zero-initialised AND gated, and zero-init alone already gives an exactly
   frozen field at step 0. The gate is a second multiplicative barrier on top of
   a sufficient one. ``-r`` measured what that costs - after 11.7k steps the
   probe had learned hard (``steer.weight`` norm 10.9, absmax 3.34) and every
   bit of it was being multiplied by 3.4e-4, while ``raw_gate`` drifted from
   -8.00 to -8.05. The probe was straining against a closed valve. The field is
   still frozen at initialisation here, by zero-init; the gate now scales
   steering rather than blocking it.

3. THE ATOM BANK IS POPULATED (12 atoms over lags 0.5 .. 128, against 4 over
   0.5 .. 32). Two reasons. Zoology/Based establish that recall quality is
   governed by RECURRENT STATE SIZE, and this field's state is what it carries
   forward: ``R`` atoms x ``head_dim``, which at R=4, one head, head_dim 37 is
   148 numbers summarising all history. And R Gaussian atoms on a log ladder ARE
   a projection of the value history onto an R-dimensional basis - the same
   object as HiPPO/LMU, which projects onto orthogonal polynomials precisely so
   history can be RECONSTRUCTED. At R=4 the frame is far too sparse to localise
   anything in lag; a populated ladder is a multi-scale delay line a downstream
   readout can decode a position from. The reference's 4 was a sweet spot on
   32x32 and 224x224 grids with 2D separability, which says nothing about
   512-patch causal text.

WHY ANY OF THIS COULD WORK AT ALL, stated as the theorem rather than a hope.
The Zoology line (Arora, Rudra, Re et al) makes the decisive property
INPUT-DEPENDENCE, not convolution-versus-attention. A convolution whose kernel
does not depend on the input provably cannot do associative recall at any width
or depth. An input-dependent gated convolution can: the lower bound is
Omega(eps log log N) layers, with a construction at poly-log layers and
parameters linear in sequence length. So SSOG with the taps SHUT is a pure
linear convolution and is in the provably-incapable class - which is where
``-r`` has been sitting - while SSOG with the taps open is a gated convolution
and is in the capable one. Recurrent depth is what makes the poly-log-layer
route affordable, since it buys layers without buying parameters. That is the
whole bet, and deviations 1 and 2 exist to get across that class boundary.

NOT DONE HERE, deliberately: the null atom (the field's softmax1), and
unbounded mu-steering read as a pointer. Both are the natural next steps once
the gates are observed to open; neither is worth its complexity while the taps
are shut. ``MAX_OFFSET`` is inherited unchanged at 8 tokens, so steering here
still only nudges - see ``next/`` for the addressing argument.

The atom count is a REGISTRY PROFILE argument, not a config field: behaviour
belongs in the registry entry (``partial(ArcSSOGAttention, num_atoms=...)``),
the way ``arc_dropoff`` carries its ablation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.attention.ssog import (
    _EPS,
    GEOM_BINS,
    MAX_OFFSET,
    MU_INIT_MIN,
    SIGMA_FLOOR,
    SIGMA_MIN_EXCESS,
    SIGMA_Q,
    TEMPERATURE_RAW_INIT,
    SSOGAttention,
    _inv_softplus,
    _log_kernel,
)

ARC_NUM_ATOMS: int = 12  # populated bank (the reference's vision sweet spot is 4)
ARC_GATE_INIT: float = -2.0  # softplus(-2) ~ 0.13: warm, not open, not shut
ARC_MU_INIT_MAX: float = 128.0  # farthest atom at init, in tokens
MAX_DECLARED_DEPTHS: int = 12  # how many per-depth cards the dashboard declares
FAR_LAG: float = 32.0  # "far" atom threshold for the far-mass diagnostic


def _depth_cards(prefix: str, title: str, y_label: str, order: int, text: str) -> dict:
    """One declaration per depth, as one multi-line card.

    Declared for ``MAX_DECLARED_DEPTHS`` regardless of the run's actual depth;
    the dashboard prunes declarations with no data, which is what that pruning
    is for.
    """
    return {
        f"{prefix}_d{d}": {
            "description": text,
            "chart": {
                "title": title,
                "y_label": y_label,
                "group": "ssog_field",
                "group_order": 30,
                "order": order + d,
                "series_group": prefix,
                "series_label": f"depth {d}",
            },
        }
        for d in range(MAX_DECLARED_DEPTHS)
    }


class ArcSSOGAttention(SSOGAttention):
    """SSOG with a per-depth field (see module docstring)."""

    default_atoms: int = ARC_NUM_ATOMS
    gate_init: float = ARC_GATE_INIT
    mu_init_max: float = ARC_MU_INIT_MAX

    metric_descriptions = {
        **_depth_cards(
            "ssog_gate_mu",
            "SSOG Steering Gate: mu (per depth)",
            "softplus(raw gate)",
            10,
            "How far content may move the atoms' CENTRES, at each recurrent "
            "pass. This is the headline of the per-depth field: a shared gate "
            "has to average what six passes want and settles shut, while these "
            "are free to disagree. The reference's per-layer result is that "
            "late layers steer hardest. A late-depth line rising while the "
            "early ones stay flat is the mechanism working; all of them flat "
            "says the field genuinely does not want content in it at this "
            "scale, and that a purely positional prior was the whole story.",
        ),
        **_depth_cards(
            "ssog_gate_sigma",
            "SSOG Steering Gate: sigma (per depth)",
            "softplus(raw gate)",
            30,
            "How far content may move the atoms' WIDTHS, per recurrent pass. "
            "Widening on demand is a focus mechanism: a token that knows it "
            "needs broad context can ask for it. Same reading as the mu gate.",
        ),
        **_depth_cards(
            "ssog_gate_lambda",
            "SSOG Steering Gate: lambda (per depth)",
            "softplus(raw gate)",
            50,
            "How far content may re-weight the MIXTURE, per recurrent pass. "
            "The reference found this one opens hardest and increasingly with "
            "depth. It is also the cheapest to serve: re-weighting a fixed "
            "atom bank per token keeps the field shift-invariant per atom, so "
            "this is the family that survives an FFT-convolution rewrite.",
        ),
        **_depth_cards(
            "ssog_reach",
            "SSOG Reach (lambda-weighted mean lag, per depth)",
            "Lag (tokens)",
            70,
            "The first moment of each pass's actual mixture, sum_r lambda_r * "
            "mu_r. A mean over the ATOM LADDER would be meaningless (with "
            "atoms at 0.5 and 128 it names a lag no atom occupies), but the "
            "first moment of the mixture is a real summary of where that pass "
            "looks. The prediction the per-depth field exists to test: these "
            "should SEPARATE, early passes staying local while late ones walk "
            "outward. Six lines on top of each other is a shared field wearing "
            "a depth axis.",
        ),
        **_depth_cards(
            "ssog_far_mass",
            f"SSOG Far Mass (lambda beyond lag {FAR_LAG:.0f}, per depth)",
            "Share of mixture weight",
            90,
            "The share of each pass's mixture sitting on atoms centred beyond "
            f"lag {FAR_LAG:.0f}. This is the direct answer to whether the "
            "block does anything at range: a populated bank is only worth its "
            "parameters if the far half keeps weight. Decaying to zero says "
            "the model paid for a delay line and then used the first few taps, "
            "and the honest response is to shorten the ladder rather than to "
            "keep claiming reach.",
        ),
        **_depth_cards(
            "ssog_temperature",
            "SSOG Field Temperature (per depth)",
            "tau",
            110,
            "Learned sharpening, floored at 0.5, now per pass. Tempering a "
            "Gaussian keeps it Gaussian with variance scaled by tau, so this "
            "is a width multiplier on that pass's whole field. Driving into "
            "the floor is the cue that a pass is trying to manufacture an "
            "outlet it does not have: there is still no null atom, so a query "
            "wanting to contribute nothing can only approximate it by going "
            "needle-narrow onto a single key.",
        ),
        "ssog_geometry": {
            "description": (
                "The reference's figure, finally drawable. It plots learned "
                "geometry per LAYER; a depth-shared field had no such axis and "
                "the card had to show atoms instead. Here each band IS one "
                "recurrent pass's mixture over lag - bottom band is pass 0, "
                "top is the deepest - on a geometric lag axis from 0.5 to 512 "
                "tokens, each normalised to its own peak. What the reference "
                "reports across layers is early passes becoming convolutions "
                "in disguise, middle passes turning into strip detectors, late "
                "passes going global; that is a diagonal brightening from "
                "bottom-left to top-right. Identical bands mean the depth axis "
                "bought nothing. The axis runs past the sequence length on "
                "purpose, so mass a pass has walked outside its own window "
                "reads as such. Computed from the learned atoms, so the causal "
                "truncation and per-query steering are not in it."
            ),
            "snapshot": {
                "title": "Attention Geometry (per depth)",
                "renderer": "heatmap_2d",
                "color_scale": "linear",
                "group": "ssog_field",
                "order": 200,
            },
        },
        "ssog_cascade": {
            "description": (
                "Where information actually is after h passes, composing the "
                "REAL per-depth kernels in order rather than self-convolving "
                "one shared kernel h times. Gaussians are closed under "
                "convolution and residuals keep every hop count live at once, "
                "so band h is the h-fold composition k_0 * k_1 * ... * "
                "k_(h-1), bottom band one pass, top the full depth, each "
                "normalised to its own peak. The top band is the farthest this "
                "block can see at all. Read it against the geometry card "
                "above: if the per-depth fields separate, the cascade should "
                "march outward FASTER than the sqrt(h) smearing of a shared "
                "field, because a late sharp far atom beats six hops of a near "
                "one. Ignores the causal window and whatever the residual "
                "stream does between passes."
            ),
            "snapshot": {
                "title": "Composed Reach by Depth",
                "renderer": "heatmap_2d",
                "color_scale": "linear",
                "group": "ssog_field",
                "order": 201,
            },
        },
    }

    def __init__(self, config, num_atoms: Optional[int] = None) -> None:
        # Set before super().__init__, because __init__ calls _build_field,
        # which needs it. A plain int assignment is safe before nn.Module's
        # __init__ (only Parameters and Modules are not).
        self.depths = max(1, int(getattr(config, "depth", 1) or 1))
        super().__init__(config, num_atoms=num_atoms)

    # ------------------------------------------------------------------ field
    def _build_field(self, hidden_size: int, H: int, R: int) -> None:
        """The base's ladder, with a leading depth axis on everything learned.

        Every depth starts on the SAME geometric ladder, with independent
        jitter so the passes break symmetry from each other at step 0. A
        depth-staggered init (short ladders early, long ones late) was the
        alternative and is declined: it would seed the very progression the
        card is meant to measure. Each pass gets the whole ladder and picks.
        """
        D = self.depths
        ramp = torch.linspace(0.0, 1.0, R) if R > 1 else torch.zeros(1)
        init_mu = MU_INIT_MIN * (self.mu_init_max / MU_INIT_MIN) ** ramp  # (R,)
        init_mu = init_mu.expand(D, H, R) * torch.exp(0.1 * torch.randn(D, H, R))
        self.raw_mu = nn.Parameter(_inv_softplus(init_mu.clamp_min(0.05)))
        init_sigma = (SIGMA_Q * init_mu - SIGMA_FLOOR).clamp_min(SIGMA_MIN_EXCESS)
        self.raw_sigma = nn.Parameter(_inv_softplus(init_sigma))
        self.log_lambda = nn.Parameter(torch.zeros(D, H, R))
        self.raw_temperature = nn.Parameter(torch.full((D, 1), TEMPERATURE_RAW_INIT))

        # One shared probe, per-depth bias and gate. This is Arc's own pattern -
        # a shared projection that each recurrent pass modulates - and it keeps
        # the depth axis on the cheap parameters rather than on a [D, hidden,
        # H*R*3] stack of projections. The weight stays zero-init, which is what
        # actually freezes the field at step 0; the gate only scales.
        self.steer = nn.Linear(hidden_size, H * R * 3, bias=True)
        nn.init.zeros_(self.steer.weight)
        nn.init.zeros_(self.steer.bias)
        self.steer_bias = nn.Parameter(torch.zeros(D, H * R * 3))
        self.raw_gate = nn.Parameter(torch.full((D, 3), self.gate_init))

    def _field(
        self, x: Tensor, current_depth: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-query atom parameters for THIS recurrent pass."""
        B, T, _ = x.shape
        H, R = self.num_heads, self.num_atoms
        d = min(int(current_depth), self.depths - 1)

        steer = (self.steer(x) + self.steer_bias[d]).float()
        steer = steer.view(B, T, H, R, 3).permute(0, 2, 1, 3, 4)
        gate = F.softplus(self.raw_gate[d].float())  # (3,)
        raw_mu = self.raw_mu[d].float()[None, :, None, :]
        mu = F.softplus(raw_mu + gate[0] * MAX_OFFSET * torch.tanh(steer[..., 0]))
        sigma0 = F.softplus(self.raw_sigma[d].float()) + _EPS + SIGMA_FLOOR
        sigma = sigma0[None, :, None, :] * torch.exp(gate[1] * torch.tanh(steer[..., 1]))
        loglam = torch.log_softmax(
            self.log_lambda[d].float()[None, :, None, :]
            + gate[2] * torch.tanh(steer[..., 2]),
            dim=-1,
        )
        tau = F.softplus(self.raw_temperature[d].float()) + 0.5
        return mu, sigma, loglam, tau

    # --------------------------------------------------------------- geometry
    def _atoms(self, depth: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """One pass's head-averaged ``(mu, sigma, lambda)``, each ``[H, R]``."""
        d = min(int(depth), self.depths - 1)
        mu = F.softplus(self.raw_mu[d].detach().float())
        sigma = F.softplus(self.raw_sigma[d].detach().float()) + _EPS + SIGMA_FLOOR
        lam = torch.softmax(self.log_lambda[d].detach().float(), dim=-1)
        return mu, sigma, lam

    def _mixture(self, depth: int) -> Tensor:
        """That pass's summed mixture on the display grid, ``[GEOM_BINS]``."""
        mu, sigma, lam = self._atoms(depth)
        lags = self.geom_lags.float()
        dens = lam[..., None] * torch.exp(
            _log_kernel(lags, mu[..., None], sigma[..., None])
        )
        return dens.mean(dim=0).sum(dim=0)

    def _kernel(self, depth: int, length: int) -> Tensor:
        """That pass's mixture on the integer lag lattice, summing to 1."""
        mu, sigma, lam = self._atoms(depth)
        lattice = torch.arange(length, dtype=torch.float32, device=mu.device)
        kernel = (
            lam[..., None]
            * torch.exp(_log_kernel(lattice, mu[..., None], sigma[..., None]))
        ).mean(dim=0).sum(dim=0)
        return kernel / kernel.sum().clamp_min(_EPS)

    def _cascade(self) -> Tensor:
        """Composition of the REAL per-depth kernels, one row per hop count.

        The base self-convolves a single shared kernel, which is the best a
        shared field allows. Here the passes differ, so band h is
        ``k_0 * k_1 * ... * k_(h-1)`` - the actual lag distribution of an
        h-hop path. Computed by multiplying in the Fourier domain on a window
        sized to the atoms' own reach, so a far field cannot wrap onto short
        lags.
        """
        reach = max(
            float((mu + 4.0 * sigma).max())
            for mu, sigma, _ in (self._atoms(d) for d in range(self.depths))
        )
        span = max(64.0, self.depths * max(reach, 1.0))
        length = int(min(8192, 2 ** math.ceil(math.log2(span))))

        acc = None
        rows = []
        for d in range(self.depths):
            spectrum = torch.fft.rfft(self._kernel(d, length), n=2 * length)
            acc = spectrum if acc is None else acc * spectrum
            hop = torch.fft.irfft(acc, n=2 * length)[:length].clamp_min(0.0)
            rows.append(self._sample_grid(hop))
        out = torch.stack(rows)
        return out / out.amax(dim=-1, keepdim=True).clamp_min(_EPS)

    def dashboard_snapshots(self) -> dict:
        """Per-depth geometry, and the composed reach."""
        geom = torch.stack([self._mixture(d) for d in range(self.depths)])
        geom = geom / geom.amax(dim=-1, keepdim=True).clamp_min(_EPS)
        cascade = self._cascade()
        lag_range = [float(self.geom_lags[0]), float(self.geom_lags[-1])]
        return {
            "ssog_geometry": {
                "grid": geom.tolist(),
                "grid_rows": int(geom.shape[0]),
                "grid_cols": GEOM_BINS,
                "x_range": lag_range,
                "y_range": [0, self.depths],
                "max_count": float(geom.max().item()),
            },
            "ssog_cascade": {
                "grid": cascade.tolist(),
                "grid_rows": int(cascade.shape[0]),
                "grid_cols": GEOM_BINS,
                "x_range": lag_range,
                "y_range": [1, self.depths],
                "max_count": float(cascade.max().item()),
            },
        }

    # ---------------------------------------------------------------- metrics
    def training_metrics(self) -> dict:
        """Per depth, never averaged over it - that average is what ``-r``
        reported and it hid six passes agreeing to do nothing."""
        out = {}
        for d in range(self.depths):
            gate = F.softplus(self.raw_gate[d].detach().float())
            out[f"ssog_gate_mu_d{d}"] = gate[0].item()
            out[f"ssog_gate_sigma_d{d}"] = gate[1].item()
            out[f"ssog_gate_lambda_d{d}"] = gate[2].item()
            out[f"ssog_temperature_d{d}"] = (
                F.softplus(self.raw_temperature[d].detach().float()) + 0.5
            ).mean().item()

            mu, _, lam = self._atoms(d)
            # First moment of the MIXTURE, which is a real summary, rather than
            # a mean over the ladder, which names a lag no atom occupies.
            out[f"ssog_reach_d{d}"] = (lam * mu).sum(dim=-1).mean().item()
            out[f"ssog_far_mass_d{d}"] = (
                (lam * (mu > FAR_LAG).float()).sum(dim=-1).mean().item()
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, head_dim={self.head_dim}, "
            f"atoms={self.num_atoms}, depths={self.depths}, "
            f"causal={self.causal}, window={self.window_size}"
        )
