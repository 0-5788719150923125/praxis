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

3. THE BANK CAME BACK DOWN, and this entry is a retraction. It shipped as 12
   atoms over 0.5 .. 128 on the argument that Zoology/Based tie recall quality
   to RECURRENT STATE SIZE, so a bigger bank should buy recall, and that R
   Gaussians on a log ladder are a HiPPO/LMU-style basis projection of history
   which at R=4 is too sparse to localise anything in lag.

   The state-size half of that does not transfer, and I pushed it too hard.
   Their state is a matrix holding key-value BINDINGS; these atoms are fixed
   positional filters holding none. More atoms buys a better basis for reading
   POSITION, not associative capacity.

   What it did buy was dilution. Attention weights are the mixture NORMALISED
   over the causal keys, so every atom added takes a share from every other:
   0.083 per atom against 0.25, cutting the previous-token atom to a third of
   its mass. Worse, three of the twelve sat centred beyond lag 32 and two
   beyond 64, so at the x1 curriculum tier their mass was truncated and
   renormalised onto the OLDEST tokens - a sink, not a long-range read, and the
   softmax has no way to decline. Measured over 11.8k steps on -r: ``far_mass``
   decayed at every depth while ``reach`` stayed flat, i.e. the model spent
   training clawing back toward concentration. (That reading used the old
   centre-indicator ``far_mass``. It survives the change to the tail integral -
   the wide bank put five atoms well past 32, so the indicator had real
   resolution there - but the numbers themselves do not compare.) With the faithful 0.5 .. 32
   ladder the same knob had gone the other way, lambda moving TOWARD the far
   atoms.

   The mechanism is dilution, NOT interference. The mixture is a sum of
   non-negative densities and cannot cancel; twelve atoms can represent
   anything four can by zeroing eight weights. An initialisation cost, not a
   capacity one.

   Default is back to the faithful 4 over 0.5 .. 32, which also puts every atom
   inside the window at every tier. The 12/128 configuration is preserved as
   the ``arc_ssog_wide`` registry profile so the measurement can be repeated.

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
unbounded mu-steering read as a pointer. ``MAX_OFFSET`` is inherited unchanged
at 8 tokens, so steering here still only nudges - see ``next/`` for the
addressing argument.

On the null atom specifically, since it is the obvious response to the dilution
above and is NOT one: a learned null logit scales every real weight by the
common factor ``Z / (Z + exp(l_0))``, so each real atom keeps exactly its 1/R
share of what is left. What it would fix is the out-of-window sink - and the
ladder coming back to 0.5 .. 32 already removes that, since no atom sits
outside the window at any tier. The field's own cue for a missing outlet is a
per-depth temperature driving toward its 0.5 floor; on -r it did not, sitting
at 0.76 - 0.89 against an init of 0.813 for 11.8k steps. Add it when that cue
fires, and as a LEARNED lambda_0: a fixed logit-0 ghost would take 40-60% of a
Gaussian field's mass, because a Gaussian log-density integrates to only ~1-2
over lags.

Bank size and ladder span are ONE decision and both are REGISTRY PROFILE
arguments, not config fields: behaviour belongs in the registry entry
(``partial(ArcSSOGAttention, num_atoms=..., mu_init_max=...)``), the way
``arc_dropoff`` carries its ablation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.attention.ssog import (
    _EPS,
    NULL_LOGIT_INIT,
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

# Four atoms over 0.5 .. 32, which is the faithful port's ladder. A twelve-atom
# bank over 0.5 .. 128 was tried and is kept as the `arc_ssog_wide` profile; see
# THE BANK CAME BACK DOWN in the class docstring for what it measured.
ARC_NUM_ATOMS: int = 4
ARC_GATE_INIT: float = -2.0  # softplus(-2) ~ 0.13: warm, not open, not shut
ARC_MU_INIT_MAX: float = 32.0  # farthest atom at init, in tokens
MAX_DECLARED_DEPTHS: int = 12  # how many per-depth cards the dashboard declares
# "Far" threshold for the far-mass diagnostic, in tokens. It deliberately
# equals ARC_MU_INIT_MAX - the question the card asks is whether the bank keeps
# weight BEYOND the ladder it was handed. That coincidence was fatal while the
# metric was an indicator on mu; `_tail_mass` integrates instead, so an atom
# parked exactly on the threshold now reports ~0.5 rather than 0 or 1.
FAR_LAG: float = 32.0


def _tail_mass(mu: Tensor, sigma: Tensor, lag: float) -> Tensor:
    """Share of each atom's OWN density sitting beyond ``lag``: P(x > lag).

    What "far mass" was always meant to be. The predecessor asked
    ``mu > FAR_LAG`` - an indicator on the CENTRE - and that is a coin flip
    precisely here, because ``FAR_LAG`` is the same 32.0 as the ladder's top
    rung and ``_build_field`` jitters every atom by ``exp(0.1 * randn)``. With
    ``num_heads=1`` there is no head average to smooth the flip out, so on the
    -s run depths 2 and 4 reported a hard 0.000 for 14k steps while holding
    25-31% of their mixture on an atom at lag 27-31 with a sigma near 10 - the
    card arguing against reach at the two depths that had the most of it.

    The integral has no such edge: an atom centred at 31 with sigma 10 reports
    just under half its weight as far, which is true, and the metric moves
    continuously as mu and sigma do. NUMBERS FROM BEFORE THIS CHANGE ARE NOT
    COMPARABLE - the indicator over-reported a wide far atom (all of its mass,
    including the half inside the threshold) and under-reported a near one at
    zero.

    This measures the FIELD, not the post-softmax weight: the Gaussian has mass
    at negative lag that causality removes and the softmax renormalises away.
    ``ssog_reach`` reads the same way, and the pair only answers where the bank
    is POINTED.
    """
    return 0.5 * torch.erfc((lag - mu) / (sigma * math.sqrt(2.0)))


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
            f"SSOG Far Mass (mixture density beyond lag {FAR_LAG:.0f}, per depth)",
            "Share of mixture weight",
            90,
            "The share of each pass's mixture density sitting beyond lag "
            f"{FAR_LAG:.0f} - sum_r lambda_r * P(x > {FAR_LAG:.0f}) under atom "
            "r's own Gaussian, so a wide atom parked on the threshold "
            "contributes the half of itself that is actually far. This is the "
            "direct answer to whether the block does anything at range: a "
            "populated bank is only worth its parameters if the far half keeps "
            "weight. Decaying to zero says the model paid for a delay line and "
            "then used the first few taps, and the honest response is to "
            "shorten the ladder rather than to keep claiming reach. It read "
            "the CENTRE (mu > threshold) until 2026-08-21, which made it a "
            "coin flip at the ladder's top rung; earlier values do not compare.",
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
        **_depth_cards(
            "ssog_null_share",
            "SSOG Null Share (mass that goes nowhere, per depth)",
            "Share of attention mass",
            130,
            "The fraction of each pass's attention the null atom absorbs - "
            "measured, not asked for. This is the card the null atom exists to "
            "produce, and it should be strongly POSITION-DEPENDENT even though "
            "nothing below the head knows absolute position: a query near the "
            "start has almost no density on the few lags available to it, so a "
            "constant learned logit takes most of its mass, while deep in the "
            "sequence the same atom sits in-window and the null takes almost "
            "none. Measured at init: 0.30 at the first query against 0.04 at "
            "position 127. Rising means a pass is learning to abstain; pinned "
            "near zero means it never wanted the outlet.",
        ),
        **_depth_cards(
            "ssog_null_logit",
            "SSOG Null Logit (what the head asked for, per depth)",
            "l_0",
            150,
            "The learned null logit itself, against an init of -4.0 (~5% of the "
            "mass). Read it with the share card: the logit is what the pass "
            "asked for, the share is what the softmax gave it, and the gap is "
            "the position-dependent part. Driving strongly negative means the "
            "pass is closing an outlet it was handed. This is the APPLIED "
            "value, EMA'd in the forward - a modular-SMEAR router targets "
            "raw_null, so the base parameter is not what runs, and until "
            "2026-08-21 the card plotted the base while the share beside it "
            "came from the merge.",
        ),
        "ssog_geometry": {
            "description": (
                "One band per recurrent pass: its learned mixture over lag (geometric "
                "axis, 0.5 to 512 tokens), normalized to its peak. Identical bands = "
                "depth bought nothing."
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
                "Where mass sits after h passes: band h composes the real per-depth "
                "kernels k_0..k_(h-1) in order. The top band is the farthest this "
                "block can reach."
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

    def __init__(
        self,
        config,
        num_atoms: Optional[int] = None,
        mu_init_max: Optional[float] = None,
        null_atom: bool = False,
    ) -> None:
        # Set before super().__init__, because __init__ calls _build_field,
        # which needs it. A plain int assignment is safe before nn.Module's
        # __init__ (only Parameters and Modules are not).
        self.depths = max(1, int(getattr(config, "depth", 1) or 1))
        super().__init__(
            config,
            num_atoms=num_atoms,
            mu_init_max=mu_init_max,
            null_atom=null_atom,
        )

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
        self._build_null(H)

    def _null_slots(self) -> int:
        return self.depths

    def _build_null(self, H: int) -> None:
        """Per-depth null logit: a pass that wants to abstain often is a
        different pass from one that never does, and a shared scalar would
        average them the way the shared steering gate did."""
        self.raw_null = (
            nn.Parameter(torch.full((self.depths, H), NULL_LOGIT_INIT))
            if self.null_atom
            else None
        )
        # Buffers from the base, never a local copy - `_null_slots` already
        # says `self.depths`, and the one time this method registered its own
        # the logit half of the card drifted onto a different tensor from the
        # share half. See SSOGAttention._register_null_buffers.
        self._register_null_buffers()

    def _null_logit(self, current_depth: int) -> Tensor:
        return self.raw_null[min(int(current_depth), self.depths - 1)].float()

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
        sigma = sigma0[None, :, None, :] * torch.exp(
            gate[1] * torch.tanh(steer[..., 1])
        )
        loglam = torch.log_softmax(
            self.log_lambda[d].float()[None, :, None, :]
            + gate[2] * torch.tanh(steer[..., 2]),
            dim=-1,
        )
        tau = F.softplus(self.raw_temperature[d].float()) + 0.5
        return mu, sigma, loglam, tau

    # --------------------------------------------------------------- geometry
    def _atoms(self, depth: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """One pass's head-averaged ``(mu, sigma, lambda)``, each ``[H, R]``, ON CPU.

        Host copy first, for the reason in ``SSOGAttention._atoms``: everything
        downstream of this runs in the snapshot producer's BACKGROUND THREAD,
        and GPU work there can block on the training stream forever. This
        variant made that far more likely than the base did - it walks every
        depth, so one cascade issued six times the device calls plus a
        ``float()`` sync per depth to size the FFT window.
        """
        d = min(int(depth), self.depths - 1)
        mu = F.softplus(self.raw_mu[d].detach().float().cpu())
        sigma = (
            F.softplus(self.raw_sigma[d].detach().float().cpu()) + _EPS + SIGMA_FLOOR
        )
        lam = torch.softmax(self.log_lambda[d].detach().float().cpu(), dim=-1)
        return mu, sigma, lam

    def _mixture(self, depth: int) -> Tensor:
        """That pass's summed mixture on the display grid, ``[GEOM_BINS]``."""
        mu, sigma, lam = self._atoms(depth)
        lags = self.geom_lags.float().cpu()
        dens = lam[..., None] * torch.exp(
            _log_kernel(lags, mu[..., None], sigma[..., None])
        )
        return dens.mean(dim=0).sum(dim=0)

    def _kernel(self, depth: int, length: int) -> Tensor:
        """That pass's mixture on the integer lag lattice, summing to 1."""
        mu, sigma, lam = self._atoms(depth)
        lattice = torch.arange(length, dtype=torch.float32, device=mu.device)
        kernel = (
            (
                lam[..., None]
                * torch.exp(_log_kernel(lattice, mu[..., None], sigma[..., None]))
            )
            .mean(dim=0)
            .sum(dim=0)
        )
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
        # All CPU now, so this is arithmetic rather than six device syncs.
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

    def _build_snapshots(self) -> dict:
        """Per-depth geometry, and the composed reach.

        Built on the TRAINING thread and stashed; the producer thread only
        reads the stash. See ``SSOGAttention.dashboard_snapshots``.
        """
        geom = torch.stack([self._mixture(d) for d in range(self.depths)])
        geom = geom / geom.amax(dim=-1, keepdim=True).clamp_min(_EPS)
        cascade = self._cascade()
        lags = self.geom_lags.detach().float().cpu()
        lag_range = [float(lags[0]), float(lags[-1])]
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
                (F.softplus(self.raw_temperature[d].detach().float()) + 0.5)
                .mean()
                .item()
            )

            mu, sigma, lam = self._atoms(d)
            # First moment of the MIXTURE, which is a real summary, rather than
            # a mean over the ladder, which names a lag no atom occupies.
            out[f"ssog_reach_d{d}"] = (lam * mu).sum(dim=-1).mean().item()
            out[f"ssog_far_mass_d{d}"] = (
                (lam * _tail_mass(mu, sigma, FAR_LAG)).sum(dim=-1).mean().item()
            )
            if self.raw_null is not None:
                # The APPLIED logit, EMA'd in the forward. `self.raw_null` is
                # the pre-merge base and is not what ran; see
                # SSOGAttention._note_null_logit.
                out[f"ssog_null_logit_d{d}"] = self.null_logit_seen[d].item()
                out[f"ssog_null_share_d{d}"] = self.null_share[d].item()
        self._snapshot_cache = self._build_snapshots()
        return out

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, head_dim={self.head_dim}, "
            f"atoms={self.num_atoms}, depths={self.depths}, "
            f"causal={self.causal}, window={self.window_size}"
        )
