"""Kaleidoscope attention: frozen mixing geometries, turned by a router.

``N`` full ``[T, T]`` mixing matrices are drawn once at construction and never
trained. What the model learns is which combination of them to look through at
each token and each recurrent pass. There are no Q/K projections - nothing is
computed by pairwise comparison - so the score half costs ``N * T^2`` instead of
``T^2 * d``.

    turn      w(x_i) = beta_d + m * tanh(W_turn x_i)     free, signed, per token
    facets    M_k^(d) = A_k + s * tanh(u_{d,k} (x) v_{d,k})
    scores    S[i, j] = sum_k w_k(x_i) * M_k^(d)[i, j]
    O         = ghostmax(mask(S)) @ dropoff(V)
    out       W_o (gamma * O),   gamma = silu(W_gamma x)

Mirrors are functions on the unit square in RELATIVE position, stored at a
canonical ``[R, R]`` and resampled to the live ``[T, T]``, so the geometry is
length-free across a sequence curriculum.

One head, following ``arc_single``: Mega (arXiv:2209.10655) Thm 1 says a SiLU
elementwise gate on a single head spans what multi-head spanned. SiLU rather
than sigmoid so the gate can amplify and flip sign, not only attenuate;
``kaleido_gate_negative`` reports whether that freedom is used.

Design points worth knowing:

- Mixing happens BEFORE the softmax. Blending distributions can only
  interpolate inside their hull; blending logits is log-linear pooling, which
  can put mass where two mirrors agree and nowhere else.
- The per-depth bias deforms the MIRRORS, not the inputs, and per mirror. A bias
  added to every mirror alike factors back out of a weighted sum. Rank-1 per
  (depth, mirror) keeps this at ``D * N * 2T`` parameters.
- The blend is free and signed, not a simplex. A softmax confines it to the
  convex hull and pressures toward one-hot, where the blend is exactly one
  frozen mirror (Synthesizer's Fixed Random, the known-worse variant). A
  negative weight subtracts a mirror, which no mixture can reach.
- Base plus deviation, SMEAR's form: ``beta_d`` is the unbounded per-depth
  preference, ``m * tanh(W_turn x)`` the bounded per-token deviation. Both
  zero-init, so step 0 is uniform attention over the causal prefix.
- Ghostmax is on, applied without materializing the column via
  ``sigmoid(logsumexp)``. Unit-scale logits keep the ghost's share small except
  near position 0, which is exactly where "nothing back there" is the right
  answer. Read ``kaleido_ghost_share`` against sequence length.
- Mirror dropout keeps the blend honest; dropping every mirror falls back to
  uniform attention exactly.

Relation to prior work: Synthesizer (arXiv:2005.00743) covered fixed-random,
trained, and mixture-of-synthesizers token mixing, but its mixture weights are
static learned scalars per head and layer. Input-conditional blending is the
cell left open, and their static mixture reached vanilla-Transformer parity, so
that is the bar.

Watch ``kaleido_turn_modes`` (effective mirrors in the blend) before perplexity;
at 1 this has silently become Fixed Random per token.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Mirrors per kaleidoscope. Four is Synthesizer's mixture scale and SMEAR's
# expert count in this repo's configs. This is the whole dictionary: the block
# runs one head, so there is no per-head multiple on it.
NUM_MIRRORS: int = 4
# Side of the canonical mirror grid. Mirrors are functions on the unit square in
# RELATIVE position, bilinearly resampled from `[R, R]` to the live `[T, T]`
# every forward, so a mirror is length-free - no span, nothing to slice.
#
# Under a sequence curriculum T changes every batch. Ratio indexing hands the
# model the SAME geometry resampled where absolute indexing would hand it a
# different corner slice at each T. Same idea as HarmonicField evaluating
# `_phase_table` rather than storing a `[T, D]` table.
#
# The cost is positional acuity: ratio structure survives resampling exactly,
# but a canonical lag-1 band lands at lag 2 width 5 at T=128 and lag 5 width 21
# at T=512. R trades length-invariance against acuity. See MIRROR_COORDS, where
# "split" reads half the dictionary on a log lag axis and recovers single-token
# acuity at any length; this constant governs only the ratio half.
MIRROR_RES: int = 64
# Logit scale of a fresh mirror. Softmax over t keys with iid N(0, 1) logits
# puts its peak roughly sqrt(2 ln t) above the mean - about 3.5 at t = 512 - so
# a unit-scale mirror is sparse without being one-hot. Flatter than this and
# every mirror is the prefix mean; sharper and each is a single random key.
MIRROR_SCALE: float = 1.0
# Radial envelope over the dictionary: mirror ``k`` (1-indexed) is scaled by
# ``k^-alpha``, the pink-noise prior HarmonicField puts on its frequency grid.
# Zero here, so the base variant has a flat dictionary.
#
# alpha=0 is the corner of the paper's interference-capacity argument
# (research/framing.tex, proposition (iii)): with no envelope, amplitude buys
# capacity directly; with one, the prior costs capacity unless the blend fights
# it. ``kaleido_envelope_fight`` is the measurement.
#
# Caveat: HarmonicField's envelope is indexed by FREQUENCY, so suppressing high
# f is a smoothness prior. The mirrors are iid and carry no frequency ordering,
# so this is a capacity-allocation prior on an arbitrary ordering. Making
# mirror k band-limited to spatial frequency ~k would make the ordering real.
MIRROR_ALPHA: float = 0.0
# Which coordinate system each mirror is read in. A mirror is a function on the
# unit square; this says what the square's axes MEAN.
#
#   "ratio" - (query fraction, key fraction). Ratio structure survives
#       resampling exactly; fixed lag does not.
#
#   "split" - half the dictionary in ratio coordinates, half in (query
#       fraction, WARPED LAG), the lag axis sampled at
#       ``log1p(i - j) / log1p(T - 1)``. The log warp gives every small lag its
#       own canonical cell at any length: at T=512, R=64, lags 0/1/2/3 land on
#       columns 0/7/11/14 where a uniform grid puts them at 0/0.1/0.2/0.4.
#
# Both rather than a warp, because a lag-warped mirror loses ratios exactly as a
# ratio mirror loses lags. Splitting lets the router reach either - the same
# argument the block makes for blending geometries, one level up.
#
# This fixes acuity only. Selection stays structural: a smooth router over a
# frozen dictionary emits a smooth attention row however well the geometry
# resolves position, and induction wants a discrete pointer.
MIRROR_COORDS: str = "ratio"
# Fixed seed for the dictionary. The mirrors are never trained, so they are
# reproducible constants of the architecture rather than learned state: they
# are generated deterministically here and registered non-persistent, exactly
# as HarmonicField does with its Weyl-phase spectrum. This keeps N * T^2 floats
# out of every checkpoint, at the cost of requiring that this constant and
# MIRROR_SCALE never change under an existing run.
MIRROR_SEED: int = 0x5CA1AB1E
# Per-cell cap on the per-depth deformation, as a fraction of a mirror's own
# scale. The frozen dictionary is meant to stay foundational and the facets
# secondary; the same 0.25 the harmonic head uses to bound its fast-weight
# overlay against the slow grid.
FACET_SCALE: float = 0.25
# Second factor of the rank-1 facet. `u` is zero-initialised so the deformation
# is exactly zero at step 0 and the model starts as the pure frozen mixture;
# `v` must be non-zero or `u` receives no gradient (d/du of u (x) v is v). Same
# asymmetry as HarmonicField's fast_u / fast_v.
FACET_V_STD: float = 0.02
# Peak per-token deviation of a blend weight, tanh-bounded. The static blend is
# free and unbounded (it is the analogue of HarmonicField.amplitudes); this
# bounds only the input-conditional half, the way AMP_MOD_DEPTH bounds the
# harmonic envelope, so the per-token term cannot run away with the effective
# logit scale - ``S ~ N(0, ||w||^2)`` at unit mirrors, so ||w|| IS the softmax
# temperature and an unbounded one would sharpen attention to a single key.
TURN_MOD: float = 0.5

# Per-mirror spatial zoom. A mirror at zoom z is read over a grid folded z times
# across the sequence, so its features are z times finer in POSITION while the
# canonical dictionary is untouched. This is granularity, not amplitude: a
# zoomed mirror is a different function, so no router can absorb it. It is also
# a multi-scale dictionary that keeps scale-equivariance - a zoomed ratio mirror
# is still a ratio mirror, so a periodic feature stays periodic at every length.
#
# The ladder is derived: it steps outward from the identity through the harmonic
# series and its sub-harmonics (... 1/3, 1/2, 1, 2, 3 ...), the same spacing
# HarmonicField uses on its frequency grid. That ties granularity to N, so a
# wider dictionary buys finer AND coarser rungs rather than more draws from one
# distribution - which is the measured problem, `kaleido_turn_modes` settling
# near 2.4 of 4 and falling as sequences lengthen.
#
# Both directions. z > 1 sharpens; z < 1 reads a sub-region stretched across the
# sequence and is genuinely coarser (mean adjacent row delta 0.474 -> 0.222 ->
# 0.095 for z = 1 -> 1/2 -> 1/4 at T=129). Factors must be positive and nonzero:
# the triangle fold is odd, so a negative factor is the same mirror reversed at
# the same granularity, and z = 0 is constant across the row and therefore
# invisible to softmax.
#
# The fold is a reflection, not a wrap, so the facets still see a smooth grid.
# Aliasing binds from z = 4 up at R=64, T=257 - the finest rungs carry less than
# the ladder implies at short sequences.


def zoom_ladder(n: int) -> Tuple[float, ...]:
    """Zoom factors for ``n`` mirrors, stepping outward from the identity.

    Alternates harmonic and sub-harmonic - ``1, 2, 1/2, 3, 1/3 ...`` - then
    returns them in ascending order, so the group is centred on ``z = 1`` (the
    plain ratio mirror this block has always used) and reaches equally into
    coarser and finer geometry. ``n=5`` gives ``1/3, 1/2, 1, 2, 3``.
    """
    out = [1.0]
    k = 2
    while len(out) < max(0, n):
        out.append(float(k))
        if len(out) < n:
            out.append(1.0 / k)
        k += 1
    return tuple(sorted(out[: max(0, n)]))


# Probability of dropping a mirror from a blend during training. SMEAR's own
# load-balancing mechanism at SMEAR's own rate; see praxis/routers/smear.py,
# where its absence let "every one of the twelve targets duly saturate to near
# one-hot" on abstractinator-m. Dropping all N is safe: w becomes zero and the
# score falls back to uniform attention exactly.
MIRROR_DROPOUT: float = 0.1
_EPS: float = 1e-9


class KaleidoscopeAttention(nn.Module):
    """N frozen ``[T, T]`` mixing matrices, blended per token by a router."""

    # This block already routes its own parameters PER TOKEN, which is the
    # structural exclusion praxis/routers/targeting.py names. Without the flag
    # SMEAR wraps the turn in a per-EXAMPLE MergedLinear - a coarser router
    # around a finer one - and `kaleido_turn_modes` would then measure SMEAR's
    # router rather than this one.
    #
    # The flag covers the whole subtree, so `value`, `gate` and `output` are
    # excluded too. Recovering them would mean splitting the geometry machinery
    # into an opaque submodule, changing parameter qualnames.
    MERGE_OPAQUE: bool = True

    num_mirrors: int = NUM_MIRRORS
    resolution: int = MIRROR_RES
    alpha: float = MIRROR_ALPHA
    coords: str = MIRROR_COORDS

    metric_descriptions = {
        "kaleido_turn_modes": {
            "description": (
                "Effective number of mirrors in the blend (participation ratio). 1 = "
                "collapsed onto a single frozen matrix; N = all in play. Absent at "
                "init."
            ),
            "chart": {
                "title": "Kaleidoscope Turn",
                "y_label": "effective mirrors / share",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "group_order": 30,
                "order": 10,
                "series_group": "kaleido_turn",
                "series_label": "effective mirrors",
            },
        },
        "kaleido_turn_negative": {
            "description": (
                "Fraction of blend weights below zero - what a softmax blend cannot "
                "produce. Pinned near 0 = the free parameterization bought nothing."
            ),
            "chart": {
                "title": "Kaleidoscope Turn",
                "y_label": "effective mirrors / share",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 12,
                "series_group": "kaleido_turn",
                "series_label": "negative share",
            },
        },
        "kaleido_turn_scale": {
            "description": (
                "Mean ||w|| across mirrors, which is the effective softmax "
                "temperature. Unbounded growth sharpens attention onto one key. 0 at "
                "init."
            ),
            "chart": {
                "title": "Kaleidoscope Turn Scale",
                "y_label": "||w||",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 14,
            },
        },
        "kaleido_turn_static_share": {
            "description": (
                "Static share of the blend, on the axis that drives the softmax. 1.0 = "
                "a learned constant per depth with the input ignored; 0.0 = purely "
                "input-driven."
            ),
            "chart": {
                "title": "Kaleidoscope Turn",
                "y_label": "normalized I(input; mirror)",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 25,
                "series_group": "kaleido_turn",
                "series_label": "static share",
            },
        },
        "kaleido_turn_depth_specialization": {
            "description": (
                "Between-depth variance of the static blend. 0 = every pass learned "
                "the same preference over mirrors (also the init value); rising = they "
                "diverge."
            ),
            "chart": {
                "title": "Kaleidoscope Depth Specialization",
                "y_label": "depth-specific fraction",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 31,
                "series_group": "kaleido_depth_spec",
                "series_label": "turn (which mirrors)",
            },
        },
        "kaleido_mirror_utilization": {
            "description": (
                "Fraction of mirrors whose |weight| clears half the mean magnitude. "
                "1/N = collapse onto one; near N argues for a larger dictionary."
            ),
            "chart": {
                "title": "Kaleidoscope Mirror Utilization",
                "y_label": "fraction above half fair share",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 15,
            },
        },
        "kaleido_envelope_fight": {
            "description": (
                "Log-slope of |w_k| against log k, normalized by the dictionary's "
                "1/k^alpha envelope: 0 = accepting it, 1.0 = cancelling it exactly. "
                "Absent when alpha=0."
            ),
            "chart": {
                "title": "Kaleidoscope Envelope Fight",
                "y_label": "d log|w| / d log k, over alpha",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 16,
            },
        },
        "kaleido_lag_share": {
            "description": (
                "Share of blend magnitude on the lag-coordinate (log-warped) half of "
                "the dictionary. 0.5 is parity; below it the warp is not paying for "
                "itself."
            ),
            "chart": {
                "title": "Kaleidoscope Lag Share",
                "y_label": "share of |w| on lag mirrors",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 17,
            },
        },
        "kaleido_zoom_mean": {
            "description": (
                "Blend-magnitude-weighted position on the zoom ladder, in log space. "
                "0 = all mass on the coarsest mirror, 1 = all on the finest. "
                "Comparable across dictionary sizes."
            ),
            "chart": {
                "title": "Kaleidoscope Zoom",
                "y_label": "mean octaves above coarsest",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 18,
            },
        },
        "kaleido_gate_negative": {
            "description": (
                "Fraction of SiLU gate values below zero - the sign flips a sigmoid "
                "gate cannot make. Pinned at 0 = a sigmoid would have done as well."
            ),
            "chart": {
                "title": "Kaleidoscope Gate",
                "y_label": "Value",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 50,
                "series_group": "kaleido_gate",
                "series_label": "negative fraction",
            },
        },
        "kaleido_gate_magnitude": {
            "description": (
                "Mean absolute SiLU gate value. Near 0 = the attention branch is "
                "closed off; read it against the negative fraction."
            ),
            "chart": {
                "title": "Kaleidoscope Gate",
                "y_label": "Value",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 51,
                "series_group": "kaleido_gate",
                "series_label": "mean magnitude",
            },
        },
        "kaleido_ghost_share": {
            "description": (
                "Share of attention mass on ghostmax's zero logit, the 'attend to "
                "nothing' escape. Length-dependent by construction, so compare like "
                "lengths."
            ),
            "chart": {
                "title": "Kaleidoscope Ghost Share",
                "y_label": "share of mass",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 45,
            },
        },
        "kaleido_facet_depth_specialization": {
            "description": (
                "Fraction of the facet deformation that is depth-specific. 0 = every "
                "pass ground its mirrors the same way; rising = each pass reshapes "
                "differently."
            ),
            "chart": {
                "title": "Kaleidoscope Depth Specialization",
                "y_label": "depth-specific fraction",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 30,
                "series_group": "kaleido_depth_spec",
                "series_label": "facets (how ground)",
            },
        },
        "kaleido_facet_strength": {
            "description": (
                "Mean |deformation| as a fraction of its cap. 1.0 = facets pinned at "
                "FACET_SCALE. 0 at init, and staying near 0 is a finding, not a "
                "failure."
            ),
            "chart": {
                "title": "Kaleidoscope Facet Strength",
                "y_label": "mean |delta| / cap",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 40,
            },
        },
    }

    def __init__(
        self,
        config,
        num_mirrors: Optional[int] = None,
        resolution: Optional[int] = None,
        alpha: Optional[float] = None,
        coords: Optional[str] = None,
        dropoff: Optional[str] = None,
        dropoff_every: bool = False,
        mix_norm: bool = False,
        zoom=None,
    ) -> None:
        super().__init__()
        self.patch_config(config)
        hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = (
            getattr(config, "head_size", None) or hidden_size // self.num_heads
        )
        self.num_mirrors = int(num_mirrors or type(self).num_mirrors)
        self.causal = config.causal
        self.window_size = getattr(config, "window_size", None)
        self.depths = max(1, int(getattr(config, "depth", 1) or 1))
        self.pos_type = "kaleido"

        # Dropoff ablation (next/dropoff.md). Same schedule options as
        # CausalAttention - see its __init__ for why the two exist and why
        # neither is measured.
        self.dropoff_mode = dropoff
        self.dropoff_every = bool(dropoff_every)
        if dropoff is None:
            self.dropoff_step = None
        else:
            layers = max(1, int(getattr(config, "num_layers", 1) or 1))
            self.dropoff_step = max(0, self.depths - layers)

        # 1/sqrt(N) on the mix, for the reason attention divides q.k by
        # sqrt(d): `scores` sums N mirror terms, so logit variance grows with N
        # and a wider dictionary opens SHARPER rather than richer (measured
        # support 181 -> 65 -> 25 positions at N = 4 -> 12 -> 24).
        #
        # OFF by default: a global constant is absorbed by `turn_static` but not
        # by the tanh-bounded conditional half, so it also divides the
        # per-token modulation ceiling. Set it for both arms of an N sweep or
        # neither.
        self.mix_norm = bool(mix_norm)
        self.mix_scale = self.num_mirrors**-0.5 if self.mix_norm else 1.0

        self.resolution = int(resolution or type(self).resolution)
        self.alpha = float(type(self).alpha if alpha is None else alpha)
        self.coords = str(coords or type(self).coords)
        if self.coords not in ("ratio", "split"):
            raise ValueError(f"unknown mirror coordinates: {self.coords!r}")

        # Per-mirror zoom over the RATIO group. `True` derives the harmonic
        # ladder from the group size; an explicit sequence is used verbatim (and
        # cycled if short), which is what an ablation on the spacing would pass.
        # None/False is off and leaves the fast `interpolate` path untouched.
        self.zoom_spec = zoom

        N, H = self.num_mirrors, self.num_heads
        # Contiguous groups, ratio first, so the resample can slice rather than
        # gather. "split" gives the lag half the smaller share when N is odd:
        # ratio is the coordinate system the block already worked in.
        self.n_lag = N // 2 if self.coords == "split" else 0
        self.n_ratio = N - self.n_lag

        # One factor per ratio mirror. The derived ladder spans the group; an
        # explicit one cycles, so a short spec still spreads across it evenly.
        if self.zoom_spec is True:
            spec = zoom_ladder(self.n_ratio)
        elif self.zoom_spec:
            spec = tuple(float(z) for z in self.zoom_spec)
        else:
            spec = ()
        if any(z <= 0 for z in spec):
            # Not an arbitrary restriction: the fold is odd, so a negative
            # factor is the same mirror reversed at the same granularity, and
            # zero is constant across the row and therefore invisible to
            # softmax. Coarser is |z| < 1. See `zoom_ladder`.
            raise ValueError(
                f"zoom factors must be positive; use z < 1 for coarser, "
                f"not z <= 0: {spec!r}"
            )
        self.zoom = spec
        if spec:
            factors = [spec[k % len(spec)] for k in range(self.n_ratio)]
        else:
            factors = [1.0] * self.n_ratio
        self.register_buffer(
            "zoom_factor", torch.tensor(factors, dtype=torch.float32), persistent=False
        )

        # The dictionary. Shared across heads on purpose: one frozen basis that
        # every head reads differently is both cheaper (N * T^2 rather than
        # H * N * T^2) and the more interesting claim - the heads differ by how
        # they turn, not by owning private geometry.
        gen = torch.Generator().manual_seed(MIRROR_SEED)
        R = self.resolution
        # The envelope multiplies the dictionary itself, so it survives the
        # resample and needs no bookkeeping in the forward. At alpha=0 it is
        # exactly ones, keeping a flat/pink A/B bit-identical.
        #
        # It ranks WITHIN a coordinate group. Ranking globally would suppress
        # the lag mirrors purely because they are stored second, handicapping
        # the warp by an accident of ordering.
        rank = torch.cat(
            [
                torch.arange(1, self.n_ratio + 1, dtype=torch.float32),
                torch.arange(1, self.n_lag + 1, dtype=torch.float32),
            ]
        )
        self.register_buffer("env_rank", rank, persistent=False)
        env = rank.pow(-self.alpha)
        self.register_buffer("envelope", env, persistent=False)
        mirrors = MIRROR_SCALE * torch.randn(
            N, R, R, generator=gen, dtype=torch.float32
        )
        self.register_buffer("mirrors", mirrors * env[:, None, None], persistent=False)

        # The turn, as base plus deviation. ``turn_static`` is the per-depth
        # free blend (the base, unbounded like HarmonicField.amplitudes);
        # ``turn`` drives the tanh-bounded input-conditional deviation. Both
        # zero-init, so the score matrix is exactly zero at step 0 and attention
        # opens uniform over the causal prefix. No bias on the Linear: the
        # static term already is the bias, per depth, and having both would be a
        # redundant parameterization of one degree of freedom.
        self.turn = nn.Linear(hidden_size, H * N, bias=False)
        nn.init.zeros_(self.turn.weight)
        self.turn_static = nn.Embedding(self.depths, H * N)
        nn.init.zeros_(self.turn_static.weight)

        # The facets: one rank-1 deformation per (depth, mirror), in CANONICAL
        # space. Deforming the grid before it is resampled keeps them
        # length-free too, and costs D*N*2R rather than D*N*2T.
        self.facet_u = nn.Parameter(torch.zeros(self.depths, N, R))
        self.facet_v = nn.Parameter(torch.randn(self.depths, N, R) * FACET_V_STD)

        self.value = nn.Linear(hidden_size, H * self.head_dim, bias=False)
        # Mega's output gate. Bias included (Mega's own form is
        # ``silu(X W + b)``), and left at the default init rather than zeroed:
        # a zero gate is a zero output, not an identity, so the "start inert"
        # discipline the mirrors and facets follow does not apply to it.
        self.gate = nn.Linear(hidden_size, H * self.head_dim, bias=True)
        self.output = nn.Linear(H * self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self._metrics: dict = {}

    @classmethod
    def patch_config(cls, config) -> None:
        """Correct the head COUNT to 1, and only the count.

        ``head_size`` is a width and stays in the config, so an unset one gives
        a single head spanning the full hidden size - exactly what the standing
        ``head_size or hidden_size // num_heads`` rule predicts once the count
        is 1. Rewriting the count here is what keeps config.json, the blueprint
        tab and the Arguments card reporting the head this block actually
        built. Idempotent: runs from the CLI and again from ``__init__``.
        """
        config.num_heads = 1
        config.num_queries = 1

    # ------------------------------------------------------------------ field
    def _canonical(self, depth: int) -> Tensor:
        """The deformed dictionary at canonical resolution: ``[N, R, R]``."""
        d = min(int(depth), self.depths - 1)
        u = self.facet_u[d].unsqueeze(-1)  # [N, R, 1]
        v = self.facet_v[d].unsqueeze(-2)  # [N, 1, R]
        return self.mirrors + FACET_SCALE * torch.tanh(u * v)

    def _lag_grid(self, T: int, device, dtype) -> Tensor:
        """Sampling grid for the lag half of the dictionary, ``[1, T, T, 2]``.

        Rows stay the query fraction; columns become ``log1p(lag)``, normalized
        so the longest representable lag lands on the canonical grid's far
        edge. The log is what buys the acuity: lag 1 sits a fixed distance from
        the diagonal in canonical cells at EVERY ``T``, where a uniform grid
        lets it collapse onto lag 0 as ``T`` grows. Non-causal cells (``j > i``)
        clamp to lag 0; they are masked before the softmax either way.
        """
        idx = torch.arange(T, device=device, dtype=dtype)
        denom = float(max(T - 1, 1))
        y = (2.0 * idx / denom - 1.0).view(T, 1).expand(T, T)
        lag = (idx.view(T, 1) - idx.view(1, T)).clamp_min(0.0)
        x = 2.0 * (torch.log1p(lag) / math.log1p(denom)) - 1.0
        return torch.stack((x, y), dim=-1).unsqueeze(0)

    def _zoom_grid(self, T: int, device, dtype) -> Tensor:
        """Folded sampling grids for the ratio group, ``[n_ratio, T, T, 2]``.

        Coordinates are the ordinary ratio grid multiplied by each mirror's zoom
        factor and folded back into ``[-1, 1]`` by a triangle wave, so a mirror
        at zoom ``z`` repeats ``z`` times across the sequence with no seam. Both
        axes zoom together: the pattern gets finer in query AND key position,
        which is what makes it a granularity knob rather than a window.
        """
        idx = torch.arange(T, device=device, dtype=dtype)
        t = 2.0 * idx / float(max(T - 1, 1)) - 1.0  # [-1, 1]
        z = self.zoom_factor.to(device=device, dtype=dtype).view(-1, 1)
        # Triangle fold of period 4 on [-1, 1]: exact identity at z == 1.
        a = ((z * t + 1.0) * 0.5) % 2.0
        folded = 2.0 * (1.0 - (a - 1.0).abs()) - 1.0  # [n_ratio, T]
        x = folded.view(-1, 1, T).expand(-1, T, T)
        y = folded.view(-1, T, 1).expand(-1, T, T)
        return torch.stack((x, y), dim=-1)

    def _resample(self, grid: Tensor, T: int) -> Tensor:
        """A canonical ``[N, R, R]`` dictionary at ``[N, T, T]``.

        Each mirror is read in its own coordinate system, so this is one
        ``interpolate`` for the ratio half and one ``grid_sample`` for the lag
        half. Differentiable through both, so the facets still learn.
        """
        parts = []
        if self.n_ratio:
            ratio = grid[: self.n_ratio]
            if self.zoom:
                # No T == resolution shortcut: a folded grid is a different
                # geometry at every length, exactly as on the lag path.
                g = self._zoom_grid(T, ratio.device, ratio.dtype)
                ratio = F.grid_sample(
                    ratio.unsqueeze(1),
                    g,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(1)
            elif T != self.resolution:
                ratio = F.interpolate(
                    ratio.unsqueeze(0),
                    size=(T, T),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
            parts.append(ratio)
        if self.n_lag:
            # No T == resolution shortcut here: the warp is a different
            # geometry at every length, including the canonical one.
            g = self._lag_grid(T, grid.device, grid.dtype)
            parts.append(
                F.grid_sample(
                    grid[self.n_ratio :].unsqueeze(0),
                    g,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0)
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _faceted_frozen(self, T: int) -> Tensor:
        """The undeformed dictionary at ``[N, T, T]`` - the facet-free control."""
        return self._resample(self.mirrors, T)

    def _faceted(self, depth: int, T: int) -> Tensor:
        """The dictionary as this depth sees it, resampled to ``[N, T, T]``.

        Deformation happens in CANONICAL space, before the resample, so the
        facets are length-free too. ``align_corners=True`` pins the canonical
        grid's corners to the sequence's, so a ratio mirror stretches to fit
        rather than being cropped. Any ``T`` works, including the ``T = 1`` of a
        cached decode step. Differentiable, so the facets learn through it.
        """
        return self._resample(self._canonical(depth), T)

    def _mirror_dropout(self, w: Tensor) -> Tensor:
        """Drop mirrors from the blend, per (example, token, head).

        SMEAR's load-balancing mechanism rather than an auxiliary balance loss,
        at its rate. Unlike SMEAR there is nothing to renormalize - these
        weights are free, not a distribution - so a drop is a plain mask, and
        dropping all N leaves ``w = 0``, i.e. uniform attention, which is the
        module's own identity state rather than a degenerate one.

        No inverted-dropout rescaling: the surviving weights are the model's
        actual coefficients on frozen matrices, and scaling them up would change
        the effective softmax temperature rather than preserve an expectation.
        The train/eval difference is a blend that is on average sparser during
        training, which is the intended pressure. Training-only is enforced
        HERE as well as at the call site: a method named ``_mirror_dropout``
        that silently drops during evaluation is a footgun.
        """
        if MIRROR_DROPOUT <= 0.0 or not self.training:
            return w
        keep = torch.rand_like(w) >= MIRROR_DROPOUT
        return w * keep

    def _maybe_dropoff(self, v: Tensor, current_depth: int) -> Tensor:
        """Withhold the causal tip (next/dropoff.md). TRAINING ONLY.

        Only the ``warp`` mode is offered. The ``shift`` mode shifts K as well
        as V, and there is no K here to shift - the scores come from the frozen
        mirrors, not from a key projection - so a V-only shift would be a
        different ablation wearing the same name.

        The envelope itself is imported rather than reimplemented: it is one
        idea, and a second copy of it would drift from the one the arc configs
        have been running. The training gate and the two schedules mirror
        ``CausalAttention._maybe_dropoff`` for the same reason.
        """
        if self.dropoff_step is None or not self.training:
            return v
        if not self.dropoff_every and current_depth != self.dropoff_step:
            return v
        from praxis.attention.causal import CausalAttention

        return CausalAttention._dropoff_warp_value(v)

    def _scores(self, w: Tensor, mirrors: Tensor) -> Tensor:
        """Blend the dictionary per query position.

        ``w`` is ``[B, T, H, N]`` and ``mirrors`` is ``[N, T, T]``; row ``i`` of
        the result is that token's own mixture, which is what makes the geometry
        input-conditional rather than merely learned. Costs ``N * T^2`` per
        (batch, head) against ``T^2 * d`` for a QK product.
        """
        out = torch.einsum("bihk,kij->bhij", w, mirrors)
        return out * self.mix_scale if self.mix_norm else out

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], float]:
        B, T, _ = inputs.shape
        N, H = self.num_mirrors, self.num_heads

        d = min(int(current_depth), self.depths - 1)
        cond = TURN_MOD * torch.tanh(self.turn(inputs).view(B, T, H, N))
        static = self.turn_static.weight[d].view(1, 1, H, N)
        w = (cond + static).float()
        if self.training:
            self._note_turn(w, cond, static)
            w = self._mirror_dropout(w)

        mirrors = self._faceted(current_depth, T).to(w.dtype)
        scores = self._scores(w, mirrors)  # [B, H, T, T]

        if self.causal:
            pos = torch.arange(T, device=inputs.device)
            lag = pos[:, None] - pos[None, :]
            allowed = lag >= 0
            if self.window_size is not None:
                allowed = allowed & (lag <= self.window_size)
            scores = scores.masked_fill(~allowed, float("-inf"))

        # Ghostmax without the column: softmax1 = softmax * Z/(1+Z), and
        # Z/(1+Z) = sigmoid(log Z), so the ghost costs one sigmoid on the
        # log-sum-exp the softmax already computes.
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, T]
        keep = torch.sigmoid(lse)
        self._note_ghost(keep)
        weights = torch.softmax(scores, dim=-1).to(inputs.dtype)

        v = self.value(inputs).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self._maybe_dropoff(v, current_depth)
        out = weights @ v  # [B, H, T, head_dim]
        out = out * keep.unsqueeze(-1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, T, H * self.head_dim)
        gate = F.silu(self.gate(inputs))
        self._note_gate(gate)
        return self.dropout(self.output(out * gate)), past_key_values, 0.0

    # ---------------------------------------------------------------- metrics
    @torch.no_grad()
    def _note_turn(self, w: Tensor, cond: Tensor, static: Tensor) -> None:
        """Turn diagnostics for a FREE SIGNED blend, ``[B, T, H, N]``.

        Entropy and the SMEAR mutual-information estimator both assume the
        weights are a distribution over mirrors. They are not any more, so they
        are gone rather than left reporting a number with no meaning. Their
        replacements are the scale-free ones the harmonic head already uses on
        its own free amplitudes.
        """
        n = w.shape[-1]
        if n < 2:
            return
        f = w.detach().float()

        # At init both halves are zero, so the blend is identically zero and
        # every ratio below is 0/0. Reporting them anyway would say "less than
        # one effective mirror" and "no mirror used", both of which read as
        # collapse - the exact opposite of an untouched identity start. Omit
        # until there is a blend to describe, the same discipline
        # kaleido_facet_depth_specialization follows.
        energy = f.pow(2).sum(-1)
        if float(energy.mean()) <= 1e-12:
            return

        # Effective number of mirrors in use: the participation ratio of the
        # weight vector, the same statistic harmonic_env_modes reports for the
        # envelope coefficients. 1 = one mirror carries everything (the
        # collapse this parameterization exists to avoid), N = all equal.
        w2 = f.pow(2)
        num = w2.sum(-1).pow(2)
        den = w2.pow(2).sum(-1).clamp_min(1e-24)
        self._metrics["kaleido_turn_modes"] = float((num / den).mean().item())

        # Is the span being used, or only the positive orthant a softmax could
        # have reached? This is the direct falsifier for leaving the simplex.
        self._metrics["kaleido_turn_negative"] = float((f < 0).float().mean().item())

        # ||w||: with unit-scale mirrors the score is ~N(0, ||w||^2), so this IS
        # the effective softmax temperature. Watch it for runaway sharpening.
        self._metrics["kaleido_turn_scale"] = float(
            f.pow(2).sum(-1).sqrt().mean().item()
        )

        # Are all N mirrors earning their keep? Fraction whose magnitude clears
        # half the mean magnitude; 1/N is collapse onto one mirror. Same
        # semantics as smear_expert_utilization, on |w| rather than on a
        # simplex share.
        mag = f.abs()
        self._metrics["kaleido_mirror_utilization"] = float(
            (mag > 0.5 * mag.mean(-1, keepdim=True)).float().mean().item()
        )

        # Proposition (iii) made measurable: is the blend SPENDING amplitude
        # against the envelope, or accepting it? Log-slope of |w_k| against
        # log k, normalized so 1.0 = fighting all the way (slope = alpha,
        # cancelling the envelope), 0 = accepting it, >1 = over-compensating.
        if self.alpha > 0.0:
            mag = f.abs().reshape(-1, n).mean(0).clamp_min(1e-9)
            # Against the rank the envelope actually used, which resets per
            # coordinate group - not against 1..N, which would read the
            # ratio/lag boundary as a jump in the ladder and bias the slope.
            x = torch.log(self.env_rank.to(mag.device).float())
            y = torch.log(mag)
            xc, yc = x - x.mean(), y - y.mean()
            slope = (xc * yc).sum() / xc.pow(2).sum().clamp_min(1e-12)
            self._metrics["kaleido_envelope_fight"] = float((slope / self.alpha).item())

        # Is the warped half of the dictionary earning its place? Share of
        # blend magnitude sitting on the lag-coordinate mirrors. The split is
        # even, so 0.5 is parity: below it the router prefers ratio geometry
        # and the warp is not paying for itself; at 0 the lag mirrors are dead
        # and this reduces to the ratio-only block. Absent when there are none.
        if self.n_lag:
            m = f.abs()
            lag_mass = m[..., self.n_ratio :].sum(-1)
            self._metrics["kaleido_lag_share"] = float(
                (lag_mass / m.sum(-1).clamp_min(1e-9)).mean().item()
            )

        # Which granularity the router actually buys. Weighted by |w| over the
        # ratio group only - the lag mirrors have no zoom.
        if self.zoom and self.n_ratio > 1:
            m = w.detach().abs().float()[..., : self.n_ratio]
            z = self.zoom_factor.to(m.device, m.dtype)
            # Position on the ladder rather than the raw factor, so the number
            # means the same thing when N changes: 0 = all mass on the coarsest
            # mirror, 1 = all on the finest, 0.5 = no preference.
            # In LOG space: the ladder is geometric, so a linear position would
            # put the identity nowhere near the middle. This way 0.5 is the
            # plain ratio mirror on a symmetric ladder.
            lz = torch.log(z)
            lo, hi = float(lz.min()), float(lz.max())
            pos = (lz - lo) / max(hi - lo, 1e-9)
            mass = m.sum(-1).clamp_min(1e-9)
            self._metrics["kaleido_zoom_mean"] = float(
                ((m * pos).sum(-1) / mass).mean().item()
            )

        # Which half of the blend does the work, on the axis that matters -
        # variance ACROSS mirrors, since a constant added to every mirror is not
        # a preference. 1.0 = a learned constant per depth, the input ignored,
        # which is Synthesizer's Mixture with per-depth alphas.
        v_static = static.detach().float().var(dim=-1).mean()
        v_cond = cond.detach().float().var(dim=-1).mean()
        total = v_static + v_cond
        if float(total) > 1e-12:
            self._metrics["kaleido_turn_static_share"] = float(
                (v_static / total).item()
            )

    @torch.no_grad()
    def _note_ghost(self, keep: Tensor) -> None:
        if not self.training or torch.compiler.is_compiling():
            return
        self._metrics["kaleido_ghost_share"] = float((1.0 - keep).mean().item())

    @torch.no_grad()
    def _note_gate(self, gate: Tensor) -> None:
        """Two on-device reductions, no host sync in the hot path.

        Skipped under torch.compile, where mutating module attributes forces a
        graph break - the same guard ``single.py::_record_gate`` uses.
        """
        if not self.training or torch.compiler.is_compiling():
            return
        self._metrics["kaleido_gate_negative"] = float(
            (gate < 0).to(gate.dtype).mean().item()
        )
        self._metrics["kaleido_gate_magnitude"] = float(gate.abs().mean().item())

    @staticmethod
    def _entropy(p: Tensor) -> Tensor:
        return -(p * (p + _EPS).log()).sum(dim=-1)

    def training_metrics(self) -> dict:
        from praxis.metrics.specialization import depth_dispersion

        out = dict(self._metrics)
        with torch.no_grad():
            stats = depth_dispersion(self.turn_static.weight.detach().float())
            if stats is not None:
                out["kaleido_turn_depth_specialization"] = stats["specialization"]
            delta = FACET_SCALE * torch.tanh(
                self.facet_u.detach().float().unsqueeze(-1)
                * self.facet_v.detach().float().unsqueeze(-2)
            )  # [D, N, T, T]
            out["kaleido_facet_strength"] = float(
                (delta.abs().mean() / FACET_SCALE).item()
            )
            if self.depths > 1:
                flat = delta.reshape(self.depths, -1)
                total = flat.pow(2).sum(dim=-1).mean()
                # No deformation at all means the question does not apply. A
                # ratio against ~0 energy reads 1.0 - a fully specialized field -
                # which is the exact opposite of the truth and is the value it
                # would report for every step before the facets leave zero.
                if total > 1e-12:
                    shared = flat.mean(dim=0).pow(2).sum()
                    out["kaleido_facet_depth_specialization"] = float(
                        (1.0 - shared / total).clamp(0.0, 1.0).item()
                    )
        return out
