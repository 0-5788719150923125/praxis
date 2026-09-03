"""Kaleidoscope attention: frozen mixing geometries, turned by a router.

A kaleidoscope's mirrors never change. Every pattern it produces comes from
turning the tube, and no pattern is stored anywhere inside it. This attention
is built the same way: ``N`` full ``[T, T]`` mixing matrices are drawn once at
construction and **never trained**, and everything the model learns is *which
combination of them to look through* at each token and each recurrent pass.

There are no Q and K projections. Nothing here is computed from content by a
pairwise comparison, so the projections have nothing to project - only ``V``
and the output survive. That is the whole point of the design and it is also
where its efficiency comes from: the score half costs ``N * T^2`` instead of
``T^2 * d``, and at ``N << d`` that is a large saving.

The mirrors are functions on the unit square in RELATIVE position, stored at a
canonical ``[R, R]`` and resampled to the live ``[T, T]`` every forward. They
are length-free: no span, nothing to slice, no sequence length that raises, and
the same geometry at every point of a sequence curriculum rather than a
different corner of one big matrix. See ``MIRROR_RES`` for what that costs.

    turn      w(x_i) = beta_d + m * tanh(W_turn x_i)     free, signed, per token
    facets    M_k^(d) = A_k + s * tanh(u_{d,k} (x) v_{d,k})
    scores    S[i, j] = sum_k w_k(x_i) * M_k^(d)[i, j]
    O         = ghostmax(mask(S)) @ dropoff(V)
    out       W_o (gamma * O),   gamma = silu(W_gamma x)

ONE HEAD, following ``arc_single``. Mega (Ma et al., arXiv:2209.10655)
Theorem 1: if G is a universal approximator then for every X there is a gate
``gamma = G(X)`` with ``gamma * O_single == O_multihead``, so one head plus a
learned elementwise gate spans what the heads spanned. The gate is SiLU and not
sigmoid for the reason ``praxis/attention/single.py`` gives at length - sigmoid
lands in (0, 1) and can only attenuate, while the theorem needs a gate that can
also amplify and flip sign - and ``kaleido_gate_negative`` reports whether that
freedom is ever used. ``patch_config`` corrects the head COUNT to 1 and touches
nothing else, so width still falls out of the standing
``head_size or hidden_size // num_heads`` rule.

The single head is a better fit here than it is for Arc, because the dictionary
was already shared across heads: multi-head kaleido would have been H
independent turns of one set of mirrors, which is a router widening rather than
new geometry. Collapsing to one head and gating the output loses nothing the
mirrors were providing.

GHOSTMAX IS ON, and ``ssog.py``'s reason for declining it does not transfer.
That module left the ghost out because "Softmax1's always-visible zero-logit
ghost would take roughly half of a Gaussian field's mass" - true there, because
its logits are log-DENSITIES and therefore large and negative, so a logit of
zero dominates them: measured, a Gaussian field hands the ghost 0.505 of the
mass at T=256 and does so at EVERY position. Kaleidoscope's logits are
unit-scale blends of N(0, 1) mirrors, and the same measurement over causal
prefixes gives a mean share of 0.054 at T=64, 0.018 at 256 and 0.010 at 512.
The objection was about a logit SCALE, not about ghostmax.

Those means are dominated by the START of the sequence and that is the point.
Position 0 has exactly one key, so its ghost share is ~0.50 whatever the logit
scale, while deep in the sequence it falls to 0.0034 at T=256 and 0.0017 at 512.
So the ghost is doing precisely the job SSOG had to build a learned null atom
for - it lets a query near the start say "there is nothing back there" - and it
costs nothing where there IS something to read. Read `kaleido_ghost_share`
against sequence length, not as an absolute.

It is applied without materializing the column. Softmax1 is ordinary softmax
scaled by ``Z / (1 + Z)``, and ``Z / (1 + Z) = sigmoid(log Z)``, so one sigmoid
on the log-sum-exp the softmax already needs gives it exactly - the same
identity ``ssog.py::_apply_null`` uses for its learned null atom. No extra
column, no wider mask.

DROPOFF is a registry profile (``kaleido_dropoff``), matching ``arc_dropoff``
and the ``arc_single_dropoff_nomem`` the abstractinator line runs. It reuses
``CausalAttention._dropoff_warp_value`` rather than reimplementing the envelope:
the ablation is one idea and two copies of it would drift.

WHY THIS IS NOT A KNOWN VARIANT. Synthesizer (Tay et al., arXiv:2005.00743)
asked whether the attention matrix can be synthesized rather than computed, and
covered nearly every cell of this space: a single frozen random ``[L, L]``
("Fixed Random", ~24 BLEU on WMT EnDe against ~27.3 for a Transformer), a
single trained one (which their own 2021 addendum notes *is* an MLP-Mixer
token-mixing layer), and a "Mixture of Synthesizers" that blends several. But
their mixture weights are ``alpha_{i,h,l}`` - static learned scalars indexed by
head and layer. They are parameters, not functions of the input. Nobody made
the blend input-conditional, which is the only cell left and the one this file
occupies. Their static mixture already reached parity with a vanilla
Transformer, so that - not fixed-random - is the bar.

MIXING HAPPENS BEFORE THE SOFTMAX, and the choice is load-bearing. Blending
after the softmax is a convex combination of distributions, which can only
*interpolate* between the frozen patterns: everything reachable lies inside
their hull. Blending logits is log-linear pooling, and
``softmax(a*A + b*B) ~ exp(A)^a * exp(B)^b`` is an intersection - it can put
mass where two mirrors *agree* and nowhere else, which is a pattern neither
mirror contains. Synthesizer also mixes inside the softmax.

THE PER-DEPTH BIAS GOES ON THE MIRRORS, NOT ON THE INPUTS, and those are
genuinely different transformations. Arc adds ``nn.Embedding(depth, dim)`` to
the *projected inputs* (``praxis/attention/arc.py``); for a linear map that is
``W(x + b) = Wx + Wb``, a constant offset, identical for every token. Biasing
the operator gives ``(W + B)x = Wx + Bx``, a correction that scales with what
it acts on. Here the operator IS the score matrix and it passes through a
softmax, so an additive bias on the mirror is a MULTIPLICATIVE reweighting of
the geometry - the same coupling argument ``HarmonicField`` makes for applying
its field as ``h * (1 + b)`` rather than ``h + b``: the upstream cannot cancel
it by emitting the difference.

The deformation has to be PER MIRROR or it collapses. The turn weights sum to
one, so a bias added to every mirror alike factors straight back out,
``sum_k w_k (A_k + B) = (sum_k w_k A_k) + B``, and all it buys is a per-depth
score bias. Giving each (depth, mirror) pair its own rank-1 deformation does
not factor: each pass sees a differently ground set of mirrors while their
frozen core persists. Rank 1 keeps this at ``D * N * 2T`` parameters.

THE BLEND IS NOT A SIMPLEX, and that is the whole point of the mechanism.

Softmax weights sum to one, which confines the blend to the CONVEX HULL of the
mirrors - an (N-1)-simplex. Worse, softmax's exponential pressures the weights
toward one-hot, and at one-hot the blend is EXACTLY one frozen mirror: the
synthesis disappears and what is left is Synthesizer's Fixed Random evaluated
per token, the variant measured to be worse than a trained matrix. Measured on
the first cut of this file: blend entropy 0.31 means the top mirror carried
~90%, and mirror utilization was oscillating down to 1/N.

That is the same failure ``praxis/routers/smear.py`` records for itself -
"nothing stopped one deviation per target from monopolizing its coefficient,
and on abstractinator-m every one of the twelve targets duly saturated to near
one-hot" - and it is why sharpening is off by default there.

Free signed weights give the linear SPAN instead (dimension N, not N-1), and a
negative weight SUBTRACTS a mirror, which no mixture of any weighting can reach.
In the product-of-experts reading below, a negative exponent is "attend where
this mirror says not to". This is also what the harmonic head already does:
``HarmonicField.amplitudes`` is a free real parameter over a frozen basis, not a
distribution over it. A simplex here was the inconsistent choice.

BASE PLUS DEVIATION, which is SMEAR's own form. ``beta_d`` is a per-depth free
blend - the static preference, "mirror 2 is generally useful at pass 3" - and
``m * tanh(W_turn x)`` is the bounded input-conditional deviation on top. The
static half is unbounded, matching ``amplitudes``; the deviation is tanh-bounded
the way ``AMP_MOD_DEPTH`` bounds the harmonic envelope, so the slow blend stays
foundational and the per-token part cannot run away with the logit scale.

Both zero-init, so at step 0 the score matrix is exactly ZERO and attention is
uniform over the causal prefix. That is a cleaner identity start than the old
softmax gave (the dictionary mean, an arbitrary random matrix), and every
departure from it is learned.

MIRROR DROPOUT keeps the blend honest, at SMEAR's own rate and by SMEAR's own
mechanism - expert dropout, not an auxiliary balance loss or a DeepSeek-style
bias. Dropping every mirror is safe here for the same reason it is safe there:
``w`` becomes zero and the score falls back to uniform attention exactly,
rather than to an all-zero parameter block.

THE TURN IS NOT A SMEAR ROUTER, and the difference is in kaleidoscope's favour.
It is a plain ``nn.Linear`` plus softmax; nothing is imported from
``praxis/routers/smear.py``. What is borrowed is the ``input_dependence``
estimator for the metric, and the count N=4, which is inherited from that
module's expert count and is therefore an arbitrary starting point rather than a
calibrated one.

The mechanism is SMEAR-SHAPED in the way that matters - merge N things by a
softmax and apply the merged object once, rather than run N and average outputs
- but the reduction is PER TOKEN. ``smear.py`` lists its own reductions as
``("token", "example", "batch")`` and states the honest limit: "Routing is per
EXAMPLE, never per TOKEN ... a per-token merge would need a distinct geometry
per position." That is exactly what a row of a ``[T, T]`` mixing matrix already
is, so the thing SMEAR cannot reach for a Linear target is free here.

WHAT TO WATCH FIRST, before perplexity: ``kaleido_turn_modes``, the effective
number of mirrors in the blend. At 1 the score IS one frozen matrix and this has
silently become Synthesizer's Fixed Random per token - the known-worse variant,
and the failure the softmax cut of this file actually hit. ``kaleido_turn_negative``
is the companion: pinned at 0 means the free parameterization bought nothing a
softmax could not have done.

Do NOT read collapse here as SMEAR's constant-router fixed point. That one is
specific to the BATCH reduction, where "the loss reaches the routing only
through ``probs.mean(0)``, so every example receives the identical routing
gradient" - measured decaying to exactly 0 on abstractinator-m. This routes per
token, with a distinct gradient per position.
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
# RELATIVE position, sampled at whatever resolution the live sequence needs, so
# a mirror is length-free: there is no span, nothing to slice, and no sequence
# length that raises. `[R, R]` is bilinearly resampled to `[T, T]` every
# forward, which at align_corners=True maps the grid's corners onto the
# sequence's corners - the whole distribution shrinks or stretches to fit.
#
# WHAT THIS BUYS. Under a sequence curriculum T changes every batch. An
# absolute-indexed dictionary hands the model a DIFFERENT geometry at each T (a
# different corner slice of one big random matrix, with no relationship between
# them); a ratio-indexed one hands it the SAME geometry resampled. It also makes
# the module a continuous frozen basis evaluated at the positions in use, which
# is exactly what HarmonicField does with `_phase_table` rather than storing a
# `[T, D]` table.
#
# WHAT IT COSTS, measured. Ratio structure survives resampling exactly -
# "attend to the start", "attend a third of the way back", the diagonal itself.
# FIXED-LAG structure does not: a canonical previous-token band (lag 1 at R=64)
# resamples to lag 2 with width 5 at T=128, and lag 5 with width 21 at T=512.
# So R is the knob trading length-invariance against positional acuity, and a
# dictionary of ratio mirrors cannot express "the token immediately before" at
# large T. A Toeplitz mirror built from a 1D lag kernel would be length-free in
# the other coordinate system and is the natural way to span both; not built.
MIRROR_RES: int = 64
# Logit scale of a fresh mirror. Softmax over t keys with iid N(0, 1) logits
# puts its peak roughly sqrt(2 ln t) above the mean - about 3.5 at t = 512 - so
# a unit-scale mirror is sparse without being one-hot. Flatter than this and
# every mirror is the prefix mean; sharper and each is a single random key.
MIRROR_SCALE: float = 1.0
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
# Probability of dropping a mirror from a blend during training. SMEAR's own
# load-balancing mechanism at SMEAR's own rate; see praxis/routers/smear.py,
# where its absence let "every one of the twelve targets duly saturate to near
# one-hot" on abstractinator-m. Dropping all N is safe: w becomes zero and the
# score falls back to uniform attention exactly.
MIRROR_DROPOUT: float = 0.1
_EPS: float = 1e-9


class KaleidoscopeAttention(nn.Module):
    """N frozen ``[T, T]`` mixing matrices, blended per token by a router."""

    # This block already routes its own parameters PER TOKEN, which is exactly
    # the condition praxis/routers/targeting.py's structural exclusion names:
    # "a module that already routes its own parameters per token gains nothing
    # from a per-batch merge wrapped around it."
    #
    # Without this flag, discovery targets `attn.turn.weight` and wraps the turn
    # in a SMEAR MergedLinear routed PER EXAMPLE - a coarser router around a
    # finer one - and also batch-mean-merges `facet_u`, `facet_v` and
    # `turn_static`, all of which are already per-depth conditioned. Verified by
    # running discover_targets against a built model.
    #
    # The decisive reason is measurement, not cost: if SMEAR varies
    # `turn.weight` per example, `kaleido_turn_modes` picks up variation
    # caused by SMEAR's router rather than by this one, and the first number
    # this architecture is meant to be judged on stops measuring what it claims.
    #
    # The flag covers the whole subtree, so `value`, `gate` and `output` are
    # excluded too. That is a real loss - they are ordinary projections and
    # routing them is what SMEAR does for arc - but it is the conservative side
    # to err on while the block's own routing is the thing under test. Splitting
    # the geometry machinery into an opaque submodule would recover them, at the
    # cost of changing parameter qualnames.
    MERGE_OPAQUE: bool = True

    num_mirrors: int = NUM_MIRRORS
    resolution: int = MIRROR_RES

    metric_descriptions = {
        "kaleido_turn_modes": {
            "description": (
                "Effective number of mirrors in the blend - the participation "
                "ratio of the weight vector, the same statistic "
                "harmonic_env_modes reports for the envelope coefficients. 1 = "
                "one mirror carries everything, which is the collapse this "
                "parameterization exists to avoid: at one mirror the score IS "
                "that frozen matrix and the block is Synthesizer's Fixed Random "
                "evaluated per token. N = all mirrors in play, where the "
                "log-linear pool reaches geometry outside the dictionary's "
                "hull. THIS IS THE FIRST NUMBER TO READ. The softmax version of "
                "this block saturated to ~90% on one mirror, which is what "
                "smear.py records for itself before it added expert dropout. "
                "Absent at init, where the blend is identically zero."
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
                "Fraction of blend weights below zero - the direct falsifier "
                "for leaving the simplex. A softmax blend cannot produce one, "
                "and a negative weight SUBTRACTS a mirror: in the "
                "product-of-experts reading it is a negative exponent, 'attend "
                "where this mirror says not to', which no mixture of any "
                "weighting can reach. Pinned near 0 means the free "
                "parameterization bought nothing a softmax could not have done, "
                "and the simplex was not the constraint after all."
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
                "Mean ||w|| across mirrors. With unit-scale mirrors the score "
                "is ~N(0, ||w||^2), so this IS the effective softmax "
                "temperature: the model controls its own attention sharpness "
                "through it, a degree of freedom a simplex does not have. "
                "Growing without bound is the failure to watch - it sharpens "
                "attention toward a single key - and is why the "
                "input-conditional half is tanh-bounded while the static half "
                "is free. 0 at init, where the score is exactly zero and "
                "attention is uniform over the causal prefix."
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
                "Which half of the blend is doing the work: variance across "
                "mirrors of the per-depth STATIC term over the total of static "
                "plus input-conditional, on the axis that actually drives the "
                "softmax (a constant added to every mirror cancels there, so a "
                "raw norm would be the wrong quantity). 1.0 = the blend is a "
                "learned constant per depth and the input is ignored, which is "
                "Synthesizer's Mixture with per-depth alphas and NOT the claim "
                "this architecture makes. 0.0 = purely input-driven. Undefined "
                "at init, when both terms are zero. Read alongside "
                "kaleido_turn_modes: a high static share is the honest "
                "way for the model to say the dictionary needs no per-token "
                "selection."
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
                "Between-depth variance of the static blend over its total "
                "energy, the same statistic as arc_qkv_specialization. 0 means "
                "every recurrent pass learned the same preference over mirrors "
                "(and is also the zero-init value), so a single shared blend "
                "would have done; rising means each pass prefers a different "
                "region of the dictionary. Distinct from "
                "kaleido_facet_depth_specialization, which asks whether each "
                "pass GRINDS the mirrors differently rather than which ones it "
                "reaches for."
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
                "Fraction of mirrors whose |weight| clears half the mean "
                "magnitude - smear_expert_utilization's estimator, on "
                "magnitudes rather than on a simplex share, since these weights "
                "are free and signed. 1/N is collapse onto one mirror. Read "
                "WITH kaleido_turn_modes: this one counts how many mirrors "
                "clear a threshold, that one gives the effective count "
                "continuously. Together they say which way an N-sweep should "
                "go - both near N argues for MORE mirrors, a falling pair says "
                "the dictionary is already larger than the model can use."
            ),
            "chart": {
                "title": "Kaleidoscope Mirror Utilization",
                "y_label": "fraction above half fair share",
                "y_scale": "linear",
                "group": "kaleidoscope",
                "order": 15,
            },
        },
        "kaleido_gate_negative": {
            "description": (
                "Fraction of SiLU output-gate values below zero. Exactly what "
                "a sigmoid gate cannot produce: pinned at 0 means the single "
                "head never needed to flip a feature's sign and a sigmoid "
                "would have done as well; rising means the gate is using the "
                "amplify/invert freedom Mega's Theorem 1 asks for - which is "
                "the whole justification for running one head over a shared "
                "dictionary instead of several."
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
                "Mean absolute SiLU gate value. Near 0 means the gate is "
                "closing the attention branch off entirely and the block is "
                "being routed around; read it against the negative fraction, "
                "since a gate can be large and one-sided or small and mixed."
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
                "Share of attention mass taken by ghostmax's always-visible "
                "zero logit - the 'attend to nothing' escape - averaged over "
                "query positions. Opens near 0.018 at 256 positions and 0.010 "
                "at 512, and the value is LENGTH-DEPENDENT by construction: "
                "position 0 has one key and gives the ghost ~0.50 whatever the "
                "logits do, while the tip gives it ~0.003. So this line moves "
                "with the sequence curriculum and a drop is not necessarily "
                "the model learning to read. Compare like lengths. Where "
                "ssog.py declined the ghost outright, a log-density field "
                "hands it 0.505 at EVERY position; unit-scale mirrors do not. "
                "A share near 1 is an attention branch that switched itself "
                "off."
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
                "Fraction of the per-depth facet deformation that is "
                "depth-specific: 1 - ||mean_d||^2 / mean_d||.||^2. Zero means "
                "every recurrent pass ground its mirrors the same way, so a "
                "single shared facet would have done, and the depth axis is "
                "not earning its parameters. Rising means each pass is "
                "reshaping the dictionary differently. The direct analogue of "
                "arc_qkv_specialization, and the reason the bias sits on the "
                "mirrors rather than on the inputs."
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
                "Mean |deformation| as a fraction of its cap, so 1.0 means the "
                "facets are pinned at FACET_SCALE and the frozen mirrors are "
                "no longer foundational. Exactly 0 at init. A run that stays "
                "near 0 is a FINDING and not a failure: it says the frozen "
                "dictionary plus routing was sufficient and depth needed no "
                "deformation at all."
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
        dropoff: Optional[str] = None,
        dropoff_every: bool = False,
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

        self.resolution = int(resolution or type(self).resolution)

        N, H = self.num_mirrors, self.num_heads

        # The dictionary. Shared across heads on purpose: one frozen basis that
        # every head reads differently is both cheaper (N * T^2 rather than
        # H * N * T^2) and the more interesting claim - the heads differ by how
        # they turn, not by owning private geometry.
        gen = torch.Generator().manual_seed(MIRROR_SEED)
        R = self.resolution
        mirrors = MIRROR_SCALE * torch.randn(
            N, R, R, generator=gen, dtype=torch.float32
        )
        self.register_buffer("mirrors", mirrors, persistent=False)

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

    def _faceted_frozen(self, T: int) -> Tensor:
        """The undeformed dictionary at ``[N, T, T]`` - the facet-free control."""
        if T == self.resolution:
            return self.mirrors
        return F.interpolate(
            self.mirrors.unsqueeze(0), size=(T, T), mode="bilinear", align_corners=True
        ).squeeze(0)

    def _faceted(self, depth: int, T: int) -> Tensor:
        """The dictionary as this depth sees it, resampled to ``[N, T, T]``.

        ``align_corners=True`` pins the canonical grid's corners to the
        sequence's, so the whole distribution stretches or shrinks to fit rather
        than being cropped. Any ``T`` works, including the ``T = 1`` of a cached
        decode step. Differentiable, so the facets learn through it.
        """
        grid = self._canonical(depth)
        if T == self.resolution:
            return grid
        return F.interpolate(
            grid.unsqueeze(0), size=(T, T), mode="bilinear", align_corners=True
        ).squeeze(0)

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
        return torch.einsum("bihk,kij->bhij", w, mirrors)

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
