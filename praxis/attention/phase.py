"""Phase-coupling stages for SelfAttention: Kuramoto Attention (arXiv:2606.11585).

``PhaseProjection`` reads the attention input as a bank of phase oscillators
and lifts it into queries, keys and values. ``TangentStep`` turns the attended
values into a synchronization step. ``SelfAttention`` takes them through its
``projection`` and ``output`` arguments, and the ``kuramoto`` attention profile
composes the two.
"""

import math
from typing import Callable, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from praxis.activations.serpent import INV_FLOOR_EPS

_GROUP = {"group": "kuramoto", "group_order": 40}


class PhaseState(NamedTuple):
    """What the projection hands the output stage, per head: ``[B, T, H, k_h]``."""

    cos: Tensor
    sin: Tensor
    # [k]: turns a phase step back into a step on the input coordinates.
    inverse_frequency: Tensor


class PhaseProjection(nn.Module):
    """Reads the input as one phase per coordinate and builds Q, K and V from it.

    Coordinate ``j`` has phase ``theta_j = alpha_j * x_j``, with a learned
    per-coordinate frequency ``alpha`` starting at 1 (Serpent's
    parameterization). Oscillators are split into heads of ``head_size // 2``.
    Within a head the value is the lift ``psi = (cos theta, sin theta)``, and
    the query and key are that lift scaled by positive gates
    ``softplus(W psi(theta))``, normalized to mean 1 over the head and read
    densely off every phase. Once rotary position is applied, the score is the
    paper's gated phase similarity
    ``tau * sum_j g^q_j g^k_j cos(theta_tj - theta_uj + omega_j (t - u))``, and
    the attended value is the resultant ``G = sum_u A_tu e^{i theta_u}``.
    """

    metric_descriptions = {
        "kuramoto_global_order": {
            "description": (
                "Phase coherence across a whole sequence, |mean of e^(i theta)| "
                "over positions. 1 = all tokens in phase. The paper saw it fall "
                "with depth while local order stayed high."
            ),
            "chart": {
                "title": "Global Phase Order",
                "y_label": "order parameter",
                "order": 20,
                **_GROUP,
            },
        },
        "kuramoto_tau": {
            "description": (
                "Learned score scale (inverse temperature of phase selection), "
                "mean over heads. Rising = sharper selection."
            ),
            "chart": {
                "title": "Phase Score Scale",
                "y_label": "tau",
                "y_scale": "logarithmic",
                "order": 60,
                **_GROUP,
            },
        },
        "kuramoto_read_frequency": {
            "description": (
                "Mean |alpha|: radians of phase per unit of normalized input. "
                "Higher = the input is read as faster-turning phases."
            ),
            "chart": {
                "title": "Phase Read Frequency",
                "y_label": "radians per unit",
                "order": 70,
                **_GROUP,
            },
        },
    }
    per_layer_metrics = frozenset(metric_descriptions)

    @classmethod
    def patch_config(cls, config) -> None:
        """Keep ``head_size`` as the head width and derive the count from it.

        Every input coordinate is an oscillator, ``head_size // 2`` per head, so
        ``num_heads = hidden_size // (head_size // 2)``. Queries and keys read
        the same phases, so there is one query per head. An unset ``head_size``
        is pinned to its fallback (``hidden_size // num_heads``) rounded down to
        even, since the fallback would move with the corrected count. Idempotent.
        """
        width = config.head_size or config.hidden_size // config.num_heads // 2 * 2
        if width < 2 or width % 2:
            raise ValueError(
                f"phase heads pair cos with sin, so head_size must be even "
                f"(got {width})"
            )
        config.head_size = width
        config.num_heads = config.hidden_size // (width // 2)
        config.num_queries = 1

    def __init__(self, config) -> None:
        super().__init__()
        self.patch_config(config)
        self.num_heads = config.num_heads
        self.head_oscillators = config.head_size // 2
        self.oscillators = self.num_heads * self.head_oscillators
        k = self.oscillators

        self.frequency = nn.Parameter(torch.ones(k))
        # [g^q | g^k] in one matmul.
        self.gates = nn.Linear(2 * k, 2 * k, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer(
            "_stats", torch.full((1,), float("nan")), persistent=False
        )

    def forward(self, inputs: Tensor):
        """Returns flat ``q, k, v`` of width ``num_heads * head_size`` and the state."""
        B, T, _ = inputs.shape
        H, kh = self.num_heads, self.head_oscillators
        theta = inputs[..., : self.oscillators].float() * self.frequency.float()
        cos, sin = theta.cos(), theta.sin()

        raw = self.gates(torch.cat([cos, sin], dim=-1))
        # Q/K/V leave in the matmul's (autocast) dtype. Float32 phases would
        # run the attention kernel in float32: ~10x slower at 1024 positions.
        dtype = raw.dtype
        gates = F.softplus(raw).view(B, T, 2, H, kh)
        gates = gates / gates.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        gq, gk = gates[:, :, 0], gates[:, :, 1]

        cos, sin = cos.view(B, T, H, kh), sin.view(B, T, H, kh)
        psi = torch.cat([cos, sin], dim=-1)  # rotate-half pairs per head
        tau = self.log_tau.exp().view(1, 1, H, 1)
        q = (psi * torch.cat([gq, gq], dim=-1) * tau).to(dtype)
        k = (psi * torch.cat([gk, gk], dim=-1)).to(dtype)
        psi = psi.to(dtype)

        if self.training:
            self._record(cos, sin)
        alpha = self.frequency
        state = PhaseState(cos, sin, alpha / (alpha.pow(2) + INV_FLOOR_EPS**2))
        return q.flatten(2), k.flatten(2), psi.flatten(2), state

    def _record(self, cos: Tensor, sin: Tensor) -> None:
        """Global order into a buffer, in-graph: no host sync, no graph break."""
        with torch.no_grad():
            order = (cos.float().mean(1).pow(2) + sin.float().mean(1).pow(2)).sqrt()
            self._stats.copy_(order.mean().reshape(1))

    def training_metrics(self) -> dict:
        with torch.no_grad():
            order = self._stats.item()
            return {
                "kuramoto_global_order": None if math.isnan(order) else order,
                "kuramoto_tau": self.log_tau.exp().mean().item(),
                "kuramoto_read_frequency": self.frequency.abs().mean().item(),
            }


class TangentStep(nn.Module):
    """Turns the attended phases into a synchronization step.

    Projecting the resultant ``G`` onto each oscillator's tangent gives
    ``a = -sin(theta) Re G + cos(theta) Im G = sum_u A_tu sin(theta_u - theta_t)``,
    exactly the Kuramoto coupling direction for the attention weights (the
    paper's Lemma 2). A signed per-oscillator gate ``v = W psi(theta) + b``
    scales it: positive pulls toward the attended circular mean, negative pushes
    away. The step is bounded by a norm-matched tanh with a learned radius,
    ``r tanh(|x| / r) x / |x|``, then divided by the read frequency to land on
    the input's scale. Coordinates past the last whole head get no step.

    In a pre-norm block the input is the normalized stream, so the direction is
    exact coupling for this layer's phases, while the size of the move on the
    next layer's phases is learned (radius, gate) rather than fixed.
    """

    metric_descriptions = {
        "kuramoto_local_order": {
            "description": (
                "Coherence of each token with what it attends to: |G| / (1 - "
                "ghost mass). 1 = the attended phases agree. The paper saw it "
                "near 1 at every layer."
            ),
            "chart": {
                "title": "Local Phase Order",
                "y_label": "order parameter",
                "order": 10,
                **_GROUP,
            },
        },
        "kuramoto_decoupled": {
            "description": (
                "Attention mass on the ghost key: the share of coupling a token "
                "declines to spend on any token."
            ),
            "chart": {
                "title": "Decoupled Mass",
                "y_label": "fraction of attention",
                "order": 30,
                **_GROUP,
            },
        },
        "kuramoto_repel": {
            "description": (
                "Fraction of value-gate entries below zero, which push an "
                "oscillator away from its attended mean. Starts near 0.5."
            ),
            "chart": {
                "title": "Repelling Oscillators",
                "y_label": "fraction",
                "order": 40,
                **_GROUP,
            },
        },
        "kuramoto_step": {
            "description": (
                "RMS phase step per oscillator after the bound, in radians: how "
                "far one layer moves the phases."
            ),
            "chart": {
                "title": "Phase Step",
                "y_label": "radians (RMS)",
                "y_scale": "logarithmic",
                "order": 50,
                **_GROUP,
            },
        },
        "kuramoto_bound": {
            "description": (
                "Bounded over raw step norm. 1 = the bound is linear; falling = "
                "it compresses large steps."
            ),
            "chart": {
                "title": "Step Bound",
                "y_label": "bounded / raw",
                "order": 55,
                **_GROUP,
            },
        },
    }
    per_layer_metrics = frozenset(metric_descriptions)

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_oscillators = config.head_size // 2
        self.oscillators = self.num_heads * self.head_oscillators
        k = self.oscillators

        self.value_gate = nn.Linear(2 * k, k, bias=True)
        # Radius sqrt(k): typical starting steps sit in the linear range.
        self.log_radius = nn.Parameter(torch.full((1,), 0.5 * math.log(k)))
        self.register_buffer(
            "_stats", torch.full((5,), float("nan")), persistent=False
        )

    def forward(
        self,
        attended: Tensor,
        state: PhaseState,
        ghost: Optional[Tensor] = None,
        resultant: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """``attended``: ``[B, T, H * head_size]``. ``ghost``: ``[B, H, T]``, the
        attention mass on the ghost key. ``resultant``: the attended phases'
        (re, im) ``[B, T, H, k_h]`` when ``attended`` is something else. The
        last two feed the metrics only."""
        B, T, _ = attended.shape
        G = attended.view(B, T, self.num_heads, 2, self.head_oscillators)
        re, im = G[..., 0, :], G[..., 1, :]
        a = (state.cos * im - state.sin * re).flatten(2)

        psi = torch.cat([state.cos.flatten(2), state.sin.flatten(2)], dim=-1)
        gate = self.value_gate(psi)
        raw = gate * a
        radius = self.log_radius.exp()
        norm = raw.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = radius * torch.tanh(norm / radius) / norm
        step = raw * scale.to(raw.dtype)

        if self.training:
            order = (re, im) if resultant is None else resultant
            self._record(*order, ghost, gate, step, scale)
        out = step * state.inverse_frequency.to(step.dtype)
        return F.pad(out, (0, self.hidden_size - self.oscillators))

    def _record(self, re, im, ghost, gate, step, scale) -> None:
        """Step statistics into a buffer, in-graph: no host sync, no graph break."""
        with torch.no_grad():
            resultant = (re.float().pow(2) + im.float().pow(2)).sqrt()
            if ghost is None:
                ghost = torch.zeros(
                    re.shape[0], re.shape[2], re.shape[1], device=re.device
                )
            coupled = (1.0 - ghost.float()).transpose(1, 2).unsqueeze(-1)
            stats = torch.stack(
                [
                    (resultant / coupled.clamp_min(1e-6)).mean(),
                    ghost.float().mean(),
                    (gate < 0).float().mean(),
                    step.float().pow(2).mean().sqrt(),
                    scale.float().mean(),
                ]
            )
            self._stats.copy_(stats)

    def training_metrics(self) -> dict:
        stats = self._stats.tolist()
        if math.isnan(stats[0]):
            return {}
        keys = (
            "kuramoto_local_order",
            "kuramoto_decoupled",
            "kuramoto_repel",
            "kuramoto_step",
            "kuramoto_bound",
        )
        return dict(zip(keys, stats))


def _squash(s: Tensor) -> Tensor:
    """Sabour et al.'s squash over the last dimension."""
    n2 = s.pow(2).sum(dim=-1, keepdim=True)
    return s * (n2 / (1 + n2)) * torch.rsqrt(n2 + 1e-8)


class CapsuleTangentStep(TangentStep):
    """TangentStep fed by routing by agreement across positions (Sabour et
    al. 2017, arXiv:1710.09829) over transported harmonic phase poses.

    Token ``u``'s part pose in a head is the first ``harmonics`` harmonics of
    its oscillators in the rotary frame, ``exp(i n (theta_u + omega u))``: each
    phase carried forward by its own natural frequency, so parts from any
    distance meet the whole at ``t`` in one frame. A learned complex weight per
    (head, harmonic, oscillator) makes it a vote. Round 1 attends with the
    paper's score; each later round adds Sabour's logit update, the agreement
    ``Re <vote_u, v_t>`` with the squashed whole of the round before, as extra
    query/key width on the same kernel. The final whole, brought back to the
    token's frame, sets the step ``a_j = sum_n Im(exp(-i n theta_j) v_j[n])``: a
    learned-harmonic (Kuramoto-Daido) coupling, the paper's
    ``sin(theta_u - theta_t)`` when only the first harmonic is on.

    Against the references: the paper's values are position-free raw phases;
    these are transported. Sabour normalizes each part over its wholes; here
    each whole normalizes over its parts, since a part spread over later
    tokens would leak them. A part and its whole are the same head, so the vote
    weight is elementwise, the only form that commutes with per-oscillator
    transport (and the coordinate-separable value path the paper's ablations
    favor). Sets ``encoding`` to ``rope_learned_rates`` for the natural
    frequencies.
    """

    metric_descriptions = {
        **TangentStep.metric_descriptions,
        "capsule_routing_shift": {
            "description": (
                "1 - cosine between the first and last round's aggregated votes. "
                "0 = agreement rounds re-select nothing; higher = they re-route."
            ),
            "chart": {
                "title": "Routing Shift",
                "y_label": "1 - cosine",
                "order": 50,
                "group": "capsules",
                "group_order": 41,
            },
        },
        "capsule_whole_length": {
            "description": (
                "Mean length of the squashed whole each token assembles from "
                "earlier tokens' votes, in (0, 1). Long = the votes agree."
            ),
            "chart": {
                "title": "Whole Length",
                "y_label": "length",
                "order": 60,
                "group": "capsules",
                "group_order": 41,
            },
        },
        "capsule_vote_harmonic_share": {
            "description": (
                "Share of vote-weight energy on harmonics 2 and up. Rising = the "
                "coupling sharpens past a single sine."
            ),
            "chart": {
                "title": "Vote Harmonic Share",
                "y_label": "fraction",
                "order": 70,
                "group": "capsules",
                "group_order": 41,
            },
        },
    }
    per_layer_metrics = frozenset(metric_descriptions)

    @classmethod
    def patch_config(cls, config) -> None:
        """The votes travel at the oscillators' natural frequencies, which only
        ``rope_learned_rates`` provides. Idempotent."""
        config.encoding = "rope_learned_rates"

    def __init__(self, config, harmonics: int = 3, rounds: int = 3) -> None:
        super().__init__(config)
        encoding = getattr(config, "encoding", None)
        if encoding != "rope_learned_rates":
            raise ValueError(
                "kuramoto_capsules carries each vote forward by its oscillator's "
                "natural frequency, so it needs encoding_type rope_learned_rates "
                f"(got {encoding!r})"
            )
        if rounds < 1:
            raise ValueError(f"routing needs at least one round (got {rounds})")
        self.harmonics = harmonics
        self.rounds = rounds
        # (real, imag) per head, harmonic, oscillator; harmonic n starts at 1/n.
        weight = torch.zeros(2, self.num_heads, harmonics, self.head_oscillators)
        weight[0] = 1.0 / torch.arange(1, harmonics + 1).float().view(1, -1, 1)
        self.vote_weight = nn.Parameter(weight)
        self.register_buffer(
            "_capsule_stats", torch.full((2,), float("nan")), persistent=False
        )

    def route(
        self,
        attend: Callable,
        q: Tensor,
        k: Tensor,
        state: PhaseState,
        rates: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """``attend(q, k, v, scale, with_ghost) -> (out, ghost)`` is the
        module's masked ghostmax attention. ``q``, ``k``: the rotated score lifts ``[B, H, T,
        head_size]``. ``rates``: ``[H, k_h]`` natural frequencies."""
        B, H, T, D = q.shape
        K, m = self.harmonics, self.head_oscillators
        if positions is None:
            positions = torch.arange(T, device=q.device)
        positions = positions.view(1, -1) if positions.dim() == 1 else positions

        # The fundamental in the rotary frame, [B, T, H, m].
        angle = positions.float()[:, :, None, None] * rates.float()[None, None]
        rc, rs = angle.cos(), angle.sin()
        zr = state.cos * rc - state.sin * rs
        zi = state.cos * rs + state.sin * rc

        pr, pi = [zr], [zi]
        for _ in range(K - 1):
            pr, pi = pr + [pr[-1] * zr - pi[-1] * zi], pi + [pr[-1] * zi + pi[-1] * zr]
        pr, pi = torch.stack(pr, dim=-2), torch.stack(pi, dim=-2)  # [B, T, H, K, m]
        wr, wi = self.vote_weight.float()
        vr, vi = pr * wr - pi * wi, pr * wi + pi * wr
        # [B, H, T, K * 2 * m]: per harmonic, real then imaginary halves.
        votes = torch.stack([vr, vi], dim=-2).flatten(-3).transpose(1, 2)
        raw = torch.cat([zr, zi], dim=-1).transpose(1, 2)  # unweighted, for order
        dtype = q.dtype
        votes, raw = votes.to(dtype), raw.to(dtype)
        values = torch.cat([votes, raw], dim=-1)
        width = votes.shape[-1]

        # scale=1 with the score half pre-scaled: round 1 is the paper's score.
        q = q * D**-0.5
        keys = torch.cat([k, votes], dim=-1)
        acc = None
        for r in range(self.rounds):
            # The ghost mass feeds the metrics only.
            with_ghost = self.training and r + 1 == self.rounds
            if acc is None:
                out, ghost = attend(q, k, values, 1.0, with_ghost)
            else:
                query = torch.cat([q, acc], dim=-1)
                out, ghost = attend(query, keys, values, 1.0, with_ghost)
            S = out[..., :width]
            whole = _squash(S.float())
            if r == 0:
                first = S
            if r + 1 < self.rounds:
                acc = whole.to(dtype) if acc is None else acc + whole.to(dtype)

        # Back to the token's frame: G = conj(rot) sum_n v[n] conj(z)^(n-1).
        v = whole.view(B, H, T, K, 2, m).transpose(1, 2)  # [B, T, H, K, 2, m]
        gr = torch.zeros_like(zr)
        gi = torch.zeros_like(zi)
        cr, ci = torch.ones_like(zr), torch.zeros_like(zi)
        for n in range(K):
            ar, ai = v[..., n, 0, :], v[..., n, 1, :]
            gr, gi = gr + ar * cr - ai * ci, gi + ar * ci + ai * cr
            cr, ci = cr * zr + ci * zi, ci * zr - cr * zi
        Gr, Gi = gr * rc + gi * rs, gi * rc - gr * rs
        attended = torch.cat([Gr, Gi], dim=-1).flatten(2).to(dtype)

        resultant = out[..., width:].float().transpose(1, 2)  # [B, T, H, 2m]
        if self.training:
            self._record_capsules(first, S, whole)
        return super().forward(
            attended,
            state,
            ghost=ghost,
            resultant=(resultant[..., :m], resultant[..., m:]),
        )

    def _record_capsules(self, first: Tensor, last: Tensor, whole: Tensor) -> None:
        with torch.no_grad():
            cos = F.cosine_similarity(first.float(), last.float(), dim=-1, eps=1e-6)
            length = whole.float().pow(2).sum(-1).sqrt()
            self._capsule_stats.copy_(torch.stack([1 - cos.mean(), length.mean()]))

    def training_metrics(self) -> dict:
        out = super().training_metrics()
        with torch.no_grad():
            energy = self.vote_weight.float().pow(2).sum(dim=(0, 1, 3))
            out["capsule_vote_harmonic_share"] = float(
                energy[1:].sum() / energy.sum().clamp_min(1e-12)
            )
            shift, length = self._capsule_stats.tolist()
            if not math.isnan(shift):
                out["capsule_routing_shift"] = shift
                out["capsule_whole_length"] = length
        return out
