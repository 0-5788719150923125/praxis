"""Phase-coupling stages for SelfAttention: Kuramoto Attention (arXiv:2606.11585).

``PhaseProjection`` reads the attention input as a bank of phase oscillators
and lifts it into queries, keys and values. ``TangentStep`` turns the attended
values into a synchronization step. ``SelfAttention`` takes them through its
``projection`` and ``output`` arguments, and the ``kuramoto`` attention profile
composes the two.
"""

import math
from typing import NamedTuple, Optional

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
        self, attended: Tensor, state: PhaseState, ghost: Optional[Tensor] = None
    ) -> Tensor:
        """``attended``: ``[B, T, H * head_size]``. ``ghost``: ``[B, H, T]``, the
        attention mass on the ghost key, used for the metrics only."""
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
            self._record(re, im, ghost, gate, step, scale)
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
