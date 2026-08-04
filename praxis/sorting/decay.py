import math

import torch
import torch.nn as nn

from .base import NoSort, register_sorting

# Bounded, learned decay length in POSITIONS, parameterized in log space (the
# RoPE learned-theta idiom) so it can neither collapse to zero nor run away:
# tau = TAU_MIN * (TAU_MAX/TAU_MIN)^sigmoid(z). TAU_INIT sets the init.
TAU_MIN: float = 8.0
TAU_MAX: float = 8192.0
TAU_INIT: float = 128.0


def bounded_tau(z: torch.Tensor) -> torch.Tensor:
    """Map the raw log-space parameter to a decay length in ``[TAU_MIN, TAU_MAX]``.

    Written as ``exp(sigmoid(z) * log(span))`` rather than ``span**sigmoid(z)``:
    a scalar-base power's backward needs ``log(base)``, which inductor cannot
    lower once dynamic shapes make the scalar symbolic (same reason as
    ``RoPE._effective_theta``).
    """
    return TAU_MIN * torch.exp(math.log(TAU_MAX / TAU_MIN) * torch.sigmoid(z))


def tau_logit(tau: float) -> float:
    """The ``z`` that lands :func:`bounded_tau` on ``tau``."""
    s = math.log(tau / TAU_MIN) / math.log(TAU_MAX / TAU_MIN)
    return math.log(s / (1.0 - s))


@register_sorting("decay_bias")
class DecayBiasSort(NoSort):
    """Additive rank-1 positional bias with an absolute-position decay envelope.

    Not a sort - a differentiable positional bias that reuses the sorting slot.
    Adds ``g(t) * v`` to each position, where ``v`` is a learnable ``[H]`` bias
    direction (zero-init, so it starts as identity) and ``g(t) = exp(-t/tau)``
    is a monotone decay over ABSOLUTE position: strong bias on the oldest
    context, fading toward 0 over a learned length scale ``tau``, so the recent,
    predictive tokens stay closest to their raw form.

    Why absolute and not ``1 - t/T``: a window-fraction envelope makes a token's
    encoding depend on the sequence length it happened to be batched at. Under
    packed batches, dynamic patching (where ``T`` is the batch's PADDED patch
    count, so batch-mates move it), the sequence-length curriculum, and
    generation (where ``T`` grows every step), the same token at the same
    position lands somewhere different in the field on every pass - and the
    "~0 at the tail" property only ever holds for the longest row in a batch.
    Anchoring to ``t`` makes the position-to-bias map identical in training and
    generation, and a function of the token alone.

    Why additive: unlike a scalar amplitude scale (which a downstream
    per-position norm divides straight back out), an additive per-feature bias
    changes the representation's *direction*, so it survives normalization. The
    model is RoPE/ArcHoPE-only (position enters relatively via attention
    rotation), so this is the one *absolute* positional signal in the stack.
    Minimal by design: ``H + 1`` params, identity at init.

    ``tau`` is learned rather than tuned. Its gradient is zero while ``v`` is
    still zero, so the length scale only starts moving once the bias itself has
    grown - the module picks its own horizon from data.
    """

    metric_descriptions = {
        "sorting/bias_norm": {
            "description": (
                "L2 norm of the sorting slot's learnable positional bias "
                "direction. Zero-init, so this is exactly how far the module "
                "has moved from identity: flat at 0 means the positional bias "
                "is inert. Compare against the decoder's hidden-state scale to "
                "tell a nudge from a shove."
            ),
            "chart": {
                "title": "Positional Bias Field",
                "y_label": "L2 norm",
                "group": "sorting",
                "order": 10,
            },
        },
        "sorting/decay_tau": {
            "description": (
                "Learned decay length (positions) of the additive positional "
                "bias envelope exp(-t/tau): the horizon over which absolute "
                "position still shapes the representation. Bounded to "
                "[8, 8192]. Only moves once the bias norm is nonzero - its "
                "gradient vanishes while the bias is still zero."
            ),
            "chart": {
                "title": "Positional Bias Decay Length",
                "y_label": "positions",
                "y_scale": "logarithmic",
                "group": "sorting",
                "order": 20,
            },
        },
    }

    def __init__(self, config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.log_tau = nn.Parameter(torch.full((1,), tau_logit(TAU_INIT)))

    def _applies(self, hidden_states: torch.Tensor) -> bool:
        """Defensive: only apply on a non-empty sequence of the right width."""
        return (
            hidden_states.shape[-2] != 0
            and self.bias.shape[-1] == hidden_states.shape[-1]
        )

    def _positions(self, hidden_states: torch.Tensor, offset: int) -> torch.Tensor:
        """Absolute positions ``[T]`` for this chunk, in fp32.

        ``offset`` is the number of tokens already consumed - cached decode
        feeds only the new suffix, so the field has to continue from where it
        left off instead of restarting at 0 on every generated token.
        """
        t = torch.arange(
            hidden_states.shape[-2], device=hidden_states.device, dtype=torch.float32
        )
        return t + float(offset)

    def _decay(self, t: torch.Tensor) -> torch.Tensor:
        """``exp(-t/tau)`` over absolute position, in fp32."""
        return torch.exp(-t / bounded_tau(self.log_tau).to(t.dtype))

    def _add_bias(self, hidden_states: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        g = self._decay(t).to(dtype)  # [T]
        return hidden_states + g.unsqueeze(-1) * self.bias.to(dtype)

    def forward(
        self, hidden_states: torch.Tensor, offset: int = 0, **kwargs
    ) -> torch.Tensor:
        if not self._applies(hidden_states):
            return hidden_states  # defensive: only apply when the dim matches
        return self._add_bias(hidden_states, self._positions(hidden_states, offset))

    def training_metrics(self) -> dict:
        with torch.no_grad():
            return {
                "sorting/bias_norm": float(self.bias.detach().float().norm()),
                "sorting/decay_tau": float(bounded_tau(self.log_tau.detach().float())),
            }
