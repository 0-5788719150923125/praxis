import torch
import torch.nn as nn

from .base import NoSort, register_sorting


@register_sorting("decay_bias")
class DecayBiasSort(NoSort):
    """Additive rank-1 positional bias with a tail-decay amplitude envelope.

    Not a sort - a differentiable positional bias that reuses the sorting slot.
    Adds ``g(t) * v`` to each position, where ``v`` is a learnable ``[H]`` bias
    direction (zero-init, so it starts as identity) and ``g(t) = 1 - t/T`` is a
    monotone decay: strong bias on the oldest context (head), fading to ~0 at
    the tail so the recent, predictive tokens stay closest to their raw form.

    Why additive: unlike a scalar amplitude scale (which a downstream
    per-position norm divides straight back out), an additive per-feature bias
    changes the representation's *direction*, so it survives normalization. The
    model is RoPE/ArcHoPE-only (position enters relatively via attention
    rotation), so this is the one *absolute* positional signal in the stack.
    Minimal by design: ``H`` params, identity at init. The frozen linear
    envelope could later gain a learnable decay exponent.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        seq_len = hidden_states.shape[-2]
        if seq_len == 0 or self.bias.shape[-1] != hidden_states.shape[-1]:
            return hidden_states  # defensive: only apply when the dim matches
        t = torch.arange(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        g = 1.0 - t / seq_len  # [T]: 1 at the head -> ~0 at the tail
        bias = g.unsqueeze(-1) * self.bias.to(hidden_states.dtype)  # [T, H]
        return hidden_states + bias
