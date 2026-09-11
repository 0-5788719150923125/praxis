"""Prismatic, the per-position top-k router over encoding-diverse experts
(praxis/routers/prismatic.py).

The experts' encoding cycling (alibi, rope, alibi, ...) is done by the decoder
that builds them; tests/decoders/test_base.py covers it.
"""

import copy

import pytest
import torch
import torch.nn as nn

from praxis.attention.causal import CausalAttention
from praxis.configuration import PraxisConfig
from praxis.routers.prismatic import Prismatic


def _config(num_experts=2):
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        num_experts=num_experts,
        dropout=0.0,
        router_type="prismatic",
    )


class SimpleDenseBlock(nn.Module):
    """A block-shaped expert: Linear then LayerNorm."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=0,
        block_ids=None,
        **kwargs,
    ):
        return self.norm(self.linear(inputs)), past_key_values, current_state, 0.0


def _router(num_experts=2):
    config = _config(num_experts)
    experts = [SimpleDenseBlock(config.hidden_size) for _ in range(num_experts)]
    return Prismatic(config, experts=experts)


def test_routing_is_one_distribution_per_position():
    router = _router()
    probs = router._compute_routing(torch.randn(2, 16, 64))
    # One distribution per POSITION, read from its prefix mean.
    assert probs.shape == (2, 16, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 16))


def test_gradients_flow_to_router():
    router = _router()
    inputs = torch.randn(2, 4, 64)
    # The seven positional args LocalLayer passes; the layer slot is unused.
    output, _, _, aux_loss = router(None, inputs, None, None, None, 0, None)
    (output.sum() + aux_loss).backward()

    assert router.router.weight.grad is not None
    assert router.router.weight.grad.abs().sum() > 0


def test_balance_loss_is_one_when_balanced_and_larger_when_not():
    router = _router()

    # importance=[0.5, 0.5], load=[0.5, 0.5] -> 2 * (0.25 + 0.25) = 1.0
    balanced = router._compute_balance_loss(
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]), torch.tensor([[0, 1], [1, 0]])
    )
    assert balanced.item() == pytest.approx(1.0, abs=0.01)

    # importance=[0.9, 0.1], load=[1.0, 0.0] -> 2 * 0.9 = 1.8
    imbalanced = router._compute_balance_loss(
        torch.tensor([[0.9, 0.1], [0.9, 0.1]]), torch.tensor([[0, 0], [0, 0]])
    )
    assert imbalanced.item() > balanced.item()


def test_metrics_report_weights_usage_and_cumulative_selection_counts():
    router = _router(num_experts=4)
    assert router.expert_selection_counts.tolist() == [0, 0, 0, 0]

    # [N, k] top-2 indices, the shape the forward passes.
    indices = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2]])
    probs = torch.softmax(torch.randn(4, 4), dim=-1)
    router._update_metrics(indices, probs, torch.tensor(0.01))
    metrics = router.get_metrics()

    for key in ("entropy", "concentration", "variance", "balance"):
        assert f"routing/{key}" in metrics
    weights = [metrics[f"routing/expert_{i}_weight"] for i in range(4)]
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)
    usage = metrics["architecture/alibi_usage"] + metrics["architecture/rope_usage"]
    assert usage == pytest.approx(100.0, abs=1e-3)

    counts = [3, 2, 2, 1]
    for i, count in enumerate(counts):
        assert metrics[f"expert_selection/expert_{i}_count"] == count
        assert metrics[f"routing/expert_{i}_load"] == pytest.approx(count / 8)

    # Selection counts accumulate across calls.
    router._update_metrics(indices, probs, torch.tensor(0.01))
    metrics = router.get_metrics()
    for i, count in enumerate(counts):
        assert metrics[f"expert_selection/expert_{i}_count"] == 2 * count


# Exercises praxis/attention/causal.py; belongs in tests/attention/test_causal.py.
def test_rope_and_alibi_produce_different_outputs():
    """Same weights, different encoding: the outputs must differ."""
    config = _config()
    rope_config, alibi_config = copy.copy(config), copy.copy(config)
    rope_config.encoding = "rope"
    alibi_config.encoding = "alibi"
    attention_rope = CausalAttention(rope_config)
    attention_alibi = CausalAttention(alibi_config)
    attention_alibi.load_state_dict(attention_rope.state_dict(), strict=False)

    inputs = torch.randn(2, 16, config.hidden_size)
    output_rope, _, _ = attention_rope(inputs)
    output_alibi, _, _ = attention_alibi(inputs)
    assert not torch.allclose(output_rope, output_alibi, atol=1e-3)
