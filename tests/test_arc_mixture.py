"""Test suite for the ArcMixture router (cyclic mixture-of-depths)."""

import pytest
import torch
import torch.nn as nn

from praxis.configuration import PraxisConfig
from praxis.routers import ROUTER_REGISTRY
from praxis.routers.arc import ArcMixture
from praxis.routers.mixture_of_depths import MixtureOfDepths


class MockLayer(nn.Module):
    """A layer that doubles its inputs and accepts the MoD call signature.

    The router may pass a trailing ``token_weights`` arg when it routes a
    subset of tokens, so we swallow *args/**kwargs.
    """

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_seq_len = None

    def forward(self, inputs, *args, **kwargs):
        self.call_count += 1
        self.last_seq_len = inputs.shape[1]
        return inputs * 2, None, None, 0.0


def make_config(**overrides):
    config = dict(hidden_size=64, depth=8, debug=False)
    config.update(overrides)
    return PraxisConfig(**config)


class TestArcMixtureRegistration:
    """Registry and construction wiring."""

    def test_registered_under_arc_mixture(self):
        assert "arc_mixture" in ROUTER_REGISTRY
        assert ROUTER_REGISTRY["arc_mixture"] is ArcMixture

    def test_is_mixture_of_depths_subclass(self):
        assert issubclass(ArcMixture, MixtureOfDepths)

    def test_construction_via_registry_ignores_experts_kwarg(self):
        # LocalLayer constructs routers with experts=expert_blocks; ArcMixture
        # should accept and ignore it like MixtureOfDepths does.
        config = make_config()
        router = ROUTER_REGISTRY["arc_mixture"](config, experts=None)
        assert isinstance(router, ArcMixture)


class TestArcMixtureLayout:
    """The default ``arc`` layout: odd layers sparse, even layers full."""

    def test_arc_layout_alternates_full_and_quarter(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        # Even depths run full (1.0); odd depths route 25% (75% sparsity).
        assert router.capacities == [1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25]

    def test_odd_layer_routes_quarter_of_tokens(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        layer = MockLayer()
        inputs = torch.randn(2, 16, config.hidden_size)

        # Odd depth -> capacity 0.25 -> k = 16 * 0.25 = 4 tokens routed.
        output, _, _, _ = router(
            layer, inputs, None, None, None, current_depth=1, block_ids=None
        )
        assert layer.last_seq_len == 4
        assert output.shape == inputs.shape

    def test_even_layer_processes_all_tokens(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        layer = MockLayer()
        inputs = torch.randn(2, 16, config.hidden_size)

        # Even depth -> capacity 1.0 -> full passthrough, all tokens doubled.
        output, _, _, _ = router(
            layer, inputs, None, None, None, current_depth=0, block_ids=None
        )
        assert layer.last_seq_len == 16
        assert torch.allclose(output, inputs * 2)


class TestArcMixtureDepthBias:
    """The per-recurrent-depth router bias - the Arc mechanism."""

    def test_bias_table_sized_to_depth_and_zero_init(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        assert router.depth_router_bias.weight.shape == (8, 1)
        # Zero-init: starts identical to MixtureOfDepths.
        assert torch.allclose(
            router.depth_router_bias.weight, torch.zeros(8, 1)
        )

    def test_logits_match_base_at_zero_init(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        inputs = torch.randn(2, 16, config.hidden_size)

        base_logits = nn.functional.linear(inputs, router.weight, router.bias)
        arc_logits = router._compute_router_logits(inputs, current_depth=3)
        assert torch.allclose(arc_logits, base_logits)

    def test_bias_is_depth_specific_and_added(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        with torch.no_grad():
            router.depth_router_bias.weight[3].fill_(5.0)
            router.depth_router_bias.weight[4].fill_(-2.0)

        inputs = torch.randn(2, 16, config.hidden_size)
        base = nn.functional.linear(inputs, router.weight, router.bias)

        logits_d3 = router._compute_router_logits(inputs, current_depth=3)
        logits_d4 = router._compute_router_logits(inputs, current_depth=4)
        logits_d0 = router._compute_router_logits(inputs, current_depth=0)

        assert torch.allclose(logits_d3, base + 5.0)
        assert torch.allclose(logits_d4, base - 2.0)
        assert torch.allclose(logits_d0, base)  # untouched depth stays at base

    def test_bias_receives_gradient_through_routing(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        router.train()
        layer = MockLayer()
        inputs = torch.randn(2, 16, config.hidden_size, requires_grad=True)

        # current_depth=1 -> sparse path, where router logits (and thus the
        # per-depth bias) feed the aux loss.
        _, _, _, aux_loss = router(
            layer, inputs, None, None, None, current_depth=1, block_ids=None
        )
        aux_loss.backward()

        grad = router.depth_router_bias.weight.grad
        assert grad is not None
        # Only the active depth row should receive gradient.
        assert not torch.allclose(grad[1], torch.zeros(1))
        assert torch.allclose(grad[0], torch.zeros(1))


class TestArcMixtureMetrics:
    """Depth-specialization telemetry."""

    def test_metrics_empty_at_zero_init(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        # Zero-init -> specialization is 0.0 (collapsed/identical rows).
        metrics = router.training_metrics()
        assert metrics["arc_router_specialization"] == pytest.approx(0.0)

    def test_metrics_rise_when_depths_diverge(self):
        config = make_config(depth=8)
        router = ArcMixture(config)
        with torch.no_grad():
            router.depth_router_bias.weight.copy_(
                torch.linspace(-3, 3, 8).unsqueeze(-1)
            )
        metrics = router.training_metrics()
        assert metrics["arc_router_specialization"] > 0.5

    def test_collector_picks_up_arc_mixture(self):
        from praxis.metrics.specialization import collect_arc_metrics

        config = make_config(depth=8)
        # Wrap the router in a parent module so the collector walks into it.
        parent = nn.Module()
        parent.router = ArcMixture(config)
        with torch.no_grad():
            parent.router.depth_router_bias.weight.copy_(
                torch.linspace(-3, 3, 8).unsqueeze(-1)
            )
        metrics = collect_arc_metrics(parent)
        assert "arc_router_specialization" in metrics


class TestArcMixtureIntegration:
    """The full LocalLayer -> ArcMixture -> TransformerBlock call chain."""

    def _build_layer(self, depth=8):
        from praxis.blocks.transformer import TransformerBlock
        from praxis.layers import LocalLayer

        config = make_config(
            num_heads=8, depth=depth, num_layers=2, router_type="arc_mixture"
        )
        block = TransformerBlock(config)
        return LocalLayer(config, block=block), config

    def test_local_layer_selects_arc_mixture(self):
        layer, _ = self._build_layer()
        assert isinstance(layer.router, ArcMixture)
        assert layer.router.capacities == [1.0, 0.25] * 4

    def test_forward_through_full_chain_even_and_odd(self):
        layer, config = self._build_layer()
        x = torch.randn(2, 16, config.hidden_size)
        for depth in (0, 1, 2, 3):
            out, _, _, _, _ = layer(
                x, None, None, None, current_depth=depth, block_ids=None
            )
            assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
