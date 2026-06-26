"""Test suite for the ArcMixture router (cyclic mixture-of-depths).

ArcMixture keys capacity to the *physical layer* index (current_depth %
num_layers), so a given layer is the routed one on every recurrent pass, and
keys an additive router bias to the *recurrent pass* (current_depth //
num_layers). This mirrors the ArcGLU idiom; see praxis/routers/arc.py.
"""

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
    # calm-d shape: depth 9, 3 physical layers -> 3 recurrent passes each.
    config = dict(hidden_size=64, depth=9, num_layers=3, debug=False)
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
    """Capacity keyed to physical layer, not the flattened depth."""

    def test_capacities_sized_to_num_layers(self):
        config = make_config(num_layers=3)
        router = ArcMixture(config)
        # Length == num_layers (not depth); odd layer (index 1) is the routed one.
        assert router.capacities == [1.0, 0.25, 1.0]

    def test_only_odd_layer_routes_across_all_passes(self):
        """The 2nd layer (index 1) is sparse on EVERY recurrent pass; the
        even layers run full on every pass. This is the property the global
        depth-indexed version got wrong."""
        config = make_config(depth=9, num_layers=3)
        router = ArcMixture(config)
        layer = MockLayer()
        inputs = torch.randn(2, 16, config.hidden_size)

        for current_depth in range(9):
            layer.last_seq_len = None
            out, _, _, _ = router(
                layer, inputs, None, None, None, current_depth, block_ids=None
            )
            assert out.shape == inputs.shape
            physical_layer = current_depth % 3
            if physical_layer == 1:
                # capacity 0.25 -> routes int(16 * 0.25) = 4 tokens
                assert layer.last_seq_len == 4, f"depth {current_depth}"
            else:
                # capacity 1.0 -> full passthrough, all tokens doubled
                assert layer.last_seq_len == 16, f"depth {current_depth}"
                assert torch.allclose(out, inputs * 2)

    def test_capacity_for_uses_modulo(self):
        config = make_config(num_layers=3)
        router = ArcMixture(config)
        # Same physical layer -> same capacity across passes.
        assert router._capacity_for(1) == router._capacity_for(4) == 0.25
        assert router._capacity_for(7) == 0.25
        assert router._capacity_for(0) == router._capacity_for(3) == 1.0
        assert router._capacity_for(2) == router._capacity_for(5) == 1.0


class TestArcMixturePassBias:
    """The per-recurrent-pass router bias - the Arc mechanism."""

    def test_bias_table_sized_to_num_passes_and_zero_init(self):
        config = make_config(depth=9, num_layers=3)
        router = ArcMixture(config)
        # ceil(9 / 3) = 3 recurrent passes.
        assert router.num_passes == 3
        assert router.depth_router_bias.weight.shape == (3, 1)
        assert torch.allclose(router.depth_router_bias.weight, torch.zeros(3, 1))

    def test_logits_match_base_at_zero_init(self):
        config = make_config()
        router = ArcMixture(config)
        inputs = torch.randn(2, 16, config.hidden_size)

        base_logits = nn.functional.linear(inputs, router.weight, router.bias)
        arc_logits = router._compute_router_logits(inputs, current_depth=4)
        assert torch.allclose(arc_logits, base_logits)

    def test_bias_is_pass_specific_and_added(self):
        config = make_config(depth=9, num_layers=3)
        router = ArcMixture(config)
        with torch.no_grad():
            router.depth_router_bias.weight[0].fill_(5.0)  # pass 0
            router.depth_router_bias.weight[1].fill_(-2.0)  # pass 1

        inputs = torch.randn(2, 16, config.hidden_size)
        base = nn.functional.linear(inputs, router.weight, router.bias)

        # Layer 1 is revisited at depth 1 (pass 0), 4 (pass 1), 7 (pass 2).
        logits_pass0 = router._compute_router_logits(inputs, current_depth=1)
        logits_pass1 = router._compute_router_logits(inputs, current_depth=4)
        logits_pass2 = router._compute_router_logits(inputs, current_depth=7)

        assert torch.allclose(logits_pass0, base + 5.0)
        assert torch.allclose(logits_pass1, base - 2.0)
        assert torch.allclose(logits_pass2, base)  # untouched pass stays at base

    def test_bias_receives_gradient_through_routing(self):
        config = make_config(depth=9, num_layers=3)
        router = ArcMixture(config)
        router.train()
        layer = MockLayer()
        inputs = torch.randn(2, 16, config.hidden_size, requires_grad=True)

        # depth 4 -> physical layer 1 (sparse path) -> pass index 1.
        _, _, _, aux_loss = router(
            layer, inputs, None, None, None, current_depth=4, block_ids=None
        )
        aux_loss.backward()

        grad = router.depth_router_bias.weight.grad
        assert grad is not None
        # Only the active pass row should receive gradient.
        assert not torch.allclose(grad[1], torch.zeros(1))
        assert torch.allclose(grad[0], torch.zeros(1))
        assert torch.allclose(grad[2], torch.zeros(1))


class TestArcMixtureMetrics:
    """Pass-specialization telemetry."""

    def test_metrics_empty_at_zero_init(self):
        config = make_config()
        router = ArcMixture(config)
        # Zero-init -> specialization is 0.0 (collapsed/identical rows).
        metrics = router.training_metrics()
        assert metrics["arc_router_specialization"] == pytest.approx(0.0)

    def test_metrics_rise_when_passes_diverge(self):
        config = make_config(depth=9, num_layers=3)
        router = ArcMixture(config)
        with torch.no_grad():
            router.depth_router_bias.weight.copy_(
                torch.tensor([[-3.0], [0.0], [3.0]])
            )
        metrics = router.training_metrics()
        assert metrics["arc_router_specialization"] > 0.5

    def test_metrics_empty_with_single_pass(self):
        # depth <= num_layers -> 1 pass -> nothing to measure.
        config = make_config(depth=3, num_layers=3)
        router = ArcMixture(config)
        assert router.num_passes == 1
        assert router.training_metrics() == {}

    def test_collector_picks_up_arc_mixture(self):
        from praxis.metrics.specialization import collect_arc_metrics

        config = make_config(depth=9, num_layers=3)
        parent = nn.Module()
        parent.router = ArcMixture(config)
        with torch.no_grad():
            parent.router.depth_router_bias.weight.copy_(
                torch.tensor([[-3.0], [0.0], [3.0]])
            )
        metrics = collect_arc_metrics(parent)
        assert "arc_router_specialization" in metrics


class TestArcMixtureIntegration:
    """The full LocalLayer -> ArcMixture -> TransformerBlock call chain."""

    def _build_layer(self, depth=9, num_layers=3):
        from praxis.blocks.transformer import TransformerBlock
        from praxis.layers import LocalLayer

        config = make_config(
            num_heads=8, depth=depth, num_layers=num_layers, router_type="arc_mixture"
        )
        block = TransformerBlock(config)
        return LocalLayer(config, block=block), config

    def test_local_layer_selects_arc_mixture(self):
        layer, _ = self._build_layer()
        assert isinstance(layer.router, ArcMixture)
        assert layer.router.capacities == [1.0, 0.25, 1.0]

    def test_forward_through_full_chain_all_layers(self):
        layer, config = self._build_layer()
        x = torch.randn(2, 16, config.hidden_size)
        for current_depth in range(config.depth):
            out, _, _, _, _ = layer(
                x, None, None, None, current_depth=current_depth, block_ids=None
            )
            assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
