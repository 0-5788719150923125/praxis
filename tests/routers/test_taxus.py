"""Taxus, the depth-buying early-exit router (praxis/routers/taxus.py)."""

import pytest
import torch
import torch.nn as nn

from praxis.configuration import PraxisConfig
from praxis.containers.loss import LossContainer
from praxis.routers.taxus import Taxus

LOSS_KEYS = ("taxus_entropy", "taxus_usage", "taxus_confidence", "taxus_cost")


def _taxus(hidden_size=64, depth=4, **kwargs):
    config = PraxisConfig(hidden_size=hidden_size, depth=depth, debug=False)
    return Taxus(config, **kwargs)


class CountingLayer(nn.Module):
    """Applies ``fn`` to its inputs, counts calls, and returns ``layer_loss``."""

    def __init__(self, fn=lambda x: x + 1, layer_loss=0.0):
        super().__init__()
        self.fn, self.layer_loss, self.call_count = fn, layer_loss, 0

    def forward(self, inputs, *args, **kwargs):
        self.call_count += 1
        return self.fn(inputs), None, None, self.layer_loss


def _route(taxus, layer, inputs, depth):
    return taxus(
        layer=layer,
        inputs=inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=depth,
        block_ids=None,
    )


def test_no_exit_is_considered_before_the_minimum_layer():
    taxus = _taxus(depth=4, min_exit_layer=2)
    assert len(taxus.exit_gates) == 4
    assert taxus.layer_costs.shape == (4,)

    layer = CountingLayer()
    inputs = torch.randn(2, 8, 64)
    for depth in [0, 1]:
        output, _, _, losses = _route(taxus, layer, inputs, depth)
        assert torch.allclose(output, inputs + 1)
        assert "taxus_entropy" not in losses
        assert "taxus_usage" not in losses


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_auxiliary_losses_are_reported_past_the_minimum_layer(mode):
    taxus = _taxus(
        hidden_size=32,
        depth=8,
        target_depth_ratio=0.5,
        entropy_weight=0.01,
        usage_weight=0.1,
        budget_weight=0.1,
    )
    taxus.train(mode == "train")
    # Keep every gate on "continue" so the layer runs and reports its loss.
    with torch.no_grad():
        for gate in taxus.exit_gates:
            gate[-1].bias.copy_(torch.tensor([10.0, -10.0]))

    layer = CountingLayer(fn=lambda x: x, layer_loss=torch.tensor(0.5))
    inputs = torch.randn(2, 4, 32)
    for depth in [2, 4, 6]:
        _, _, _, losses = _route(taxus, layer, inputs, depth)
        for key in LOSS_KEYS + ("layer",):
            assert key in losses, f"{key} missing at depth {depth}"
        assert losses.get_loss("taxus_usage").item() >= 0


def test_training_runs_the_layer_and_eval_skips_it_when_all_exit():
    taxus = _taxus(min_exit_layer=1, temperature=0.5)
    layer = CountingLayer(fn=lambda x: x * 1.5)
    inputs = torch.randn(2, 8, 64)

    # Training blends through gumbel-softmax, so the layer always runs.
    taxus.train()
    _, _, _, losses = _route(taxus, layer, inputs, 2)
    assert layer.call_count == 1
    assert "taxus_exit_prob" in losses

    # Force every sample to exit: inference then skips the layer entirely.
    taxus.eval()
    layer.call_count = 0
    with torch.no_grad():
        taxus.exit_gates[2][2].weight.fill_(0)
        taxus.exit_gates[2][2].bias.copy_(torch.tensor([-10.0, 10.0]))
    output, _, _, losses = _route(taxus, layer, inputs, 2)
    assert layer.call_count == 0
    assert torch.equal(output, inputs)
    assert losses.get_loss("taxus_should_exit").item() == 1.0


def test_layer_loss_containers_are_merged_into_the_output():
    taxus = _taxus(min_exit_layer=0)
    layer_losses = LossContainer()
    layer_losses.add_loss("custom_loss", 0.25)
    layer_losses.add_loss("another_loss", 0.1)
    layer = CountingLayer(fn=lambda x: x + 0.1, layer_loss=layer_losses)

    taxus.train()  # training never skips the layer
    _, _, _, losses = _route(taxus, layer, torch.randn(2, 8, 64), 2)
    assert losses.get_loss("custom_loss").item() == pytest.approx(0.25)
    assert losses.get_loss("another_loss").item() == pytest.approx(0.1)
