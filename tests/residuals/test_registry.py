"""Sweep over the residuals registry: every entry runs a width-then-depth
connection and returns the input shape. Residuals are built from hidden_size
alone, so that is the only axis."""

import itertools

import pytest
import torch
from torch import nn

from praxis import registry

RESIDUAL_KEYS = list(registry.namespace("residuals"))


@pytest.mark.parametrize(
    "key,hidden_size", list(itertools.product(RESIDUAL_KEYS, [64, 128]))
)
def test_forward_pass(key, hidden_size):
    module = registry.lookup("residuals", key)(hidden_size)
    x = torch.randn(4, 16, hidden_size)

    residual, beta = module.connect_width(x)
    y = nn.Identity()(module.format_state(residual))
    merged = module.connect_depth(residual, y, beta)
    assert module.format_state(merged).shape == x.shape
