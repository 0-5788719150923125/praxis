"""Sweep over the blocks registry: every block runs forward and backward."""

import itertools
import math

import pytest
import torch

from praxis import PraxisConfig, registry

BLOCK_KEYS = list(registry.namespace("blocks"))
HIDDEN_SIZES = [64, 128]
NUM_HEADS = [1, 2]
# mru's state head (hidden_size / num_heads) must be a perfect square, and it
# needs embed_size == hidden_size.
SQUARE_BLOCKS = {"mru"}


def _square(hidden_size, num_heads):
    return math.isqrt(hidden_size // num_heads) ** 2 == hidden_size // num_heads


CASES = [
    (key, h, n)
    for key, h, n in itertools.product(BLOCK_KEYS, HIDDEN_SIZES, NUM_HEADS)
    if key not in SQUARE_BLOCKS or _square(h, n)
]


def test_every_block_is_swept():
    assert {key for key, _, _ in CASES} == set(BLOCK_KEYS)


@pytest.mark.parametrize("key,hidden_size,num_heads", CASES)
def test_forward_and_backward(key, hidden_size, num_heads):
    extra = {"embed_size": hidden_size} if key in SQUARE_BLOCKS else {}
    config = PraxisConfig(hidden_size=hidden_size, num_heads=num_heads, **extra)
    module = registry.lookup("blocks", key)(config)

    x = torch.randn(4, 16, hidden_size, requires_grad=True)
    output, _, _, _ = module(
        x, attention_mask=None, router_weights=None, current_state=None, current_depth=0
    )
    assert output.shape == x.shape

    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
