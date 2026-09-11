"""Tests for praxis/utils/tensors.py."""

import pytest
import torch

from praxis.utils import create_block_ids

# A special token closes the block it ends; the next token opens a new one.
BLOCKS = [
    ([[1, 2, 0, 3, 4, 0, 5]], [[1, 1, 1, 2, 2, 2, 3]]),
    ([[1, 2, 0, 3, 0, 4, 0, 5]], [[1, 1, 1, 2, 2, 3, 3, 4]]),
    ([[1, 2, 0, 3, 4], [5, 0, 6, 0, 7]], [[1, 1, 1, 2, 2], [1, 1, 2, 2, 3]]),
    ([[1, 0, 2, 0, 3]], [[1, 1, 2, 2, 3]]),
    ([[0, 0, 0]], [[1, 2, 2]]),
    ([[0, 1, 0]], [[1, 2, 2]]),
    ([[1, 2, 3]], [[1, 1, 1]]),
]


@pytest.mark.parametrize(
    "special_tokens",
    # Production passes a scalar id (sep/eos); lists and tensors take other branches.
    [0, [0], torch.tensor([0])],
    ids=["int", "list", "tensor"],
)
@pytest.mark.parametrize("ids,expected", BLOCKS)
def test_create_block_ids(ids, expected, special_tokens):
    input_ids = torch.tensor(ids)
    block_ids = create_block_ids(input_ids, special_tokens)
    assert block_ids.shape == input_ids.shape
    assert torch.equal(block_ids, torch.tensor(expected))


@pytest.mark.parametrize(
    "special_tokens", [[0, 1], torch.tensor([0, 1])], ids=["list", "tensor"]
)
def test_create_block_ids_with_several_special_tokens(special_tokens):
    block_ids = create_block_ids(torch.tensor([[5, 0, 6, 1, 7]]), special_tokens)
    assert torch.equal(block_ids, torch.tensor([[1, 1, 2, 2, 3]]))
