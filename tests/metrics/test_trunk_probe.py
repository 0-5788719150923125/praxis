"""The trunk swap: bits lost when a row is decoded on another row's trunk output.

The reading exists because magnitude cannot answer the question. A trunk whose
output is large but identical for every input costs nothing to swap, and that is
the case these tests pin against one whose output is actually read.
"""

import math

import pytest
import torch
import torch.nn as nn

from praxis.metrics.trunk_probe import trunk_swap_cost

VOCAB = 32


class _Trunk(nn.Module):
    """Stands in for the decoder: returns a per-row state in a tuple, as the
    real one does."""

    def __init__(self, constant=False):
        super().__init__()
        self.constant = constant

    def forward(self, hidden_states, *args, **kwargs):
        b, t, _ = hidden_states.shape
        if self.constant:
            state = torch.ones(b, t, VOCAB)
        else:
            # Row r announces which row it came from, in one-hot.
            state = torch.zeros(b, t, VOCAB)
            state[torch.arange(b), :, torch.arange(b) % VOCAB] = 1.0
        return state, None, None, None


class _Model(nn.Module):
    """A model whose logits are its trunk's state: swapping the trunk between
    rows therefore changes every prediction, unless the trunk is constant."""

    def __init__(self, constant=False):
        super().__init__()
        self.encoder = nn.Identity()  # only its presence is read
        self.decoder = _Trunk(constant=constant)

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        embeds = torch.zeros(b, t, 4)
        state = self.decoder(embeds)[0]
        # Each row's correct answer is its own index, so a swapped trunk is wrong.
        logits = 10.0 * state
        return type("Out", (), {"logits": logits})()


def _rows(b=4, t=16):
    return torch.stack(
        [torch.full((t,), r % VOCAB, dtype=torch.long) for r in range(b)]
    )


def test_swap_costs_bits_when_the_trunk_is_read():
    cost = float(trunk_swap_cost(_Model(), _rows()))
    assert cost > 1.0


def test_swap_is_free_when_the_trunk_is_the_same_for_every_row():
    cost = float(trunk_swap_cost(_Model(constant=True), _rows()))
    assert cost == pytest.approx(0.0, abs=1e-6)


def test_the_decoder_is_left_as_it_was_found():
    model = _Model()
    forward = model.decoder.forward
    trunk_swap_cost(model, _rows())
    assert model.decoder.forward == forward


def test_it_runs_the_uncompiled_module():
    wrapper = nn.Module()
    wrapper._orig_mod = _Model()
    wrapper.forward = lambda **kwargs: pytest.fail("the compiled module was called")
    assert float(trunk_swap_cost(wrapper, _rows())) > 1.0


def test_it_skips_what_it_cannot_read():
    assert trunk_swap_cost(_Model(), _rows(b=1)) is None  # nothing to swap with
    bare = _Model()
    del bare.encoder
    assert trunk_swap_cost(bare, _rows()) is None  # no separable trunk
