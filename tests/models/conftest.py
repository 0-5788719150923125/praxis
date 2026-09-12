"""A tiny published-shaped model, built locally so the tests need no network."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from praxis import PraxisConfig

VOCAB = 128
HIDDEN = 32


@pytest.fixture(scope="module")
def hosted():
    """A `LlamaForCausalLM` standing in for anything loaded from the hub."""
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=HIDDEN * 2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


@pytest.fixture
def praxis_config():
    """The run-side config a foreign wrapper reads its objectives from."""
    return PraxisConfig(vocab_size=VOCAB, hidden_size=HIDDEN, embed_size=HIDDEN)


@pytest.fixture
def foreign(hosted, praxis_config):
    """A fresh wrapper over a fresh copy of the hosted model."""
    import copy

    from praxis.models import ForeignCausalLM

    return ForeignCausalLM(copy.deepcopy(hosted), praxis_config, model_id="test/tiny")


@pytest.fixture
def batch():
    """``(input_ids, labels)`` in the Praxis convention: labels pre-shifted."""
    ids = torch.randint(0, VOCAB, (2, 16))
    return ids, ids[..., 1:].contiguous()
