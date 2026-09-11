"""Fixtures shared by the losses tests: a tiny full model and one training forward."""

import pytest
import torch

# The smallest model that carries a harmonic field (prismatic5) and a HALO main
# loss - what dissonance, harmonic_kl and the objectives container act on.
TINY_MODEL = dict(
    vocab_size=1024,
    hidden_size=32,
    embed_size=96,
    num_heads=4,
    num_layers=1,
    depth=2,
    tokenizer_type="byte_level",
    decoder_type="sequential",
    head_type="prismatic5",
    residual_type="smear",
    loss_func="halo",
)


@pytest.fixture
def tiny_model():
    """Factory: ``tiny_model(**overrides)`` builds a seeded PraxisForCausalLM
    from TINY_MODEL in training mode."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    def build(**overrides):
        torch.manual_seed(0)
        return PraxisForCausalLM(PraxisConfig(**{**TINY_MODEL, **overrides})).train()

    return build


@pytest.fixture
def train_step():
    """Factory: ``train_step(model)`` runs one labelled forward on random bytes
    and returns the output."""

    def step(model):
        ids = torch.randint(4, 900, (2, 24))
        return model(input_ids=ids, labels=ids[..., 1:].contiguous())

    return step
