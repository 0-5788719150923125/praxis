"""AbstractinatorCALM: a continuous CALM arm beside the discrete RVQ arm.

The thesis under test is that the DISCRETE arm can pay for the CONTINUOUS one.
CALM's energy score is a weak, high-variance signal that needs far more tokens
than this line can afford; an RVQ code is a dense, low-variance, mode-seeking
target, and predicting the next code from the same conditioning hidden the
energy generator reads is what should concentrate its conditional.

These tests pin the mechanics, not the thesis - the run decides that.
"""

import pytest
import torch

from praxis import PraxisConfig
from praxis.encoders.abstractinator import AbstractinatorCALM
from praxis.modeling import PraxisForCausalLM

PROFILE = "abstractinator_v1_calm"
PARENT = "abstractinator_v1"


def cfg(encoder=PROFILE, d=64):
    return PraxisConfig(
        vocab_size=1024,
        hidden_size=d,
        embed_size=d,
        num_heads=2,
        depth=2,
        max_length=512,
        decoder_type="sequential",
        classifier_type="forward",
        encoder_type=encoder,
        tokenizer_type="byte_level",
        codebook_size=256,
    )


def build(encoder=PROFILE, seed=0):
    torch.manual_seed(seed)
    return PraxisForCausalLM(cfg(encoder)).train()


def test_the_encoder_declares_its_own_modes():
    """An encoder with exactly one decoding path names it, so the run never has
    to. CALM only ever decodes by vote; a plain byte-latent encoder drives no
    custom path at all."""
    from praxis.encoders.abstractinator import AbstractinatorCALM
    from praxis.encoders.base import BaseEncoder
    from praxis.encoders.calm.encoder import CALMEncoder

    assert CALMEncoder.generation_modes == ("vote",)
    assert CALMEncoder.default_generation_mode == "vote"
    assert AbstractinatorCALM.generation_modes == ("standard", "vote")
    assert AbstractinatorCALM.default_generation_mode == "standard"
    assert BaseEncoder.generation_modes == ()


def test_an_unsupported_mode_fails_loudly_at_build():
    """Silently decoding the other way is the failure to avoid."""
    m = build()
    with pytest.raises(ValueError, match="supports generation_mode"):
        m.encoder.resolve_generation_mode("nonsense")
    # An encoder with no custom path rejects a mode it cannot drive.
    parent = build(PARENT)
    assert parent.encoder.resolve_generation_mode(None) == "standard"
    with pytest.raises(ValueError, match="no custom generation path"):
        parent.encoder.resolve_generation_mode("vote")
