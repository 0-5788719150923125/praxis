"""Sweeps over every entry of the ``encoders`` registry."""

import pytest
import torch

from praxis import registry
from praxis.modeling import resolve_head_type

ENCODER_KEYS = sorted(registry.namespace("encoders"))


@pytest.fixture(params=ENCODER_KEYS)
def module_setup(request, config):
    setattr(config, "device_map", "cpu")
    module = registry.lookup("encoders", request.param)(config)
    # Mirror PraxisModel: build and inject input embeddings from the registry
    # for encoders that name an embedding profile.
    profile = getattr(module, "embedding_profile", None)
    if profile:
        module.set_embeddings(
            registry.lookup("embeddings", profile)(config, encoder=module)
        )
    # Mirror PraxisForCausalLM again: loss-owning encoders (CALM) do not own a
    # token classifier, they borrow the LM head and apply it internally, so
    # decode() cannot classify until the head is injected.
    if hasattr(module, "set_head"):
        head_type = resolve_head_type(config, has_encoder=True)
        head_cls = registry.namespace("heads").get(
            head_type, registry.lookup("heads", "forward")
        )
        module.set_head(head_cls(config, encoder=module))
    return module, config


def test_forward_pass(module_setup):
    """Every listed encoder encodes and decodes back to one feature (or logit)
    vector per input token."""
    module, config = module_setup
    # Create sample input
    batch_size = 2
    seq_len = 16  # Should be less than max_seq_len (512)
    # For ByteLatent encoder, vocab_size is 260 (256 bytes + 4 special tokens)
    # Use a smaller value for input_ids to avoid out of bounds
    input_vocab_size = 256  # Max value for input IDs
    input_ids = torch.randint(0, input_vocab_size, (batch_size, seq_len))

    # Step 1: Encode (returns 6 values)
    h, h_encoder, patch_lengths, block_ids, entropy_loss, local_decoder_tokens = (
        module.encode(input_ids=input_ids)
    )

    # Step 2: Decode
    logits, decoder_embeds = module.decode(
        h,
        h_encoder,
        input_ids,
        patch_lengths,
        local_decoder_tokens,
    )

    # ByteLatent decode now returns features only (the LM head owns
    # classification, so logits is None); its features are dim_token_emb.
    # Encoders that own their output (CALM, etc.) still emit aligned logits.
    if hasattr(module, "byte_config"):
        assert logits is None, "Byte-latent decode should not emit logits"
        out, expected_dim = decoder_embeds, module.byte_config.dim_token_emb
    else:
        out, expected_dim = logits, config.vocab_size
    assert len(out.shape) == 3, "Expected 3D output from decoder"
    assert out.shape == (batch_size, seq_len, expected_dim)


# ------------------------------------------------------- registry names


def test_unlisted_encoder_names_resolve_but_are_not_listed():
    """Descriptive names build the same profiles as the listed names they map
    to; only the listed names reach the CLI choices and the docs."""

    listed = set(registry.namespace("encoders"))
    for name, target in registry.namespace("encoders").aliases().items():
        assert name in registry.namespace("encoders") and name not in listed, name
        assert target in listed, target
        assert registry.lookup("encoders", name) is registry.lookup("encoders", target)
    for name, profile in registry.namespace("encoders").unlisted().items():
        assert name in registry.namespace("encoders") and name not in listed, name
        assert registry.namespace("encoders").get(name) is profile
    assert registry.namespace("encoders").get("no_such_encoder") is None
