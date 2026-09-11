from itertools import product

import pytest
import torch

from praxis import registry
from praxis.modeling import resolve_head_type

# Define test parameters
MODULE_CLASSES = list(registry.namespace("encoders").values())
META_MODES = [
    ["space", "ngram"],
    ["space"],
    # ["entropy"] # currently fails
]

# Create parameter combinations
MODULE_PARAMS = list(product(MODULE_CLASSES, META_MODES))


@pytest.fixture(params=MODULE_PARAMS)
def module_setup(request, config):
    module_class, meta_mode = request.param
    # Use the update method from our existing config
    setattr(config, "meta", meta_mode)
    setattr(config, "device_map", "cpu")
    module = module_class(config)
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
    """Test using parametrized module and dimensions."""
    module, config = module_setup
    # Create sample input
    batch_size = 32
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


# def test_topk_mean_pooling():
#     """Test the correctness of topk_mean_pooling function."""
#     # Setup a simple test case
#     batch_size = 2
#     seq_len = 6
#     emb_dim = 2
#     max_num_patches = 3
#     k = 2

#     # Create input tensor with known values
#     h = torch.tensor(
#         [
#             # Batch 1
#             [
#                 [1.0, 1.0],  # Patch 0
#                 [2.0, 2.0],  # Patch 0
#                 [3.0, 3.0],  # Patch 1
#                 [4.0, 4.0],  # Patch 1
#                 [5.0, 5.0],  # Patch 2
#                 [6.0, 6.0],
#             ],  # Patch 2
#             # Batch 2
#             [
#                 [2.0, 2.0],  # Patch 0
#                 [4.0, 4.0],  # Patch 0
#                 [6.0, 6.0],  # Patch 1
#                 [8.0, 8.0],  # Patch 1
#                 [10.0, 10.0],  # Patch 2
#                 [12.0, 12.0],
#             ],  # Patch 2
#         ],
#         dtype=torch.float32,
#     )

#     # Define patch assignments
#     patch_ids = torch.tensor(
#         [
#             [0, 0, 1, 1, 2, 2],  # Batch 1
#             [0, 0, 1, 1, 2, 2],  # Batch 2
#         ],
#         dtype=torch.long,
#     )

#     # Call the function
#     result = topk_mean_pooling(h, max_num_patches, k, patch_ids)

#     # Expected results (mean of top-k values in each patch):
#     # Batch 1:
#     # - Patch 0: mean of [1.0, 2.0] = [1.5, 1.5]
#     # - Patch 1: mean of [3.0, 4.0] = [3.5, 3.5]
#     # - Patch 2: mean of [5.0, 6.0] = [5.5, 5.5]
#     # Batch 2:
#     # - Patch 0: mean of [2.0, 4.0] = [3.0, 3.0]
#     # - Patch 1: mean of [6.0, 8.0] = [7.0, 7.0]
#     # - Patch 2: mean of [10.0, 12.0] = [11.0, 11.0]
#     expected = torch.tensor(
#         [[[1.5, 1.5], [3.5, 3.5], [5.5, 5.5]], [[3.0, 3.0], [7.0, 7.0], [11.0, 11.0]]],
#         dtype=torch.float32,
#     )

#     # Verify results
#     assert torch.allclose(result, expected, rtol=1e-5), (
#         f"Mismatch in topk_mean_pooling results.\n"
#         f"Got:\n{result}\n"
#         f"Expected:\n{expected}"
#     )

#     # Add variable patch size test
#     patch_ids_var = torch.tensor(
#         [
#             [0, 0, 0, 1, 1, 2],  # Batch 1: patches of size 3,2,1
#             [0, 1, 1, 1, 2, 2],  # Batch 2: patches of size 1,3,2
#         ],
#         dtype=torch.long,
#     )

#     result_var = topk_mean_pooling(h, max_num_patches, k, patch_ids_var)

#     # Verify shape
#     assert result_var.shape == (batch_size, max_num_patches, emb_dim), (
#         f"Incorrect output shape for variable patch sizes. "
#         f"Got {result_var.shape}, expected {(batch_size, max_num_patches, emb_dim)}"
#     )


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
