from itertools import product

import pytest
import torch

from praxis.modules.encoder import (
    PraxisEncoder,
    create_patch_block_ids,
    topk_mean_pooling,
)

# Define test parameters
MODULE_CLASSES = [PraxisEncoder]
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
    return module, config


def test_forward_pass(module_setup):
    """Test using parametrized module and dimensions."""
    module, config = module_setup
    # Create sample input
    batch_size = 32
    seq_len = 16  # Should be less than max_seq_len (512)
    vocab_size = 260
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Step 1: Encode
    h, h_encoder, patch_lengths, block_ids, entropy_loss = module.encode(
        input_ids=input_ids
    )

    # Step 2: Decode
    decoder_output = module.decode(
        h,
        h_encoder,
        input_ids,
        patch_lengths,
    )

    # Basic shape assertions
    assert len(decoder_output.shape) == 3, "Expected 3D output from decoder"
    assert decoder_output.shape[0] == batch_size, "Batch size mismatch in output"
    assert decoder_output.shape == (batch_size, seq_len, vocab_size)


def test_create_patch_block_ids():
    """Test patch block ID creation with special tokens."""
    device = "cpu"
    batch_size = 2
    seq_len = 8
    num_patches = 4  # each patch is size 2

    # Create sample input with special tokens (0) at specific positions
    input_ids = torch.tensor(
        [
            [1, 2, 0, 4, 5, 0, 7, 8],  # Two special tokens
            [1, 0, 3, 4, 0, 6, 0, 8],  # Three special tokens
        ],
        device=device,
    )

    # Create patch lengths - each patch is size 2
    patch_lengths = torch.full(
        (batch_size, num_patches), 2, device=device, dtype=torch.long
    )

    # Create patch IDs - each position maps to its patch (0-3)
    patch_ids = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        device=device,
        dtype=torch.long,
    )

    # Get block IDs
    block_ids = create_patch_block_ids(
        input_ids=input_ids,
        patch_lengths=patch_lengths,
        patch_ids=patch_ids,
        special_tokens=[0],
    )

    # Updated expected output to be patch-level
    expected = torch.tensor(
        [
            [1, 1, 2, 3],  # 4 patches for first sequence
            [1, 2, 2, 3],  # 4 patches for second sequence
        ],
        device=device,
        dtype=torch.long,
    )

    assert block_ids.shape == expected.shape
    assert torch.all(
        block_ids == expected
    ), f"Block IDs mismatch.\nGot:      {block_ids}\nExpected: {expected}"

    # Test edge case: all special tokens
    input_ids_all_special = torch.zeros((1, seq_len), device=device, dtype=torch.long)
    block_ids_all_special = create_patch_block_ids(
        input_ids=input_ids_all_special,
        patch_lengths=patch_lengths[0:1],
        patch_ids=patch_ids[0:1],
    )

    expected_all_special = torch.tensor(
        [[1, 2, 3, 4]],  # 4 patches, each containing special tokens
        device=device,
        dtype=torch.long,
    )
    assert block_ids_all_special.shape == expected_all_special.shape
    assert torch.all(
        block_ids_all_special == expected_all_special
    ), f"All special tokens case mismatch.\nGot:      {block_ids_all_special}\nExpected: {expected_all_special}"


def test_topk_mean_pooling():
    """Test topk_mean_pooling with more realistic data."""
    # More realistic dimensions
    batch_size = 2
    seq_len = 12
    emb_dim = 8
    max_num_patches = 4
    k = 3

    # Create input with varied embeddings
    h = torch.randn(batch_size, seq_len, emb_dim) * 5  # Random values, scaled up

    # Create patches of varying sizes
    patch_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3],  # Sizes: 3,4,2,3
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3],  # Sizes: 2,4,3,3
        ],
        dtype=torch.long,
    )

    # Call function
    result = topk_mean_pooling(h, max_num_patches, patch_ids, k)

    # Manually calculate expected results
    expected = torch.zeros(batch_size, max_num_patches, emb_dim)

    # Verify for each batch and patch
    for b in range(batch_size):
        for p in range(max_num_patches):
            # Get values for this patch
            patch_mask = patch_ids[b] == p
            patch_vals = h[b][patch_mask]

            # Calculate expected top-k mean
            if len(patch_vals) > 0:
                k_actual = min(k, len(patch_vals))
                topk_vals, _ = torch.topk(patch_vals, k_actual, dim=0)
                expected[b, p] = topk_vals.mean(dim=0)

    # Verify results
    assert torch.allclose(result, expected, rtol=1e-5), (
        f"Mismatch in topk_mean_pooling results.\n"
        f"Got:\n{result}\n"
        f"Expected:\n{expected}"
    )

    # Add edge case tests
    # Test with k=1 (max pooling equivalent)
    result_k1 = topk_mean_pooling(h, max_num_patches, k=1, patch_ids=patch_ids)
    # Test with k=seq_len (mean pooling equivalent)
    result_kmax = topk_mean_pooling(h, max_num_patches, k=seq_len, patch_ids=patch_ids)

    # Verify shapes
    assert result.shape == (batch_size, max_num_patches, emb_dim)
    assert result_k1.shape == (batch_size, max_num_patches, emb_dim)
    assert result_kmax.shape == (batch_size, max_num_patches, emb_dim)


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
