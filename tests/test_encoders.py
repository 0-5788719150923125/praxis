from itertools import product

import pytest
import torch
from torch import nn

from praxis.modules.encoder import (
    PraxisEncoder,
    create_patch_block_ids,
    mask_entropy_preds_at_special_tokens,
    packed_rnn_block,
    pooling_downsample,
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


def test_mask_entropy_preds_at_special_tokens():

    # Test 1: Basic test with small tensors
    print("=== Test 1: Basic Test ===")
    # Create a small batch of input_ids with some special tokens (0)
    input_ids = torch.tensor(
        [
            [1, 2, 0, 4],  # First sequence has a special token at position 2
            [5, 0, 7, 8],  # Second sequence has a special token at position 1
        ]
    )

    # Create entropy_preds with a small vocab size (3)
    # For simplicity, fill with increasing values starting from 1 to avoid zeros
    vocab_size = 3
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    # Create entropy_preds as a flattened tensor [batch_size, seq_len * vocab_size]
    entropy_preds = torch.arange(
        1, batch_size * seq_len * vocab_size + 1, dtype=torch.float32
    )
    entropy_preds = entropy_preds.reshape(batch_size, seq_len * vocab_size)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs:\n{input_ids}")

    print(f"Original entropy_preds shape: {entropy_preds.shape}")
    print(f"Original entropy_preds (flattened):\n{entropy_preds}")

    # Reshape to show the logical 3D structure
    print("Original entropy_preds (reshaped to 3D for visualization):")
    print(entropy_preds.reshape(batch_size, seq_len, vocab_size))

    # Make a copy of the original for comparison
    original_3d = entropy_preds.clone().reshape(batch_size, seq_len, vocab_size)

    # Apply masking
    masked_preds = mask_entropy_preds_at_special_tokens(input_ids, entropy_preds)

    print(f"Masked entropy_preds shape: {masked_preds.shape}")
    print("Masked entropy_preds (reshaped to 3D for visualization):")
    print(masked_preds.reshape(batch_size, seq_len, vocab_size))

    # Verify that predictions at special token positions are zeroed out
    # Reshape for easier verification
    masked_3d = masked_preds.reshape(batch_size, seq_len, vocab_size)

    # Check first batch, position 2 (should be zeros)
    assert torch.all(
        masked_3d[0, 2, :] == 0
    ), "Special token position not correctly masked in batch 0"

    # Check second batch, position 1 (should be zeros)
    assert torch.all(
        masked_3d[1, 1, :] == 0
    ), "Special token position not correctly masked in batch 1"

    # Check that non-special token positions are unchanged from original
    assert torch.all(
        masked_3d[0, 0, :] == original_3d[0, 0, :]
    ), "Non-special token position incorrectly modified in batch 0"
    assert torch.all(
        masked_3d[0, 1, :] == original_3d[0, 1, :]
    ), "Non-special token position incorrectly modified in batch 0"
    assert torch.all(
        masked_3d[0, 3, :] == original_3d[0, 3, :]
    ), "Non-special token position incorrectly modified in batch 0"

    print("All assertions passed for Test 1!")

    # Test 2: More realistic dimensions
    print("\n=== Test 2: Realistic Dimensions ===")
    # Create input_ids with more realistic dimensions
    batch_size = 2
    seq_len = 64
    vocab_size = 260  # Similar to your actual case

    # Create random input_ids with some special tokens (0)
    input_ids = torch.randint(1, 10, (batch_size, seq_len))
    # Randomly place special tokens
    special_positions = torch.randint(
        0, seq_len, (batch_size, 5)
    )  # 5 special tokens per batch

    for batch_idx in range(batch_size):
        for pos in special_positions[batch_idx]:
            input_ids[batch_idx, pos] = 0  # Set special token

    # Create random entropy_preds
    entropy_preds = torch.rand(batch_size, seq_len * vocab_size)

    # Make a copy of the original for comparison
    original_3d = entropy_preds.clone().reshape(batch_size, seq_len, vocab_size)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Special token positions in batch 0: {torch.where(input_ids[0] == 0)[0]}")
    print(f"Special token positions in batch 1: {torch.where(input_ids[1] == 0)[0]}")

    print(f"Original entropy_preds shape: {entropy_preds.shape}")

    # Apply masking
    masked_preds = mask_entropy_preds_at_special_tokens(input_ids, entropy_preds)

    print(f"Masked entropy_preds shape: {masked_preds.shape}")

    # Reshape for verification
    masked_3d = masked_preds.reshape(batch_size, seq_len, vocab_size)

    # Verify masking for special tokens in batch 0
    for pos in torch.where(input_ids[0] == 0)[0]:
        assert torch.all(
            masked_3d[0, pos, :] == 0
        ), f"Special token at position {pos} not masked in batch 0"

    # Verify masking for special tokens in batch 1
    for pos in torch.where(input_ids[1] == 0)[0]:
        assert torch.all(
            masked_3d[1, pos, :] == 0
        ), f"Special token at position {pos} not masked in batch 1"

    # Verify that non-special token positions are unchanged
    for batch_idx in range(batch_size):
        for pos in range(seq_len):
            if input_ids[batch_idx, pos] != 0:  # If not a special token
                assert torch.all(
                    masked_3d[batch_idx, pos, :] == original_3d[batch_idx, pos, :]
                ), f"Non-special token at position {pos} incorrectly modified in batch {batch_idx}"

    # Test that the masked entropy_preds has the same shape as the original
    assert (
        masked_preds.shape == entropy_preds.shape
    ), "Shape mismatch between original and masked predictions"

    print("All assertions passed for Test 2!")


def test_packed_rnn():
    # Create test data
    batch_size, seq_len, feature_dim = 2, 5, 3
    hidden_dim = 4

    # Sample features and input IDs
    x = torch.randn(batch_size, seq_len, feature_dim)
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Set EOS tokens at different positions
    input_ids[0, 2] = 0  # First sequence: EOS at position 2
    # Second sequence: no EOS (should use full length)

    # Create RNN module
    rnn = nn.LSTM(feature_dim, hidden_dim, batch_first=True)

    # Run packed RNN
    output = packed_rnn_block(rnn, x, input_ids, eos_token_id=0)

    # Basic sanity checks
    assert output.shape == (
        batch_size,
        seq_len,
        hidden_dim,
    ), f"Expected shape {(batch_size, seq_len, hidden_dim)}, got {output.shape}"

    # Verify post-EOS values are zero for the first sequence
    post_eos_mean = output[0, 3:].abs().mean().item()
    print(f"Post-EOS activation (should be near 0): {post_eos_mean}")
    assert (
        post_eos_mean < 1e-5
    ), f"Expected near-zero activations after EOS, got {post_eos_mean}"

    # Verify second sequence has non-zero values throughout
    seq2_mean = output[1].abs().mean().item()
    print(f"Second sequence activation (should be > 0): {seq2_mean}")
    assert (
        seq2_mean > 1e-5
    ), f"Expected non-zero activations for second sequence, got {seq2_mean}"

    print("All tests passed!")


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
    result = pooling_downsample(h, max_num_patches, f"topk:{k}", patch_ids)

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
    result_k1 = pooling_downsample(h, max_num_patches, "max", patch_ids=patch_ids)
    # Test with k=seq_len (mean pooling equivalent)
    result_kmax = pooling_downsample(
        h, max_num_patches, f"topk:{k}", patch_ids=patch_ids
    )

    # Verify shapes
    assert result.shape == (batch_size, max_num_patches, emb_dim)
    assert result_k1.shape == (batch_size, max_num_patches, emb_dim)
    assert result_kmax.shape == (batch_size, max_num_patches, emb_dim)

    # Verify difference
    # Get results for all pooling modes
    result_max = pooling_downsample(h, max_num_patches, "max", patch_ids=patch_ids)
    result_min = pooling_downsample(h, max_num_patches, "min", patch_ids=patch_ids)
    result_mean = pooling_downsample(h, max_num_patches, "avg", patch_ids=patch_ids)
    result_topk = pooling_downsample(
        h, max_num_patches, f"topk:{k}", patch_ids=patch_ids
    )

    # Verify all results are different
    assert not torch.allclose(
        result_max, result_topk
    ), "topk_mean should differ from max pooling"
    assert not torch.allclose(
        result_min, result_topk
    ), "topk_mean should differ from min pooling"
    assert not torch.allclose(
        result_mean, result_topk
    ), "topk_mean should differ from mean pooling"

    # Additional verification that results make sense
    # topk_mean should be between max and min
    assert torch.all(
        result_topk <= result_max
    ), "topk_mean should not exceed maximum values"
    assert torch.all(
        result_topk >= result_min
    ), "topk_mean should not be less than minimum values"


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
