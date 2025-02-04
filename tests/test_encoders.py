from itertools import product

import pytest
import torch

from praxis.modules.encoder import PraxisEncoder, create_patch_block_ids

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
