from itertools import product

import pytest
import torch

from praxis.modules.encoder import PraxisEncoder

# Define test parameters
MODULE_CLASSES = [PraxisEncoder]
META_MODES = [
    ["space", "ngram"],
    ["space"],
    #  ["entropy"] # currently fails
]

# Create parameter combinations
MODULE_PARAMS = list(product(MODULE_CLASSES, META_MODES))


@pytest.fixture(params=MODULE_PARAMS)
def module_setup(request, config):
    """
    Parametrized fixture that provides both module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, hidden_size)
    """
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
    h, h_encoder, patch_lengths, entropy_loss = module.encode(input_ids=input_ids)

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
