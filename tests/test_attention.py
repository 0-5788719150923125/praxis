import itertools
from enum import Enum
from typing import NamedTuple

import pytest
import torch

from praxis.modules.attention import PraxisAttention

# Define test parameters
MODULE_CLASSES = [PraxisAttention]
HIDDEN_SIZES = [32, 64, 128, 256]


class AttentionMode(Enum):
    BASE = "base"
    LINEAR = "linear"
    DIFFERENTIAL = "differential"
    STICKBREAKING = "stickbreaking"


class AttentionConfig(NamedTuple):
    mode: AttentionMode
    hidden_size: int
    num_heads: int = 4
    num_queries: int = 1


def get_attention_configs():
    """Generate valid attention configurations."""
    configs = []
    for hidden_size in HIDDEN_SIZES:
        for mode in AttentionMode:
            configs.append(AttentionConfig(mode=mode, hidden_size=hidden_size))
    return configs


@pytest.fixture(params=list(itertools.product(MODULE_CLASSES, get_attention_configs())))
def module_setup(request, config):
    """
    Parametrized fixture that provides module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, config)
    """
    module_class, attention_config = request.param

    # Set hidden size
    setattr(config, "hidden_size", attention_config.hidden_size)
    setattr(config, "num_heads", attention_config.num_heads)
    setattr(config, "num_queries", attention_config.num_queries)

    # Reset all boolean flags
    setattr(config, "linear", False)
    setattr(config, "differential", False)
    setattr(config, "stickbreaking", False)

    # Set the appropriate mode
    if attention_config.mode == AttentionMode.LINEAR:
        setattr(config, "linear", True)
    elif attention_config.mode == AttentionMode.DIFFERENTIAL:
        setattr(config, "differential", True)
    elif attention_config.mode == AttentionMode.STICKBREAKING:
        setattr(config, "stickbreaking", True)
        setattr(config, "encoding", "nope")  # Required for stickbreaking

    module = module_class(config)
    return module, attention_config


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    module, attention_config = module_setup
    batch_size = 32
    seq_len = 16

    # Create input tensor
    x = torch.randn(batch_size, seq_len, attention_config.hidden_size)

    # Run forward pass
    output, aux_loss = module(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, attention_config.hidden_size)

    # Add mode-specific assertions if needed
    if attention_config.mode == AttentionMode.LINEAR:
        # Add assertions specific to linear mode
        pass
    elif attention_config.mode == AttentionMode.DIFFERENTIAL:
        # Add assertions specific to differential mode
        pass
    elif attention_config.mode == AttentionMode.STICKBREAKING:
        # Add assertions specific to stickbreaking mode
        pass


def test_combination_count():
    """Helper test to print total number of parameter combinations."""
    total = len(MODULE_CLASSES) * len(get_attention_configs())
    print(f"\nTotal test combinations: {total}")
    # Should be: len(MODULE_CLASSES) * len(HIDDEN_SIZES) * len(AttentionMode)
    assert True  # Always passes, just for information
