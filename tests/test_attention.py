import itertools
from enum import Enum
from typing import Dict, List, NamedTuple

import pytest
import torch

from praxis.modules.attention import ENCODING_REGISTRY, PraxisAttention


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
    kv_rank: int = None
    encoding: str = "nope"


MODULE_CLASSES = [PraxisAttention]


# Define test parameters in a more structured way
TEST_PARAMS = {
    "hidden_sizes": [32, 64, 128, 256],
    "modes": list(AttentionMode),
    "num_heads": [1, 2, 3, 4],
    "num_queries": [1, 2],
    "encodings": list(ENCODING_REGISTRY.keys()),
    "kv_rank": [None, 1, 2],
}


def get_attention_configs() -> List[AttentionConfig]:
    """Generate valid attention configurations using itertools.product."""
    return [
        AttentionConfig(
            mode=mode,
            hidden_size=hidden_size,
            encoding=encoding,
            num_heads=num_heads,
            num_queries=num_queries,
            kv_rank=kv_rank,
        )
        for hidden_size, mode, encoding, num_heads, num_queries, kv_rank in itertools.product(
            TEST_PARAMS["hidden_sizes"],
            TEST_PARAMS["modes"],
            TEST_PARAMS["encodings"],
            TEST_PARAMS["num_heads"],
            TEST_PARAMS["num_queries"],
            TEST_PARAMS["kv_rank"],
        )
    ]


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

    setattr(config, "hidden_size", attention_config.hidden_size)
    setattr(config, "num_heads", attention_config.num_heads)
    setattr(config, "num_queries", attention_config.num_queries)

    setattr(config, "encoding", attention_config.encoding)
    setattr(config, "kv_rank", attention_config.kv_rank)

    # Set the appropriate mode
    setattr(config, "linear", False)
    setattr(config, "differential", False)
    setattr(config, "stickbreaking", False)
    if attention_config.mode == AttentionMode.LINEAR:
        setattr(config, "linear", True)
    elif attention_config.mode == AttentionMode.DIFFERENTIAL:
        setattr(config, "differential", True)
    elif attention_config.mode == AttentionMode.STICKBREAKING:
        setattr(config, "stickbreaking", True)

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
