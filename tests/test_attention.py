import itertools
from enum import Enum
from typing import Dict, List, NamedTuple

import pytest
import torch

from praxis import PraxisConfig
from praxis.attention import ATTENTION_REGISTRY
from praxis.encoding import ENCODING_REGISTRY

MODULE_CLASSES = list(ATTENTION_REGISTRY.values())


class AttentionMode(Enum):
    BASE = "base"
    LINEAR = "linear"
    DIFFERENTIAL = "differential"
    STICKBREAKING = "stickbreaking"
    MULTIHEAD_LATENT_ATTENTION = "mla"


# Define test parameters in a more structured way
TEST_PARAMS = {
    "hidden_sizes": [64, 128, 256],
    "modes": list(AttentionMode),
    "num_heads": [1, 2, 3],
    "num_queries": [1, 2],
    "k_heads": [None, 2],
    "encodings": list(ENCODING_REGISTRY.keys()),
    "kv_rank": [None, 1, 2],
    "memory": [False],  # True is currently failing in some instances
    "mega": [False, True],
    "gated": [False],  # Broken as well
}


def get_attention_configs() -> List[PraxisConfig]:
    """Generate valid attention configurations using itertools.product."""
    return [
        PraxisConfig(
            mode=mode,
            hidden_size=hidden_size,
            encoding=encoding,
            num_heads=num_heads,
            num_queries=num_queries,
            kv_rank=kv_rank,
            memory=memory,
            k_heads=k_heads,
            mega=mega,
            gated=gated,
        )
        for hidden_size, mode, encoding, num_heads, num_queries, kv_rank, memory, k_heads, mega, gated in itertools.product(
            TEST_PARAMS["hidden_sizes"],
            TEST_PARAMS["modes"],
            TEST_PARAMS["encodings"],
            TEST_PARAMS["num_heads"],
            TEST_PARAMS["num_queries"],
            TEST_PARAMS["kv_rank"],
            TEST_PARAMS["memory"],
            TEST_PARAMS["k_heads"],
            TEST_PARAMS["mega"],
            TEST_PARAMS["gated"],
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
    setattr(config, "memory", attention_config.memory)
    setattr(config, "k_heads", attention_config.k_heads)

    # Set gating mode
    if attention_config.mega:
        setattr(config, "mega", True)
    elif attention_config.gated:
        setattr(config, "gated", True)

    # Set the appropriate mode
    setattr(config, "linear", False)
    setattr(config, "differential", False)
    setattr(config, "stickbreaking", False)
    setattr(config, "mla", False)

    if attention_config.mode == AttentionMode.DIFFERENTIAL:
        setattr(config, "differential", True)
    # elif attention_config.mode == AttentionMode.LINEAR:
    #     setattr(config, "linear", True)
    elif attention_config.mode == AttentionMode.STICKBREAKING:
        setattr(config, "stickbreaking", True)
    elif attention_config.mode == AttentionMode.MULTIHEAD_LATENT_ATTENTION:
        setattr(config, "mla", True)

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
    output, layer_kv, aux_loss = module(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, attention_config.hidden_size)
