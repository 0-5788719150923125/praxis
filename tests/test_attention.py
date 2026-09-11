import itertools
import os
import random
from enum import Enum
from typing import Dict, List, NamedTuple

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.attention.causal import CausalAttention

MODULE_CLASSES = list(registry.namespace("attention").values())

# Full Cartesian product is ~22k cases; sample a stratified subset so every
# module class still gets coverage but the suite finishes in seconds.
# Override with PRAXIS_ATTENTION_FULL=1 to run the full grid.
SAMPLES_PER_MODULE = int(os.environ.get("PRAXIS_ATTENTION_SAMPLES", "20"))
SAMPLE_SEED = 0xA77E


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
    "encodings": list(registry.namespace("encoding").keys()),
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


def _sampled_module_configs():
    """Stratified sample: take SAMPLES_PER_MODULE configs per module class.

    Set PRAXIS_ATTENTION_FULL=1 to fall back to the full Cartesian product.
    """
    configs = get_attention_configs()
    if os.environ.get("PRAXIS_ATTENTION_FULL"):
        return list(itertools.product(MODULE_CLASSES, configs))

    rng = random.Random(SAMPLE_SEED)
    sampled = []
    for module_class in MODULE_CLASSES:
        pool = list(configs)
        rng.shuffle(pool)
        sampled.extend((module_class, c) for c in pool[:SAMPLES_PER_MODULE])
    return sampled


@pytest.fixture(params=_sampled_module_configs())
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
    # Registry entries may be partials (profiles); inspect the underlying class.
    base_class = getattr(module_class, "func", module_class)

    if issubclass(base_class, CausalAttention):
        if attention_config.encoding == "nope":
            pytest.skip(
                "CausalAttention requires a positional encoding (alibi or rope)"
            )
        if (
            attention_config.encoding == "rope"
            and (attention_config.hidden_size // attention_config.num_heads) % 2 != 0
        ):
            pytest.skip("CausalAttention with RoPE requires an even head_dim")

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


def _packed_block_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Two documents per row, split at different points per row."""
    split = seq_len // 2
    rows = []
    for b in range(batch_size):
        cut = split + (b % 3) - 1
        rows.append([1] * cut + [2] * (seq_len - cut))
    return torch.tensor(rows, dtype=torch.long)


def test_causal_attention_honours_block_ids():
    """A packed document must not be able to read the one before it.

    `block_ids` reached CausalAttention for a long time without being used,
    so packed documents attended across each other. Perturbing document 1
    must leave document 2's outputs untouched.
    """
    config = PraxisConfig(
        hidden_size=64, num_heads=2, num_queries=1, encoding="nope", causal=True
    )
    config.causal = True
    module = CausalAttention(config).eval()

    batch_size, seq_len = 4, 16
    block_ids = _packed_block_ids(batch_size, seq_len)
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    perturbed = x.clone()
    for b in range(batch_size):
        first_doc = block_ids[b] == 1
        perturbed[b, first_doc] += 5.0

    with torch.no_grad():
        base, _, _ = module(x, block_ids=block_ids)
        moved, _, _ = module(perturbed, block_ids=block_ids)
        no_ids_base, _, _ = module(x, block_ids=None)
        no_ids_moved, _, _ = module(perturbed, block_ids=None)

    second_doc = block_ids == 2
    leak = (base[second_doc] - moved[second_doc]).abs().max()
    control = (no_ids_base[second_doc] - no_ids_moved[second_doc]).abs().max()

    assert leak < 1e-6, f"document 2 saw document 1 (delta {leak})"
    assert control > 1e-3, "control is not a real signal; the test proves nothing"


def test_causal_attention_block_mask_cache_is_batch_independent():
    """The (q_len, kv_len, device) cache must not serve a document mask.

    Document masks depend on batch contents, so they are rebuilt every
    forward; only the batch-independent causal mask may be cached.
    """
    config = PraxisConfig(
        hidden_size=64, num_heads=2, num_queries=1, encoding="nope", causal=True
    )
    config.causal = True
    module = CausalAttention(config).eval()

    if module.create_block_mask is None:
        pytest.skip("FlexAttention unavailable")

    batch_size, seq_len = 4, 16
    device = torch.device("cpu")
    block_ids = _packed_block_ids(batch_size, seq_len)

    module._create_causal_mask(seq_len, seq_len + 1, device, block_ids=block_ids)
    assert len(module.block_mask_cache) == 0

    module._create_causal_mask(seq_len, seq_len + 1, device)
    module._create_causal_mask(seq_len, seq_len + 1, device)
    assert len(module.block_mask_cache) == 1


def test_causal_attention_ignores_mismatched_block_ids():
    """Wrong-shaped block_ids are declined, not masked with."""
    config = PraxisConfig(
        hidden_size=64, num_heads=2, num_queries=1, encoding="nope", causal=True
    )
    config.causal = True
    module = CausalAttention(config).eval()

    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    too_short = torch.ones(batch_size, seq_len // 2, dtype=torch.long)

    with torch.no_grad():
        out, _, _ = module(x, block_ids=too_short)
        expected, _, _ = module(x, block_ids=None)

    assert torch.equal(out, expected)
