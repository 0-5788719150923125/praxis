"""InfiniAttention (and Arc, built on it): the segment loop under every encoding,
and the compressive-memory state a cached decode carries."""

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM, registry
from praxis.attention.arc import ArcAttention
from praxis.attention.cache import PraxisCache
from praxis.attention.infini import InfiniAttention


@pytest.mark.parametrize("encoding", sorted(registry.namespace("encoding")))
@pytest.mark.parametrize("cls", [InfiniAttention, ArcAttention])
def test_segment_loop_runs_under_every_encoding(cls, encoding):
    config = PraxisConfig(
        hidden_size=64, num_heads=4, num_queries=1, depth=2, dropout=0.0
    )
    config.causal = True  # modeling.py sets this at assembly; the bare config is False
    config.encoding = encoding
    config.window_size = 32  # repurposed as segment_size by InfiniAttention
    attn = cls(config).eval()
    x = torch.randn(2, 96, 64)  # 3 segments of 32
    with torch.no_grad():
        y, _, aux = attn(inputs=x, current_depth=0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert aux == 0


def test_cache_state_folds_segments():
    """The live segment folds into memory at the segment boundary."""
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        attention_type="infini",
        encoding="rope",
        window_size=8,
    )
    model = PraxisForCausalLM(config).eval()
    ids = torch.randint(0, 200, (1, 8))  # exact multiple: tail starts empty

    cache = PraxisCache()
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache)
    state = next(iter(cache.states.values()))
    assert state["pos"] == 8
    assert state["k"].size(2) == 0  # fully folded, no live tail

    with torch.no_grad():
        model(input_ids=ids[:, :1], past_key_values=cache)
    state = next(iter(cache.states.values()))
    assert state["pos"] == 9
    assert state["k"].size(2) == 1  # decode token started a new live segment
