import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.attention.arc import ArcAttention
from praxis.attention.cache import PraxisCache
from praxis.attention.infini import InfiniAttention

# ------------------------------------------------------------------------------
# infini_arc_encoding
# ------------------------------------------------------------------------------
# Smoke tests for InfiniAttention and ArcAttention through the encoding registry.
#
# Confirms each registered encoding (rope, alibi, hope, nope) flows cleanly through both
# the segment loop and the segment-blocked path (block_ids set, which forces
# _local_attention_blocked instead of the fast flex_attention branch). The refactor
# moved positional encoding off hardcoded pos_type branches and onto the ``encoding``
# interface; these tests are the guard that the move didn't drop a code path.


def _make_config(encoding: str) -> PraxisConfig:
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        depth=2,
        block_size=128,
        dropout=0.0,
        encoding=encoding,
        causal=True,
        window_size=32,  # repurposed as segment_size by InfiniAttention
    )


@pytest.mark.parametrize("encoding", ["rope", "alibi", "hope", "nope"])
def test_infini_forward(encoding: str):
    cfg = _make_config(encoding)
    attn = InfiniAttention(cfg)
    attn.eval()
    x = torch.randn(2, 96, 64)  # 96 = 3 segments of 32
    with torch.no_grad():
        y, _, aux = attn(inputs=x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert aux == 0


@pytest.mark.parametrize("encoding", ["rope", "alibi", "hope", "nope"])
def test_infini_forward_with_block_ids(encoding: str):
    # block_ids forces the _local_attention_blocked path, which the refactor
    # also touched (its own inline ALiBi bias is now routed through encoding).
    cfg = _make_config(encoding)
    attn = InfiniAttention(cfg)
    attn.eval()
    batch, seq_len = 2, 96
    x = torch.randn(batch, seq_len, 64)
    # Two packed documents per row, split at the midpoint.
    block_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    block_ids[:, seq_len // 2 :] = 1
    with torch.no_grad():
        y, _, aux = attn(inputs=x, block_ids=block_ids)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("encoding", ["rope", "alibi", "hope", "nope"])
def test_arc_forward(encoding: str):
    cfg = _make_config(encoding)
    attn = ArcAttention(cfg)
    attn.eval()
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        # ArcAttention takes current_depth for its per-depth biases.
        y, _, aux = attn(inputs=x, current_depth=0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_infini_hope_pos_dim_set_after_forward():
    # HoPE resolves its band cutoff on first call. Confirm running through
    # InfiniAttention actually populates that state (i.e. the registry path
    # really invoked before_scores).
    cfg = _make_config("hope")
    attn = InfiniAttention(cfg)
    attn.eval()
    assert attn.encoding._pos_dim is None
    with torch.no_grad():
        attn(inputs=torch.randn(1, 32, 64))
    assert attn.encoding._pos_dim is not None
    assert 0 < attn.encoding._pos_dim <= attn.head_dim


# ------------------------------------------------------------------------------
# kv_cache
# ------------------------------------------------------------------------------
# KV-cache equivalence: cached decode must reproduce full-recompute outputs.
#
# Covers the vanilla path (gpt2-1.yml), the Infini/Arc memory-state cache, and the safe
# fallback for cache-less attentions (CausalAttention).


def build_model(**overrides):
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        **overrides,
    )
    return PraxisForCausalLM(cfg).eval()


def test_infini_state_folds_segments():
    """The live segment folds into memory at the segment boundary."""
    model = build_model(attention_type="infini", encoding="rope", window_size=8)
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
