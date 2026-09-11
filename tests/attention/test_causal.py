from types import SimpleNamespace

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.attention.causal import CausalAttention

# ------------------------------------------------------------------------------
# attention
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# width
# ------------------------------------------------------------------------------
# Mixture-of-widths: the helical deflation policy and its profile.


# ─── Attention head-drop recipe ──────────────────────────────────────────────


def _arc_attention(hidden=128, num_heads=4, num_queries=2):
    from praxis import PraxisConfig
    from praxis.attention.arc import ArcAttention

    cfg = PraxisConfig(
        hidden_size=hidden,
        num_heads=num_heads,
        num_queries=num_queries,
        depth=8,
        dropout=0.0,
        encoding="rope",
        causal=False,
    )
    return ArcAttention(cfg)


def test_head_budget_preserves_output_and_restores():
    attn = _arc_attention()
    x = torch.randn(2, 12, 128)
    with attn.head_budget(torch.tensor([1])):
        assert attn.num_heads == 1 and attn.num_query_heads == 2
        out, *_ = attn(x, current_depth=3)
        assert out.shape == (2, 12, 128)  # residual stream stays full width
    assert attn.num_heads == 4 and attn.num_query_heads == 8  # restored


def test_head_budget_grads_only_kept_heads():
    """Keeping 1 of 4 KV heads should grad exactly its channels: 2 query heads
    (GQA) + 1 K + 1 V, times head_dim, of the fused QKV; and only the kept query
    heads of the per-head betas."""
    attn = _arc_attention()
    x = torch.randn(2, 12, 128)
    attn.zero_grad()
    with attn.head_budget(torch.tensor([2])):
        attn(x, current_depth=5)[0].pow(2).mean().backward()
    hd = attn.head_dim
    assert (attn.qkv.weight.grad.abs().sum(1) > 0).sum().item() == (2 + 1 + 1) * hd
    assert (attn.betas.grad.abs().sum((0, 2, 3)) > 0).sum().item() == 2


# ------------------------------------------------------------------------------
# kaleidoscope
# ------------------------------------------------------------------------------
# Kaleidoscope attention: frozen mirrors, input-conditional turn, per-depth facets.


def _config(**over):
    cfg = SimpleNamespace(
        hidden_size=32,
        num_heads=1,  # patch_config forces this; set >1 only to test the patch
        head_size=16,
        num_queries=1,
        causal=True,
        dropout=0.0,
        depth=4,
        window_size=None,
        max_position_embeddings=64,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_arc_inherits_the_training_gate_and_the_always_schedule():
    """The fix lives in CausalAttention so every dropoff user gets it at once."""

    cfg = _config(depth=6, num_layers=1)
    cfg.encoding, cfg.vocab_size, cfg.dropout = "nope", 256, 0.0
    a = registry.lookup("attention", "arc_single_dropoff_always_nomem")(cfg)
    assert a.dropoff_every is True and a.dropoff_step == 5
    k = v = torch.ones(1, 1, 8, 4)
    a.eval()
    assert torch.equal(a._maybe_dropoff(k, v, 5)[1], v)  # inference: no-op
    a.train()
    assert not torch.equal(a._maybe_dropoff(k, v, 0)[1], v)  # every pass
