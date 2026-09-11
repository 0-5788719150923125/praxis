"""CausalAttention and what its subclasses (Infini, Arc) inherit from it: packed
document isolation, the block-mask cache, the head-budget recipe, and the dropoff
gate."""

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.attention.causal import CausalAttention

ENCODINGS = sorted(registry.namespace("encoding"))


def _config(**fields):
    config = PraxisConfig(hidden_size=64, num_heads=2, num_queries=1, dropout=0.0)
    config.causal = True  # modeling.py sets this at assembly; the bare config is False
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _packed_block_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Two documents per row, split at different points per row."""
    split = seq_len // 2
    rows = []
    for b in range(batch_size):
        cut = split + (b % 3) - 1
        rows.append([1] * cut + [2] * (seq_len - cut))
    return torch.tensor(rows, dtype=torch.long)


# Within a segment the blocked path isolates documents (arc_nomem passes), but the
# compressive memory folded from earlier segments is read by every later query,
# whichever document wrote it.
MEMORY_CROSSES_DOCUMENTS = pytest.mark.xfail(
    strict=True,
    reason="Infini's compressive memory carries document 1 into document 2 across "
    "segments; block_ids never reach the memory read or its fold",
)


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize(
    "key",
    [
        "causal",
        pytest.param("infini", marks=MEMORY_CROSSES_DOCUMENTS),
        pytest.param("arc", marks=MEMORY_CROSSES_DOCUMENTS),
        "arc_nomem",
    ],
)
def test_block_ids_isolate_packed_documents(key, encoding):
    """A packed document must not be able to read the one before it: perturbing
    document 1 leaves document 2's outputs untouched. Infini and Arc run their
    own segment-blocked path, over several segments here."""
    torch.manual_seed(0)
    config = _config(encoding=encoding)
    if key != "causal":
        config.window_size = 8  # Infini's segment size
    module = registry.lookup("attention", key)(config).eval()

    batch_size, seq_len = 4, 32
    block_ids = _packed_block_ids(batch_size, seq_len)
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    perturbed = x.clone()
    perturbed[block_ids == 1] += 5.0

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


def test_block_mask_cache_is_batch_independent():
    """The (q_len, kv_len, device) cache must not serve a document mask.

    Document masks depend on batch contents, so they are rebuilt every
    forward; only the batch-independent causal mask may be cached.
    """
    module = CausalAttention(_config(encoding="nope")).eval()
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


def test_ignores_mismatched_block_ids():
    """Wrong-shaped block_ids are declined, not masked with."""
    module = CausalAttention(_config(encoding="nope")).eval()
    x = torch.randn(4, 16, 64)
    too_short = torch.ones(4, 8, dtype=torch.long)
    with torch.no_grad():
        out, _, _ = module(x, block_ids=too_short)
        expected, _, _ = module(x, block_ids=None)
    assert torch.equal(out, expected)


# ------------------------------------------------------------------------------
# head budget
# ------------------------------------------------------------------------------
# The mixture-of-widths recipe for attention: restrict one forward to a subset of
# KV heads, then restore.


def _arc_attention():
    from praxis.attention.arc import ArcAttention

    return ArcAttention(
        _config(hidden_size=128, num_heads=4, num_queries=2, depth=8, encoding="rope")
    )


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
# dropoff
# ------------------------------------------------------------------------------


def test_dropoff_is_training_only_and_the_always_schedule_fires_every_pass():
    """The gate lives in CausalAttention, so every dropoff user inherits it."""
    cfg = _config(depth=6, num_layers=1, encoding="nope", head_size=16)
    a = registry.lookup("attention", "arc_single_dropoff_always_nomem")(cfg)
    assert a.dropoff_every is True and a.dropoff_step == 5
    k = v = torch.ones(1, 1, 8, 4)
    a.eval()
    assert torch.equal(a._maybe_dropoff(k, v, 5)[1], v)  # inference: no-op
    a.train()
    assert not torch.equal(a._maybe_dropoff(k, v, 0)[1], v)  # every pass
