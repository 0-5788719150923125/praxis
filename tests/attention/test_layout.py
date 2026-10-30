"""The block layout a coarse-to-fine trunk publishes: a causal context stream,
plus blocks that read earlier chunks' context and themselves, and nothing else."""

import torch

from praxis.attention.layout import BlockLayout


def _layout():
    # Context: 4 patches in 2 chunks of 2 (docs 1, 1, 2, 2 in row 1).
    # Blocks: chunk 0 level 0 (1 position), chunk 1 level 0 (1 position).
    group = torch.tensor([-1, -1, -1, -1, 0, 1])
    chunk = torch.tensor([0, 0, 1, 1, 0, 1])
    positions = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.5, 2.5])
    doc = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 1, 2]])
    return BlockLayout(positions=positions, group=group, chunk=chunk, doc=doc)


def test_ghost_key_is_always_readable():
    assert _layout().dense(2)[..., 0].all()


def test_context_is_causal_within_its_document():
    mask = _layout().dense(2)[:, 0, :4, 1:5]
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask[0], causal)
    assert not mask[1, 2, :2].any(), "a document read the one before it"


def test_blocks_read_earlier_chunks_and_themselves_only():
    mask = _layout().dense(2)[0, 0, :, 1:]
    assert not mask[4, :4].any(), "chunk 0's block read context"
    assert mask[4, 4] and not mask[4, 5]
    assert mask[5, :2].all() and not mask[5, 2:4].any(), "chunk 1 read its own chunk"
    assert mask[5, 5] and not mask[5, 4], "a block read another block"
    assert not mask[:4, 4:].any(), "context read a block"


def test_a_one_stream_layout_is_plain_causal_attention():
    """With no blocks, the layout path must equal SelfAttention's own causal
    path: same ghostmax, same rotary positions, on the SDPA kernel."""
    import praxis.attention  # noqa: F401  (registers the attention entries)
    from praxis import PraxisConfig, registry
    from praxis.attention.self_attention import SelfAttention

    torch.manual_seed(0)
    config = PraxisConfig(
        hidden_size=32,
        embed_size=32,
        num_heads=2,
        num_queries=1,
        head_size=16,
        dropout=0.0,
        encoding="rope",
        causal=True,
    )
    for staged in (False, True):
        if staged:
            attention = registry.lookup("attention", "kuramoto")(config)
        else:
            attention = SelfAttention(config)
        attention.eval()
        x = torch.randn(2, 10, 32)
        expected = attention(x)[0]
        T = x.shape[1]
        attention.layout = BlockLayout(
            positions=torch.arange(T).float(),
            group=torch.full((T,), -1),
            chunk=torch.zeros(T, dtype=torch.long),
            doc=torch.zeros(2, T, dtype=torch.long),
        )
        got = attention(x)[0]
        attention.layout = None
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
