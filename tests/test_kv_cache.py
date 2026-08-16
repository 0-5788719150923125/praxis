"""KV-cache equivalence: cached decode must reproduce full-recompute outputs.

Covers the vanilla path (gpt2-1.yml), the Infini/Arc memory-state cache, and
the safe fallback for cache-less attentions (CausalAttention).
"""

import pytest
import torch
from transformers import GenerationConfig

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.attention.cache import PraxisCache

PROMPT_LEN = 13  # deliberately not a multiple of the segment size
NEW_TOKENS = 12  # crosses at least one segment fold with window_size=8


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


def greedy(model, ids, use_cache):
    with torch.no_grad():
        return model.generate(
            ids,
            generation_config=GenerationConfig(
                max_new_tokens=NEW_TOKENS, do_sample=False, use_cache=use_cache
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        # gpt2-1.yml stack: vanilla MHA + learned absolute positions
        dict(attention_type="vanilla", embeddings="positional", encoding="nope"),
        dict(attention_type="infini", encoding="rope", window_size=8),
        dict(attention_type="arc", encoding="arc", window_size=8),
        # Harmonic stem (input envelope + fast weights) + crystal SMEAR arm:
        # both carry context across chunks through the head-side cache state.
        dict(
            attention_type="vanilla",
            embeddings="positional",
            encoding="nope",
            head_type="prismatic6",
            block_size=128,
        ),
    ],
    ids=["vanilla", "infini", "arc", "prismatic6"],
)
def test_cached_generate_matches_uncached(kwargs):
    model = build_model(**kwargs)
    ids = torch.randint(0, 200, (1, PROMPT_LEN))
    assert torch.equal(greedy(model, ids, False), greedy(model, ids, True))


def test_cacheless_attention_falls_back_correctly():
    """CausalAttention never writes the cache; use_cache=True must still be
    correct (full recompute each step), just without the speedup."""
    model = build_model(attention_type="causal", encoding="rope")
    ids = torch.randint(0, 200, (1, PROMPT_LEN))
    assert torch.equal(greedy(model, ids, False), greedy(model, ids, True))


def test_vanilla_cached_logits_match():
    """Step-by-step logit equivalence, not just argmax agreement."""
    model = build_model(
        attention_type="vanilla", embeddings="positional", encoding="nope"
    )
    ids = torch.randint(0, 200, (1, PROMPT_LEN))

    with torch.no_grad():
        full = model(input_ids=ids).logits

        cache = PraxisCache()
        prefill = model(input_ids=ids[:, :-1], past_key_values=cache).logits
        step = model(input_ids=ids[:, -1:], past_key_values=cache).logits

    torch.testing.assert_close(prefill, full[:, :-1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(step, full[:, -1:], rtol=1e-4, atol=1e-5)


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


def test_positional_embedding_offset():
    from praxis.embeddings.positional import PositionalEmbedding

    cfg = PraxisConfig(
        vocab_size=50, hidden_size=32, embed_size=32, device="cpu", dropout=0.0
    )
    emb = PositionalEmbedding(cfg).eval()
    ids = torch.randint(0, 50, (1, 6))
    full = emb(ids)
    suffix = emb(ids[:, 4:], offset=4)
    torch.testing.assert_close(suffix, full[:, 4:])


@pytest.mark.parametrize("head_type", ["crystal_harmonic", "prismatic6"])
def test_harmonic_head_cached_logits_match(head_type):
    """The harmonic field is anchored to absolute position and (prismatic6's
    stem) carries a causal prefix mean and a fast-weight bank across tokens.
    Under cached decode the head sees only the suffix, so without the head-side
    cache state it evaluated every chunk at positions 0.. and pooled the suffix
    alone. Prefill + one-token steps must match the full forward, logit for
    logit, across a FAST_SEGMENT boundary."""
    model = build_model(
        attention_type="vanilla",
        embeddings="positional",
        encoding="nope",
        head_type=head_type,
        block_size=128,
    )
    ids = torch.randint(0, 200, (1, 70))  # crosses the 64-token segment fold
    with torch.no_grad():
        full = model(input_ids=ids).logits
        cache = PraxisCache()
        prefill = model(input_ids=ids[:, :60], past_key_values=cache).logits
        steps = [
            model(input_ids=ids[:, i : i + 1], past_key_values=cache).logits
            for i in range(60, 70)
        ]
    torch.testing.assert_close(prefill, full[:, :60], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 60:], rtol=1e-4, atol=1e-5)
    # The head actually left state behind (it is not falling back silently).
    assert any(k.startswith("HarmonicField:") for k in cache.head_states)
