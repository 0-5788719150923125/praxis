"""PraxisCache: cached decode must reproduce full-recompute outputs.

Covers the vanilla path (gpt2-1.yml), the Infini/Arc memory-state cache, and
the safe fallback for cache-less attentions (CausalAttention). The head-side
cache state of the harmonic heads is checked logit for logit in
tests/classifiers/test_harmonic.py.
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
        # Memory-free Arc: a plain growing KV cache, no memory fold.
        dict(attention_type="arc_nomem", encoding="arc", window_size=8),
        dict(attention_type="arc_single_dropoff_nomem", encoding="arc", window_size=8),
        # CausalAttention never writes the cache; use_cache=True must still be
        # correct (full recompute each step), just without the speedup.
        dict(attention_type="causal", encoding="rope"),
    ],
    ids=[
        "vanilla",
        "infini",
        "arc",
        "arc_nomem",
        "arc_single_dropoff_nomem",
        "causal",
    ],
)
def test_cached_generate_matches_uncached(kwargs):
    model = build_model(**kwargs)
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
