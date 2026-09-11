"""Decode-length bucketing: the ladder, the padding, and what it must not change.

The point of bucketing is that a turn stops minting a new shape per step, which
is what makes anything shape-sensitive (compiled decode bodies today) safe to
run over a growing context. It is only allowed to buy that if the model writes
the same bytes, so most of this file is about equivalence rather than shapes.
"""

import pytest
import torch
from transformers import GenerationConfig

from praxis import PraxisConfig
from praxis.generation.bucketing import (
    DECODE_BUCKETS,
    active_buckets,
    bucket_length,
    decode_buckets,
    pad_for_decode,
)
from praxis.modeling import PraxisForCausalLM

# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_inactive_by_default():
    """Nothing pads unless a generation opened the scope. Training must never
    see a padded row, and the cheapest way to guarantee that is for the default
    to be off rather than for every call site to remember to opt out."""
    assert active_buckets() is None
    assert bucket_length(37) == 37
    ids = torch.zeros(1, 37, dtype=torch.long)
    out, mask, true_len = pad_for_decode(ids)
    assert out is ids and mask is None and true_len == 37


def test_rounds_up_to_the_next_rung():
    with decode_buckets(rungs=(64, 128, 256)):
        assert bucket_length(1) == 64
        assert bucket_length(64) == 64
        assert bucket_length(65) == 128
        assert bucket_length(256) == 256


def test_past_the_top_rung_is_left_alone():
    """Not padded to some multiple: the only lengths above the top rung are
    ones the positional cap already bounds, and padding past what the model can
    represent is not a shape it should compile for."""
    with decode_buckets(rungs=(64, 128)):
        assert bucket_length(200) == 200


def test_cap_drops_rungs_it_cannot_represent():
    with decode_buckets(rungs=(64, 128, 256, 512), cap=200):
        assert bucket_length(65) == 128
        assert bucket_length(140) == 140  # 256 is above the cap, so no padding


def test_scope_restores_the_previous_ladder():
    with decode_buckets(rungs=(64,)):
        assert active_buckets() == (64,)
        with decode_buckets(enabled=False):
            assert active_buckets() == (64,)  # disabled is a no-op, not a clear
        with decode_buckets(rungs=(32,)):
            assert active_buckets() == (32,)
        assert active_buckets() == (64,)
    assert active_buckets() is None


def test_whole_context_produces_few_shapes():
    """The property the compiled decode bodies need: a turn that walks every
    length from a prompt to the positional cap sees a handful of shapes."""
    with decode_buckets(cap=4096):
        shapes = {bucket_length(t) for t in range(1, 4097)}
    assert len(shapes) <= len(DECODE_BUCKETS)
    assert shapes <= set(DECODE_BUCKETS)


# ---------------------------------------------------------------------------
# padding mechanics
# ---------------------------------------------------------------------------


def test_pad_preserves_the_prefix_and_reports_the_true_length():
    ids = torch.arange(1, 101, dtype=torch.long).unsqueeze(0)
    with decode_buckets(rungs=(64, 128)):
        out, mask, true_len = pad_for_decode(ids)
    assert true_len == 100
    assert out.shape == (1, 128)
    assert torch.equal(out[:, :100], ids)
    assert torch.equal(out[:, 100:], torch.zeros(1, 28, dtype=torch.long))
    assert mask is None


def test_a_supplied_mask_gates_the_pad_off():
    ids = torch.arange(1, 101, dtype=torch.long).unsqueeze(0)
    mask = torch.ones(1, 100, dtype=torch.long)
    with decode_buckets(rungs=(128,)):
        _, out_mask, _ = pad_for_decode(ids, mask)
    assert out_mask.shape == (1, 128)
    assert out_mask[:, :100].all() and not out_mask[:, 100:].any()


def test_ragged_batch_pads_on_the_row_axis_only():
    ids = torch.zeros(4, 70, dtype=torch.long)
    with decode_buckets(rungs=(128,)):
        out, _, true_len = pad_for_decode(ids)
    assert out.shape == (4, 128) and true_len == 70


# ---------------------------------------------------------------------------
# equivalence on a real byte-latent model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def byte_model():
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        decoder_type="sequential",
        head_type="forward",
        encoder_type="abstractinator_v1",
        tokenizer_type="byte_level",
        byte_offset=0,
        byte_vocab_size=256,
        codebook_size=256,
        max_position_embeddings=1024,
        mtp_depth=3,
        mtp_type="per_depth",
    )
    torch.manual_seed(0)
    return PraxisForCausalLM(cfg).eval()


def _greedy(model, ids, rungs=None):
    ctx = (
        decode_buckets(rungs=rungs, cap=1024)
        if rungs
        else decode_buckets(enabled=False)
    )
    with ctx, torch.no_grad():
        return model.generate(
            ids,
            generation_config=GenerationConfig(
                max_new_tokens=24, do_sample=False, use_cache=True
            ),
        )


@pytest.mark.parametrize("prompt_len", [31, 64, 100])
@pytest.mark.parametrize("rungs", [(64, 128, 256), (128, 256)])
def test_bucketed_generation_writes_the_same_bytes(byte_model, prompt_len, rungs):
    """The claim the whole mechanism rests on. Every stage of this stack is
    causal, so positions appended after the last real byte cannot reach it; the
    length-sensitive parts (Kaleidoscope's ratio mirrors, the memory's chunk
    grid) move the logits by ~1e-5 relative, far below what changes an argmax.
    """
    torch.manual_seed(1)
    ids = torch.randint(32, 127, (1, prompt_len))
    assert torch.equal(_greedy(byte_model, ids), _greedy(byte_model, ids, rungs))


def test_next_byte_logits_barely_move_under_padding(byte_model):
    """The same claim one level down, and stated as a number rather than an
    argmax: right-padding is a perturbation, not a no-op, and this is how big
    it is allowed to get before the equivalence above is luck."""
    from praxis.generation.speculative import spec_logits_and_hidden

    torch.manual_seed(2)
    ids = torch.randint(32, 127, (1, 100))
    with torch.no_grad():
        base, _ = spec_logits_and_hidden(byte_model, ids)
        with decode_buckets(rungs=(256,), cap=1024):
            padded, _ = spec_logits_and_hidden(byte_model, ids)
    assert padded.shape == base.shape  # trimmed back for the caller
    last = base[0, -1]
    drift = (padded[0, -1] - last).abs().max() / last.abs().max()
    assert drift < 1e-3, drift
    assert int(padded[0, -1].argmax()) == int(last.argmax())


def test_training_mode_is_never_padded(byte_model):
    """Belt and braces on the guard in PraxisForCausalLM.forward: the scope
    being open is not enough. Training must be unreachable from here, because
    a padded row would silently change what the model is fit against."""
    ids = torch.randint(32, 127, (1, 100))
    byte_model.train()
    try:
        with decode_buckets(rungs=(256,), cap=1024), torch.no_grad():
            out = byte_model(input_ids=ids)
    finally:
        byte_model.eval()
    assert out.logits.shape[1] == 100


def test_kv_cached_decode_is_unaffected(byte_model):
    """Padding is only inert because nothing reads the pad positions, and a KV
    cache is exactly a thing that reads them later: a padded prefill would write
    pad K/V into the cache and every later step would attend to it. So a cached
    forward must go through unpadded even inside the scope.

    Built on a TOKEN model, because that is the configuration that reaches the
    cache at all - the encoder branch of `prepare_inputs_for_generation` hands
    back no cache, which is why bucketing suits it.
    """
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        block_type="transformer",
        attention_type="vanilla",
        embeddings="positional",
        encoding="nope",
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    model = PraxisForCausalLM(cfg).eval()
    ids = torch.randint(0, 200, (1, 13))
    gen = GenerationConfig(max_new_tokens=12, do_sample=False, use_cache=True)
    with torch.no_grad():
        plain = model.generate(ids, generation_config=gen)
        with decode_buckets(rungs=(64, 128), cap=256):
            bucketed = model.generate(ids, generation_config=gen)
    assert torch.equal(plain, bucketed)


def test_a_labelled_forward_is_never_padded(byte_model):
    """The other half of the same guard: eval mode with labels is validation,
    and a padded row would move the number it reports."""
    ids = torch.randint(32, 127, (1, 100))
    with decode_buckets(rungs=(256,), cap=1024), torch.no_grad():
        out = byte_model(input_ids=ids, labels=ids[:, 1:])
    assert out.logits.shape[1] == 100
