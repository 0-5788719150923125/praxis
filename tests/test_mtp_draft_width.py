"""The speculative draft width: what it costs, and what it must never change.

Width is the one speculative knob that is chosen at DECODE time from the run's
own accepted-run lengths, so it is also the one that can quietly waste a fifth
of a turn. These pin the two properties that make it safe to adapt: a wider or
narrower width writes the same bytes, and the growth margin is a probe rather
than a standing charge.
"""

import pytest
import torch
from transformers import GenerationConfig

from praxis import PraxisConfig
from praxis.heads.mtp import _ACCEPT_WIDTH_MARGIN, _WIDTH_PROBE_EVERY
from praxis.modeling import PraxisForCausalLM


@pytest.fixture(scope="module")
def model():
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
        max_position_embeddings=512,
        mtp_depth=5,
        mtp_type="per_depth",
    )
    torch.manual_seed(0)
    return PraxisForCausalLM(cfg).eval()


def _widths(mtp, ema, steps):
    """The widths a run at a steady accepted-run length actually spends."""
    seen, mtp._accept_seen = mtp._accept_seen, 0
    ema_before, mtp._accept_ema = mtp._accept_ema, ema
    try:
        out = []
        for _ in range(steps):
            out.append(mtp.draft_width)
            mtp._accept_seen += 1
        return out
    finally:
        mtp._accept_seen, mtp._accept_ema = seen, ema_before


def test_margin_is_spent_on_a_cadence_not_every_step(model):
    """The bug this replaces: `_accept_ema` starts at 1.0 and the margin was
    added unconditionally, so a model accepting runs of 1 drafted 2 forever and
    the second candidate never landed. On abstractinator-r that cost 19% of a
    turn for nothing."""
    widths = _widths(model.mtp, ema=1.0, steps=4 * _WIDTH_PROBE_EVERY)
    assert widths.count(1 + _ACCEPT_WIDTH_MARGIN) == 4
    assert widths.count(1) == len(widths) - 4


def test_the_probe_still_exists(model):
    """It is a cadence, not a removal: a model whose drafts improve has to be
    able to discover a longer run without any external signal."""
    assert max(_widths(model.mtp, ema=1.0, steps=_WIDTH_PROBE_EVERY)) > 1


def test_width_follows_the_observed_run(model):
    assert min(_widths(model.mtp, ema=3.2, steps=_WIDTH_PROBE_EVERY)) == 4


def test_width_is_bounded_by_the_trained_depth(model):
    depth = model.config.mtp_depth
    assert max(_widths(model.mtp, ema=99.0, steps=_WIDTH_PROBE_EVERY)) == depth


def test_width_never_switches_drafting_off(model):
    assert min(_widths(model.mtp, ema=0.0, steps=_WIDTH_PROBE_EVERY)) >= 1


@pytest.mark.parametrize("width", [1, 2, 5])
def test_greedy_output_is_identical_at_every_width(model, width):
    """The property that makes width a pure speed knob. Every committed byte is
    confirmed against a real forward, so candidates only ever change how much
    work is thrown away - never what is written."""
    torch.manual_seed(1)
    ids = torch.randint(32, 127, (1, 48))
    cfg = GenerationConfig(max_new_tokens=20, do_sample=False, use_cache=True)

    def run(forced):
        prop = type(model.mtp).draft_width
        if forced is not None:
            type(model.mtp).draft_width = property(lambda self, w=forced: w)
        try:
            with torch.no_grad():
                return model.generate(ids, generation_config=cfg)
        finally:
            type(model.mtp).draft_width = prop

    assert torch.equal(run(None), run(width))
