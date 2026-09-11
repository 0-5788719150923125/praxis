from types import SimpleNamespace

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM, registry

# ------------------------------------------------------------------------------
# calm
# ------------------------------------------------------------------------------
# CALM encoder + energy head + LF-temperature sanity tests.
#
# These are shape / plumbing checks rather than training-quality assertions. The smoke-
# test in the CALM README covers the latter.


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=256,
        embed_size=32,
        hidden_size=64,
        num_heads=4,
        num_queries=2,
        num_layers=2,
        depth=2,
        block_size=32,
        max_position_embeddings=32,
        encoder_type="calm_small",
    )
    defaults.update(overrides)
    return PraxisConfig(**defaults)


def test_stacked_head_logits_match_manual_compose():
    # forward == terminal(transform(h)): the field is genuinely in the path.
    cfg = _tiny_config(head_type="crystal_harmonic")
    model = PraxisForCausalLM(cfg)
    model.eval()
    head = model.head
    harmonic, crystal = head.heads[0], head.heads[1]
    feat = torch.randn(2, 6, model.encoder.output_dim)
    with torch.no_grad():
        composed = head(feat)
        manual = crystal(harmonic.transform(feat))
    assert torch.allclose(composed, manual, atol=1e-5)
    assert composed.shape == (2, 6, model.encoder.output_vocab_size)


# ------------------------------------------------------------------------------
# parallel_head
# ------------------------------------------------------------------------------
# ParallelHead: gated parallel branches + namespaced per-branch dashboards.


def _cfg(**over):
    base = dict(
        hidden_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        encoder_type="",
        loss_func="cross_entropy",
        crystal_n=None,
        crystal_label_smoothing=None,
        tie_word_embeddings=False,
        embed_size=16,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_crystal_harmonic_descriptions_unchanged():
    # Regression guard for the SequentialHead.all_metric_descriptions override:
    # the single-field profile must still surface bare (unprefixed) keys.
    torch.manual_seed(0)
    head = registry.lookup("heads", "crystal_harmonic")(_cfg(), encoder=None)
    descs = head.all_metric_descriptions()
    assert "harmonic_amplitudes_norm" in descs
    assert not any(k.startswith("p0_") for k in descs)
