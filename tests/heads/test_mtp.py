import copy

import pytest
import torch

from praxis import PraxisConfig
from praxis.heads.mtp import _ACCEPT_WIDTH_MARGIN, _WIDTH_PROBE_EVERY
from praxis.losses.regression import MeanSquaredErrorLoss
from praxis.modeling import PraxisForCausalLM

# ------------------------------------------------------------------------------
# modeling
# ------------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# Lossless multi-token (speculative) inference for the byte-latent stack.
#
# The byte-latent core patches non-causally within a partial patch, so a single
# verify forward over ``committed + drafts`` reads contaminated earlier
# positions. The fix reads each truncated prefix at its OWN last real position
# (causal) and batches them behind an attention mask. Two properties make that
# lossless, and these tests pin both so a regression in either is caught:
#   1. padding invariance - a right-padded, mask-gated prefix predicts the same
#      last-real-position token as its unpadded form (incl. the prismatic4
#      CrystalVearHead router, which must route per-sequence and mask pads);
#   2. greedy speculative decoding reproduces byte-by-byte greedy exactly, up to
#      floating-point argmax ties (batched-GEMM reduction order) where greedy is
#      itself ill-defined.
# --------------------------------------------------------------------------- #


@pytest.fixture
def spec_config():
    """Byte-latent + prismatic4 head + dual memory + VEAR MTP (drafting stack)."""
    return PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
        memory_type="mal_energy_dual",
        mtp_type="vear",
        mtp_depth=4,
    )


@pytest.fixture
def deep_spec_config(spec_config):
    """The drafting stack at abstractinator-c's width, where the cost of
    drafting/verifying candidates acceptance never reaches actually bites."""
    spec_config.mtp_depth = 16
    return spec_config


def test_mtp_honors_the_prompt_mask(spec_config):
    """MTP takes UNDETACHED hidden states and the SHARED head, so an unweighted
    auxiliary CE trains the trunk on positions `assistant_mask` zeroes - prompt
    text keeps shaping the model no matter what the mask says. Passing the mask
    through is what makes `--no-mask-prompts` mean something."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config)
    mtp = model.mtp
    # The vear bank routes stochastically while training, so two identical
    # calls do not agree; eval mode is what makes these comparisons about the
    # weights rather than about the sampling.
    model.eval()

    ids = torch.randint(4, 260, (2, 24))
    hidden = torch.randn(2, 24, spec_config.embed_size)

    # Mask keeping only the back half - a prompt/answer split.
    mask = torch.zeros(2, 24, dtype=torch.uint8)
    mask[:, 12:] = 1

    unmasked = mtp(mtp.prepare_inputs(hidden, ids, None, model.embeds, model.head))
    masked = mtp(
        mtp.prepare_inputs(
            hidden, ids, None, model.embeds, model.head, loss_weights=mask
        )
    )
    assert torch.isfinite(masked.get_loss("mtp"))
    # Different positions -> a different loss. Equality would mean the weights
    # were accepted and then dropped on the floor.
    assert not torch.allclose(masked.get_loss("mtp"), unmasked.get_loss("mtp"))

    # An all-ones mask must reproduce the unweighted loss exactly, so masking
    # changes WHICH positions train without rescaling the gradient.
    ones = torch.ones(2, 24, dtype=torch.uint8)
    all_on = mtp(
        mtp.prepare_inputs(
            hidden, ids, None, model.embeds, model.head, loss_weights=ones
        )
    )
    assert torch.allclose(all_on.get_loss("mtp"), unmasked.get_loss("mtp"), atol=1e-6)

    # An all-zero mask contributes nothing rather than dividing by zero.
    zeros = torch.zeros(2, 24, dtype=torch.uint8)
    none_on = mtp(
        mtp.prepare_inputs(
            hidden, ids, None, model.embeds, model.head, loss_weights=zeros
        )
    )
    assert torch.isfinite(none_on.get_loss("mtp"))
    assert none_on.get_loss("mtp").detach().item() == pytest.approx(0.0, abs=1e-6)


def test_serpent_rnn_mtp_bank(spec_config):
    """serpent_rnn: one shared gated cell owns every depth. Builds inside the
    byte-latent stack, produces the mtp loss and on-device draft-acc capture,
    drafts at the adaptive width, and its parameter count is O(1) in depth
    (only the K x (H+E) depth-embedding table grows with the unroll)."""
    import copy

    spec_config.mtp_type = "serpent_rnn"
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config)
    mtp = model.mtp
    assert mtp.bank is not None and mtp.depths is None

    # Training path: byte-level loss + per-depth draft-acc kept as tensors
    # (the sync happens once in training_metrics, not per depth per step).
    ids = torch.randint(4, 260, (2, 24))
    hidden = torch.randn(2, 24, spec_config.embed_size)
    inputs = mtp.prepare_inputs(hidden, ids, None, model.embeds, model.head)
    losses = mtp(inputs)
    assert torch.isfinite(losses.get_loss("mtp"))
    assert mtp._draft_accs and all(torch.is_tensor(a) for a in mtp._draft_accs)
    metrics = mtp.training_metrics()
    assert isinstance(metrics["mtp_draft_acc"], float)
    assert isinstance(metrics["mtp_rnn_gate_d0"], float)
    assert metrics["mtp_rnn_depth_embed_d0"] == 0.0  # zero-init specialization

    # Draft path: adaptive width, same cell.
    with torch.no_grad():
        drafted = mtp.draft_next_tokens(
            hidden[:1, -1:, :], ids[:1, :1], model.embeds, model.head
        )
    assert drafted.shape == (1, mtp.draft_width)

    # O(1) in depth: a 4x deeper unroll adds only depth-embedding rows.
    from praxis.heads.mtp.rnn import SerpentRNNMTPBank

    view = copy.copy(spec_config)
    view.hidden_size = spec_config.embed_size  # byte-level depth space
    n4 = sum(p.numel() for p in SerpentRNNMTPBank(view, 4).parameters())
    n16 = sum(p.numel() for p in SerpentRNNMTPBank(view, 16).parameters())
    assert n16 - n4 == 12 * (view.hidden_size + view.embed_size)


def test_per_depth_mtp_bank(spec_config):
    """per_depth: K independent light harmonic transforms, chained by hidden.

    The DeepSeek shape - nothing shared between depths, no forced blend back
    toward the previous state - with a POINTWISE transform, which is what keeps
    the drafted function equal to the trained one. Grows linearly in depth (the
    price of independence) and its depths are instrumented for the failure the
    shared cell cannot have: converging on one transform anyway.
    """
    import copy

    spec_config.mtp_type = "per_depth"
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config)
    mtp = model.mtp
    assert mtp.bank is not None and mtp.depths is None

    ids = torch.randint(4, 260, (2, 24))
    hidden = torch.randn(2, 24, spec_config.embed_size)
    losses = mtp(mtp.prepare_inputs(hidden, ids, None, model.embeds, model.head))
    assert torch.isfinite(losses.get_loss("mtp"))
    # No repulsion term: distinctness is measured here, not enforced.
    assert "mtp_vear_repulsion" not in losses.loss_dict

    metrics = mtp.training_metrics()
    assert isinstance(metrics["mtp_field_distinctness"], float)
    for k in range(spec_config.mtp_depth):
        assert metrics[f"mtp_depth_weight_d{k}"] > 0.0
    # Every metric this bank emits must reach a chart, or it is invisible.
    described = mtp.field_metric_descriptions()
    assert not [
        k for k in metrics if k not in described and not k.startswith("mtp_draft_acc")
    ]

    with torch.no_grad():
        drafted = mtp.draft_next_tokens(
            hidden[:1, -1:, :], ids[:1, :1], model.embeds, model.head
        )
    assert drafted.shape == (1, mtp.draft_width)

    # Linear in depth: independence is exactly what costs K transforms.
    from praxis.heads.mtp.independent import PerDepthMTPBank

    view = copy.copy(spec_config)
    view.hidden_size = spec_config.embed_size  # byte-level depth space
    n2 = sum(p.numel() for p in PerDepthMTPBank(view, 2).parameters())
    n4 = sum(p.numel() for p in PerDepthMTPBank(view, 4).parameters())
    assert n4 == 2 * n2


def test_pointwise_banks_draft_what_they_trained(spec_config):
    """A depth transform is run over the whole sequence at training time and
    over a SINGLE position at draft time, with no cache. Any transform that
    reads context is therefore a different function in the two settings - the
    silent failure that makes drafts garbage while the aux loss still falls.

    The pointwise banks agree to float noise, and the context-dependent
    registry modules are refused outright on the drafting path rather than
    allowed to build.
    """
    from praxis.heads.mtp import MultiTokenPrediction

    ids_h = torch.randn(2, 16, spec_config.embed_size)
    ids_e = torch.randn(2, 16, spec_config.embed_size)
    for mtp_type in ("per_depth", "vear", "serpent_rnn"):
        cfg = copy.copy(spec_config)
        cfg.mtp_type = mtp_type
        torch.manual_seed(0)
        mtp = MultiTokenPrediction(cfg).eval()
        with torch.no_grad():
            full = mtp._run_depth(0, ids_h, ids_e, None)
            one = mtp._run_depth(0, ids_h[:, -1:], ids_e[:, -1:], None)
        assert torch.allclose(full[:, -1:], one, atol=1e-5), mtp_type

    for mtp_type in ("transformer", "conv"):
        cfg = copy.copy(spec_config)
        cfg.mtp_type = mtp_type
        with pytest.raises(ValueError, match="context-dependent"):
            MultiTokenPrediction(cfg)


def test_draft_width_tracks_accepted_runs(deep_spec_config):
    """Speculative width follows the accepted-run length, not the trained depth.

    Every candidate past the first divergence is discarded but still costs a
    sequential draft and (byte-latent) its own verify row, so a wide mtp_depth
    whose drafts rarely land would make each step pay O(depth) to commit a byte
    or two. The width starts CONSERVATIVE and only climbs toward the trained
    depth as acceptance actually delivers longer runs.
    """
    torch.manual_seed(0)
    model = PraxisForCausalLM(deep_spec_config).eval()
    mtp = model.mtp
    depth = deep_spec_config.mtp_depth

    assert mtp.draft_width < depth  # conservative at init, not the full depth
    assert mtp.draft_width >= 1

    for _ in range(60):
        mtp.note_accepted(1)  # short runs keep the window closed in
    narrow = mtp.draft_width
    assert narrow < depth
    assert narrow >= 1  # never switches drafting off

    for _ in range(120):
        mtp.note_accepted(depth)  # drafts land again -> widen toward the depth
    assert mtp.draft_width > narrow
    assert mtp.draft_width <= depth  # bounded by trained depth


# ------------------------------------------------------------------------------
# mtp_draft_width
# ------------------------------------------------------------------------------
# The speculative draft width: what it costs, and what it must never change.
#
# Width is the one speculative knob that is chosen at DECODE time from the run's own
# accepted-run lengths, so it is also the one that can quietly waste a fifth of a turn.
# These pin the two properties that make it safe to adapt: a wider or narrower width
# writes the same bytes, and the growth margin is a probe rather than a standing charge.


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


def test_width_follows_the_observed_run(model):
    assert min(_widths(model.mtp, ema=3.2, steps=_WIDTH_PROBE_EVERY)) == 4


def test_width_is_bounded_by_the_trained_depth(model):
    depth = model.config.mtp_depth
    assert max(_widths(model.mtp, ema=99.0, steps=_WIDTH_PROBE_EVERY)) == depth


def test_width_never_switches_drafting_off(model):
    assert min(_widths(model.mtp, ema=0.0, steps=_WIDTH_PROBE_EVERY)) >= 1


# ------------------------------------------------------------------------------
# objectives
# ------------------------------------------------------------------------------
# Every loss a run carries is registered in one container.
#
# The model used to declare its objectives in three different ways: a ``criterion``
# module, a ``reg`` list, and bare ``F.cross_entropy`` calls inside the heads and the
# MTP stack that appeared in neither. These tests pin the collapsed arrangement - one
# container, one entry per term, each owned exactly once.


# ── the model wiring ───────────────────────────────────────────────────────


def _model(**overrides):
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    cfg = dict(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=1,
        depth=2,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        head_type="prismatic5",
        residual_type="smear",
        byte_level=True,
        loss_func="halo",
        mtp_type="per_depth",
        mtp_depth=2,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return PraxisForCausalLM(PraxisConfig(**cfg))


def test_the_mtp_term_is_a_regression_on_the_patch_path():
    m = _model(encoder_type="calm", hidden_size=64, embed_size=64, byte_level=True)
    assert isinstance(m.criterion.mtp, MeanSquaredErrorLoss)
