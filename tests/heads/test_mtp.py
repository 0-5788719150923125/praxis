"""praxis/heads/mtp: MultiTokenPrediction - the depth banks, prompt masking,
drafting the function that was trained, the adaptive draft width, and the
objective each path trains under."""

import copy
from types import SimpleNamespace

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.heads.mtp import (
    _ACCEPT_WIDTH_MARGIN,
    _WIDTH_PROBE_EVERY,
    MultiTokenPrediction,
)
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.regression import MeanSquaredErrorLoss
from praxis.modeling import PraxisForCausalLM


@pytest.fixture
def spec_config():
    """Byte-latent + prismatic4 head + dual memory + VEAR MTP: the drafting stack."""
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
        head_type="prismatic4",
        memory_type="mal_energy_dual",
        mtp_type="vear",
        mtp_depth=4,
    )


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


@pytest.mark.parametrize(
    "mtp_type", ["per_depth", "vear", "serpent_rnn", *registry.namespace("mtp")]
)
def test_every_mtp_type_drafts_what_it_trained_or_refuses(spec_config, mtp_type):
    """A depth transform is run over the whole sequence at training time and
    over a SINGLE position at draft time, with no cache. Any transform that
    reads context is therefore a different function in the two settings - the
    silent failure that makes drafts garbage while the aux loss still falls.

    The pointwise banks must agree to float noise. A registry module either
    agrees too or is refused outright on the byte-latent drafting path.
    """
    cfg = copy.copy(spec_config)
    cfg.mtp_type = mtp_type
    torch.manual_seed(0)
    try:
        mtp = MultiTokenPrediction(cfg).eval()
    except ValueError as err:
        assert "context-dependent" in str(err)
        assert mtp_type in registry.namespace("mtp"), "a pointwise bank refused"
        return
    h = torch.randn(2, 16, spec_config.embed_size)
    e = torch.randn(2, 16, spec_config.embed_size)
    with torch.no_grad():
        full = mtp._run_depth(0, h, e, None)
        one = mtp._run_depth(0, h[:, -1:], e[:, -1:], None)
    assert torch.allclose(full[:, -1:], one, atol=1e-5), mtp_type


def test_the_mtp_term_follows_the_path(spec_config):
    """Byte-latent MTP drafts bytes, so it trains under cross-entropy; a
    patch-level encoder (CALM) regresses the next patch representation."""
    assert isinstance(
        MultiTokenPrediction(spec_config).objectives()["mtp"], CrossEntropyLoss
    )
    cfg = copy.copy(spec_config)
    cfg.encoder_type = "calm"
    assert isinstance(
        MultiTokenPrediction(cfg).objectives()["mtp"], MeanSquaredErrorLoss
    )


# ── The speculative draft width ────────────────────────────────────────────
#
# Width is chosen at DECODE time from the run's own accepted-run lengths, so it
# is also the knob that can quietly waste a fifth of a turn. It can only change
# speed: greedy output at every width is pinned in
# tests/generation/test_speculative.py.


def _widths(ema, steps, depth=5):
    """The widths a run at a steady accepted-run length actually spends.
    draft_width reads only these three fields."""
    state = SimpleNamespace(num_depths=depth, _accept_ema=ema, _accept_seen=0)
    out = []
    for _ in range(steps):
        out.append(MultiTokenPrediction.draft_width.fget(state))
        state._accept_seen += 1
    return out


def test_margin_is_spent_on_a_cadence_not_every_step():
    """`_accept_ema` starts at 1.0, so a margin added unconditionally makes a
    model accepting runs of 1 draft 2 forever, and the second candidate never
    lands."""
    widths = _widths(ema=1.0, steps=4 * _WIDTH_PROBE_EVERY)
    assert widths.count(1 + _ACCEPT_WIDTH_MARGIN) == 4
    assert widths.count(1) == len(widths) - 4


@pytest.mark.parametrize(
    "ema, lo, hi",
    [(3.2, 4, 5), (99.0, 5, 5), (0.0, 1, 1)],
    ids=["follows_the_run", "bounded_by_trained_depth", "never_switches_off"],
)
def test_width_bounds(ema, lo, hi):
    widths = _widths(ema=ema, steps=_WIDTH_PROBE_EVERY, depth=5)
    assert (min(widths), max(widths)) == (lo, hi)


def test_draft_width_tracks_accepted_runs(spec_config):
    """Every candidate past the first divergence is discarded but still costs a
    sequential draft and (byte-latent) its own verify row, so the width starts
    CONSERVATIVE and only climbs toward the trained depth as acceptance
    delivers longer runs."""
    spec_config.mtp_depth = 16
    mtp = MultiTokenPrediction(spec_config)
    depth = spec_config.mtp_depth
    assert mtp.draft_width < depth  # conservative at init, not the full depth

    for _ in range(60):
        mtp.note_accepted(1)  # short runs keep the window closed in
    narrow = mtp.draft_width
    assert narrow < depth

    for _ in range(120):
        mtp.note_accepted(depth)  # drafts land again -> widen toward the depth
    assert mtp.draft_width > narrow
