"""Whole-sequence readout probe: it must discriminate the received picture (a
causal state knows its prefix) from the paper's conjecture (the head already
carries the compressed whole), and read nothing where there is nothing.

The three synthetic regimes below have analytic signatures for the BAG band:

    own     each state is its own token only          -> ~0 everywhere
    prefix  each state is the running prefix mean     -> rises linearly ~t/T
    whole   each state is the whole-window mean       -> flat, near 1
"""

import pytest
import torch

import praxis.metrics.density as density
from praxis.metrics.density import BUCKETS, POSITION_LABELS, DensityProbe

BAND_INDEX = {"bag": 0, "coarse": 1, "mid": 2}

B, T, D = 32, 128, 64
NOISE = 0.3


@pytest.fixture(autouse=True)
def _sample_every_forward(monkeypatch):
    monkeypatch.setattr(density, "SAMPLE_EVERY", 1)
    # The readings are EMA'd across forwards; the tests read the converged
    # value, so smoothing only slows them down.
    monkeypatch.setattr(density, "SCORE_EMA", 0.0)


def _states(seq, mode):
    """Depth-step states [B, T, D] for a given regime; unit-variance signals."""
    if mode == "own":
        signal = seq
    elif mode == "prefix":
        counts = torch.arange(1, T + 1).view(1, T, 1).sqrt()
        signal = torch.cumsum(seq, 1) / counts
    elif mode == "whole":
        signal = (seq.mean(1, keepdim=True) * T**0.5).expand(B, T, D)
    else:
        raise ValueError(mode)
    return signal + NOISE * torch.randn(B, T, D)


def _drive(mode, forwards=60, steps=2, tail=5):
    """Run the probe over ``forwards`` batches; average the last ``tail`` readings."""
    torch.manual_seed(0)
    probe = DensityProbe()
    acc = {}
    for f in range(forwards):
        seq = torch.randn(B, T, D)  # i.i.d. tokens: nothing about the whole in any one
        probe.begin(seq)
        for _ in range(steps):
            probe.observe(_states(seq, mode))
        probe.finalize()
        if f >= forwards - tail:
            for k, v in probe.get_metrics().items():
                acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _profile(metrics, band):
    return [
        metrics[f"readout_cell_{label}_{BAND_INDEX[band]}"] for label in POSITION_LABELS
    ]


def test_own_token_reads_nothing():
    """No state carries anything about the whole: every band sits at chance."""
    m = _drive("own")
    for band in ("bag", "coarse", "mid"):
        assert max(abs(v) for v in _profile(m, band)) < 0.08, band
    assert abs(m["readout_bag_rim_gap"]) < 0.08


def test_prefix_reads_the_bag_in_proportion_to_prefix_length():
    """The received picture: the bag rises linearly head to tip."""
    m = _drive("prefix")
    bag = _profile(m, "bag")
    assert bag[0] < 0.2
    assert bag[-1] > 0.6
    assert all(b - a > -0.05 for a, b in zip(bag, bag[1:])), bag  # monotone
    assert m["readout_bag_rim_gap"] > 0.5
    # The loop's states carry more of the whole than the raw tokens did.
    assert m["readout_bag_depth_gain"] > 0.2


def test_whole_at_every_position_reads_flat_and_high():
    """The conjecture's limit: the head carries the whole as well as the tip."""
    m = _drive("whole")
    bag = _profile(m, "bag")
    assert min(bag) > 0.7
    assert abs(m["readout_bag_rim_gap"]) < 0.1


def test_mid_band_stays_at_chance_where_a_mean_cannot_carry_it():
    """A whole-window mean holds no mid-frequency content, so mid reads chance
    in the whole regime and for own tokens; only a prefix can hint at it."""
    for mode in ("own", "whole"):
        m = _drive(mode, forwards=40)
        assert max(abs(v) for v in _profile(m, "mid")) < 0.08, mode


def test_short_windows_are_measured():
    """A byte-latent decoder's window is patches - a few tens at short
    curriculum tiers. The probe must fire there, not only past 64 positions."""
    torch.manual_seed(0)
    probe = DensityProbe()
    for _ in range(12):
        seq = torch.randn(B, 20, D)
        probe.begin(seq)
        probe.observe((seq.mean(1, keepdim=True) * 20**0.5).expand(B, 20, D))
        probe.finalize()
    m = probe.get_metrics()
    assert m, "no reading on a 20-position window"
    assert min(_profile(m, "bag")) > 0.5


def test_position_labels_sort_head_to_tip():
    """The heatmap renderer sorts row labels alphabetically; the labels must
    come out head..tip under that sort or the strip reads backwards."""
    assert sorted(POSITION_LABELS) == list(POSITION_LABELS)


def test_probe_stays_out_of_state_dict_and_skips_short_sequences():
    probe = DensityProbe()
    assert not isinstance(probe, torch.nn.Module)
    probe.begin(torch.randn(4, 8, 8))  # below MIN_LENGTH
    probe.observe(torch.randn(4, 8, 8))
    probe.finalize()
    assert probe.get_metrics() == {}


def test_length_change_mid_loop_is_skipped_not_garbage():
    """A compressor can shorten the sequence between boundaries; the probe
    must not pair states with a target of another shape."""
    probe = DensityProbe()
    seq = torch.randn(B, T, D)
    probe.begin(seq)
    probe.observe(torch.randn(B // 2, T, D))  # batch mismatch -> skipped
    probe.finalize()
    assert probe.get_metrics() == {}
