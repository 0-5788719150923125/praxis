"""Tests for praxis/transforms/__init__.py: the model walker (``apply_transform``)
and the alignment request auto-sized modules read (``block_alignment``)."""

from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from praxis import registry
from praxis.transforms import apply_transform, block_alignment
from praxis.transforms.algebra import ALGEBRAS
from praxis.transforms.ghost import AUTO_ORDER, pick_algebra

# --- the walker -------------------------------------------------------------


def test_ghostify_hits_exactly_the_profiled_targets(toy_model):
    model = toy_model()
    stats = apply_transform(model, "ghost_conv_complex")
    assert len(stats.targets) == 6
    assert stats.before == 6 * 64 * 32 * 3
    assert stats.after == stats.before // 2
    assert all(n.startswith("encoder.") for n, _, _, _ in stats.targets)
    assert not parametrize.is_parametrized(model.decoder.conv, "weight")
    assert not parametrize.is_parametrized(
        model.encoder.encoder.layers[0].proj, "weight"
    )


def test_broad_profile_spares_vocab_small_and_indivisible(toy_model):
    model = toy_model()
    stats = apply_transform(model, "ghost_all_complex")
    names = [n for n, _, _, _ in stats.targets]
    for attr, reason in (
        ("scorer", "vocab"),
        ("tiny", "too_small"),
        ("odd", "indivisible"),
    ):
        assert attr not in names
        assert attr in stats.missed[reason]
    assert isinstance(model.scorer, nn.Linear)
    greedy = apply_transform(toy_model(), "ghost_all_greedy_complex")
    assert "scorer" in [n for n, _, _, _ in greedy.targets]


def test_ghost_opaque_subtrees_are_skipped_and_reported(toy_model):
    """A module whose parameters are addressed by a name captured at
    construction, or rewritten in place by an inner loop, cannot be ghosted:
    the transform RENAMES (`weight` -> `parametrizations.weight.original`) and
    DERIVES. `praxis.memory.NeuralMemory` is the real case - its fast-weight Adam
    looks its own tensors up through `self._param_names`, and ghosting it raised
    `KeyError: '0.weight'` on the first forward."""
    model = toy_model()
    model.encoder.decoder.GHOST_OPAQUE = True
    stats = apply_transform(model, "ghost_all_complex")
    names = [n for n, _, _, _ in stats.targets]
    assert not any(n.startswith("encoder.decoder.") for n in names)
    assert all(n.startswith("encoder.decoder.") for n in stats.missed["opaque"])
    assert any(n.startswith("encoder.encoder.") for n in names)


def test_missed_records_names_not_just_counts(toy_model):
    stats = apply_transform(toy_model(), "ghost_all_complex")
    for reason, count in stats.skipped.items():
        assert len(stats.missed[reason]) == count


def test_ghostify_none_is_a_noop_and_unknown_raises(toy_model):
    model = toy_model()
    assert apply_transform(model, "none").targets == []
    assert not parametrize.is_parametrized(
        model.encoder.encoder.layers[0].conv, "weight"
    )
    with pytest.raises(ValueError):
        apply_transform(toy_model(), "no_such_profile")
    with pytest.raises(ValueError):
        apply_transform(nn.Linear(4, 4), "ghost_conv_complex")


def test_double_ghostify_is_refused_not_stacked(toy_model):
    model = toy_model()
    apply_transform(model, "ghost_conv_complex")
    stats = apply_transform(model, "ghost_all_complex")
    already = stats.missed.get("already_parametrized", [])
    assert len(already) == 6 and all(n.endswith(".conv") for n in already)
    # ...and none of them were expanded a second time.
    assert not any(n in already for n, _, _, _ in stats.targets)


# --- the alignment request --------------------------------------------------


def _peer(transform_type, profile="peer_glu"):
    return registry.lookup("dense", profile)(
        SimpleNamespace(
            hidden_size=272,
            num_heads=1,
            activation="swish",
            dropout=0.0,
            transform_type=transform_type,
        )
    )


def test_peer_requests_alignment_and_that_reaches_the_banks():
    """The interface that replaced `even_keys`. At hidden_size 272 the plain round
    lands on 27 keys, giving 729 = 3^6 experts against 272 = 2^4 * 17. Coprime, so
    no d > 1 divides both axes and the largest tensor group in the model is
    unreachable. Configuring a broad ghost profile is now the whole fix: PEER asks
    what it needs and rounds to 26."""
    plain = _peer("none")
    aligned = _peer("ghost_all_complex")
    assert plain.num_experts == 729 and aligned.num_experts == 676
    assert pick_algebra(tuple(plain.down.weight.shape), "auto") is None
    assert pick_algebra(tuple(aligned.down.weight.shape), "auto") is not None


def test_site_specific_profiles_leave_other_sites_alone():
    """A profile that names one site must not resize anything else, or the run
    stops being one change off its baseline. `ghost_conv_complex` is -q, and -q's
    PEER banks have to match -o's exactly."""
    assert block_alignment(SimpleNamespace(transform_type="ghost_conv_complex")) == 1
    assert _peer("ghost_conv_complex").num_experts == _peer("none").num_experts


def test_auto_requests_the_deepest_cut():
    """`auto` walks AUTO_ORDER preferring the deepest algebra, so the request has
    to be the deepest too - an axis divisible by 4 is divisible by 2, so asking
    for the larger costs nothing and keeps quaternion reachable."""
    d = block_alignment(SimpleNamespace(transform_type="ghost_all_auto"))
    assert d == max(len(ALGEBRAS[name][0]) for name in AUTO_ORDER)
    assert _peer("ghost_all_auto").num_experts % d == 0


def test_ghost_opaque_is_not_merge_opaque():
    """PEER sets MERGE_OPAQUE - a claim about routing granularity - and must stay
    ghost-eligible, because its banks are the largest tensor group in the
    decoder."""
    host = nn.Module()
    host.ffn = _peer("ghost_all_complex")
    assert host.ffn.MERGE_OPAQUE is True

    stats = apply_transform(host, "ghost_all_complex")
    names = [n for n, _, _, _ in stats.targets]
    assert "ffn.down" in names
    assert not stats.missed.get("opaque")
