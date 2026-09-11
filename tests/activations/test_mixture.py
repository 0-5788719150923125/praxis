import pytest
import torch

from praxis.activations import build_activation


def test_mixture_blends_the_whole_bank():
    """A mixture is a weighted sum of ALL its branches, not a selection.

    Asserted against a hand-set coefficient vector rather than against "the
    output differs", because a module that ignored its weights entirely and
    averaged the bank would pass the looser check at init - uniform is exactly
    where every mixture starts.
    """
    from praxis.activations.mixture import ActivationMixture

    mixture = ActivationMixture(("linear", "relu", "tanh"), mode="convex")
    x = torch.randn(4, 16)

    with torch.no_grad():
        mixture.logits.copy_(torch.tensor([10.0, -10.0, -10.0]))
        assert torch.allclose(mixture(x), x, atol=1e-3), "should collapse to identity"
        mixture.logits.copy_(torch.zeros(3))
        expected = (x + torch.relu(x) + torch.tanh(x)) / 3
        assert torch.allclose(mixture(x), expected, atol=1e-6)


def test_mixture_modes_hold_their_constraints():
    """conv(F) is non-negative and sums to one; aff(F) sums to one with the
    sign constraint dropped. The affine one is only interesting BECAUSE it can
    go negative, so that has to be reachable rather than merely unenforced."""
    from praxis.activations.mixture import ActivationMixture

    convex = ActivationMixture(("linear", "relu", "tanh"), mode="convex")
    with torch.no_grad():
        convex.logits.copy_(torch.tensor([3.0, -1.0, 0.5]))
    c = convex._static_coefficients()
    assert torch.allclose(c.sum(), torch.tensor(1.0), atol=1e-6)
    assert (c >= 0).all()

    affine = ActivationMixture(("linear", "relu", "tanh"), mode="affine")
    with torch.no_grad():
        affine.coefficients.copy_(torch.tensor([2.0, 0.0, 0.0]))
    a = affine._static_coefficients()
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-6)
    assert (a < 0).any(), "the affine hull must be able to subtract a branch"


def test_gated_mixture_routes_per_element_and_reports_it():
    """The gate reads the input VALUE, so its coefficients vary across elements
    and the mixture is not a shape-dependent module.

    `activation_mix_routing` is the diagnostic that separates real routing from
    a static preference - a lesson carried from the Servant chirp, whose signal
    saturated into a constant and looked healthy on a magnitude metric.
    """
    from praxis.activations.mixture import ActivationMixture

    mixture = ActivationMixture(("linear", "relu", "tanh"), mode="gated")
    mixture.train()

    with torch.no_grad():
        mixture.slope.copy_(torch.tensor([2.0, -2.0, 0.0]))

    x = torch.randn(8, 32)
    mixture(x)
    metrics = mixture.training_metrics()
    assert metrics["activation_mix_routing"] > 0.0

    # Zero slope is a static blend: same coefficients everywhere, no routing.
    with torch.no_grad():
        mixture.slope.zero_()
    mixture(x)
    assert mixture.training_metrics()["activation_mix_routing"] == pytest.approx(0.0)

    # And it works where the last axis is not a feature axis (PEER hands its
    # activation `[b, n, h, k]`), which is what per-channel weights could not.
    assert mixture(torch.randn(2, 3, 4, 8)).shape == (2, 3, 4, 8)


def test_mixture_metrics_are_declared():
    """A metric with no declaration is written to the database and then dropped
    on the floor, so every key `training_metrics` emits needs a chart entry -
    including the per-branch shares, whose names depend on the bank."""
    from praxis.activations.mixture import ActivationMixture

    mixture = ActivationMixture(("serpent", "swish", "linear"), mode="gated")
    mixture.train()
    mixture(torch.randn(2, 8, 16))
    for key in mixture.training_metrics():
        assert key in type(mixture).metric_descriptions, f"undeclared metric: {key}"


def test_keyed_mixture_partitions_by_an_external_index():
    """`keyed` is the discrete arm: the branch is chosen by the caller's index,
    not by the value and not by a learned parameter.

    The fraction contract is the point - the caller owns the index space and
    normalizes into [0, 1), so the mixture never has to learn what PEER's bank
    (or a head axis, or a depth) looks like.
    """
    from praxis.activations.mixture import ActivationMixture

    mixture = ActivationMixture(("relu", "tanh"), mode="keyed")
    assert mixture.wants_keys
    assert not list(mixture.parameters()), "the partition IS the key; nothing to learn"

    x = torch.randn(4, 8)
    keys = torch.linspace(0, 1, 8).expand(4, 8)
    out = mixture(x, keys=keys)
    # First half of the key range takes relu, second half takes tanh.
    assert torch.allclose(out[:, :4], torch.relu(x[:, :4]), atol=1e-6)
    assert torch.allclose(out[:, 4:], torch.tanh(x[:, 4:]), atol=1e-6)

    # A caller with no index gets values[0] rather than an exception - see
    # test_mix_split_without_an_index_is_the_first_value.
    assert torch.allclose(mixture(x), torch.relu(x), atol=1e-6)


def test_keyed_mixture_reports_realized_occupancy():
    """Segments are equal by construction but the KEYS are not uniformly drawn -
    PEER retrieves experts by score - so what each branch carries is a
    measurement, not a declared ratio."""
    from praxis.activations.mixture import ActivationMixture

    mixture = ActivationMixture(("relu", "tanh"), mode="keyed")
    mixture.train()
    x = torch.randn(4, 8)

    # Every element in the first segment.
    mixture(x, keys=torch.zeros(4, 8))
    metrics = mixture.training_metrics()
    assert metrics["activation_mix_share_relu"] == pytest.approx(1.0)
    assert metrics["activation_mix_share_tanh"] == pytest.approx(0.0)
    assert metrics["activation_mix_entropy"] == pytest.approx(0.0, abs=1e-6)

    mixture(x, keys=torch.linspace(0, 1, 8).expand(4, 8))
    assert mixture.training_metrics()["activation_mix_share_relu"] == pytest.approx(0.5)


def test_mix_split_without_an_index_is_the_first_value():
    """The fallback that lets `mix_split` be declared model-wide.

    PEER's expert bank is the only place with an index to partition on; the
    encoder, the heads and the controllers have none. Falling back to values[0]
    is what makes one line mean "split it where there is something to split, and
    otherwise run the primary activation" - so an arm that adds the split stays
    ONE change off the arm that does not.
    """
    from praxis.activations.mixture import ActivationMixture

    split = build_activation({"type": "mix_split", "values": ["servant", "swish"]})
    x = torch.randn(4, 16)
    assert torch.allclose(split(x), split.branches[0](x))

    # With an index it really does partition.
    keys = torch.cat([torch.zeros(4, 8), torch.full((4, 8), 0.9)], dim=-1)
    out = split(x, keys=keys)
    assert torch.allclose(out[:, :8], split.branches[0](x)[:, :8])
    assert torch.allclose(out[:, 8:], split.branches[1](x)[:, 8:])


def test_unused_branches_are_still_materialized():
    """A lazily-shaped value that the fallback never calls would still hold
    UninitializedParameter when the optimizer walked model.parameters(), and
    raise there. A bank whose first value is parameter-free and whose second is
    not is a perfectly reasonable config, so it must not crash the run."""
    split = build_activation({"type": "mix_split", "values": ["gelu", "servant"]})
    split(torch.randn(4, 16))
    torch.optim.SGD(split.parameters(), lr=0.1)  # raises if any stayed lazy
    assert [n for n, _ in split.named_parameters()], "servant should have params"
