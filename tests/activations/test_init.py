import pytest
import torch

from praxis.activations import build_activation, linear_activation


def test_the_spec_is_always_a_type_over_values():
    """One shape, with a bare name as shorthand for the single-value case.

    The shorthand is what keeps `activation_type: gelu` the thing anyone would
    write, so it has to produce exactly what the long form does.
    """
    from praxis.activations.mixture import ActivationMixture

    assert type(build_activation("gelu")) is type(
        build_activation({"type": "single", "values": ["gelu"]})
    )
    assert not isinstance(build_activation("gelu"), ActivationMixture)

    split = build_activation({"type": "mix_split", "values": ["servant", "swish"]})
    assert isinstance(split, ActivationMixture)
    assert split.type_name == "mix_split" and split.names == ("servant", "swish")

    # A comma-separated string is the same list, for configs carrying scalars.
    assert build_activation({"type": "mix", "values": "gelu, tanh"}).names == (
        "gelu",
        "tanh",
    )

    # An already-built module passes through, so a caller can accept either.
    assert build_activation(split) is split


def test_bad_specs_say_what_is_wrong():
    """A misdeclared activation trains something other than what was written and
    nothing downstream would notice, so every way of getting it wrong raises."""
    with pytest.raises(ValueError, match="exactly one value"):
        build_activation({"type": "single", "values": ["gelu", "tanh"]})
    with pytest.raises(ValueError, match="needs `values`"):
        build_activation({"type": "mix"})
    with pytest.raises(ValueError, match="Unknown activation type"):
        build_activation({"type": "mix_everything", "values": ["gelu", "tanh"]})
    with pytest.raises(ValueError, match="Unknown activation key"):
        build_activation({"type": "single", "values": ["gelu"], "gate": "relu"})
    with pytest.raises(KeyError):
        build_activation("no_such_activation")


def test_linear_is_the_only_key_that_is_not_a_gate():
    """`values` are all gate activations; `linear` fills the held-out half a
    gated feedforward leaves untouched. Absent means absent, not identity - that
    distinction is what makes the filled case a one-variable arm."""
    assert linear_activation("servant") is None
    assert linear_activation({"type": "single", "values": ["servant"]}) is None
    filled = linear_activation(
        {"type": "single", "values": ["servant"], "linear": "gelu"}
    )
    assert filled is not None and not isinstance(filled, torch.nn.Identity)


def test_a_mixture_can_hold_a_mixture():
    """Values resolve through the same builder, so nesting needs no special
    case. Worth pinning because the recursion is the only reason `values` holds
    NAMES rather than modules."""
    from praxis.activations.mixture import ActivationMixture

    outer = build_activation(
        {
            "type": "mix",
            "values": ["gelu", {"type": "mix_gated", "values": ["relu", "tanh"]}],
        }
    )
    assert isinstance(outer, ActivationMixture)
    assert isinstance(outer.branches[1], ActivationMixture)
    x = torch.randn(2, 8)
    assert outer(x).shape == x.shape


def test_harmonic_spectrum_survives_a_mixture_in_the_slot():
    """The regression that emptied the MTP field charts for a whole run.

    Several diagnostics read `act.a` / `act.g` straight off whatever the
    activation slot holds. That assumed a Serpent, and a mixture put an
    `ActivationMixture` there instead - `AttributeError: no attribute 'a'`,
    swallowed by the dynamics logger, charts blank, training unaffected and
    nothing louder than a repeating log line to say so.
    """
    from praxis.activations import harmonic_spectrum

    x = torch.randn(2, 8, 16)

    direct = build_activation("servant")
    direct(x)
    spec = harmonic_spectrum(direct)
    assert spec is not None and spec[0].shape == (16,)

    # Found inside a bank, at whatever position it sits.
    mixed = build_activation({"type": "mix", "values": ["swish", "servant", "relu"]})
    mixed(x)
    spec = harmonic_spectrum(mixed)
    assert spec is not None and spec[0].shape == (16,)

    # None, not a crash, when there is no periodic activation to find.
    plain = build_activation({"type": "mix", "values": ["gelu", "relu"]})
    plain(x)
    assert harmonic_spectrum(plain) is None
    assert harmonic_spectrum(build_activation("gelu")) is None

    # None while still lazy - callers use that to skip the metric entirely.
    assert harmonic_spectrum(build_activation("servant")) is None
