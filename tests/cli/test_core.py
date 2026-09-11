import pytest

from praxis import registry
from praxis.cli.core import create_base_parser
from praxis.registry import Entry, Namespace

# ------------------------------------------------------------------------------
# annotated_config
# ------------------------------------------------------------------------------
# The annotated config behind the web app's Download button.


class _Documented:
    """A documented entry.

    Second paragraph, left out of the one-line description."""


class _Bare:
    pass


_REGISTRY = Namespace(
    "toy-kinds",
    {
        "documented": _Documented,
        "bare": _Bare,
        "profile": Entry(dict(a=1), "A profile with its own doc."),
    },
    doc="The toy namespace's doc, the long form of every flag bound to it.",
)


def _toy_parser():
    parser = create_base_parser()
    group = parser.add_argument_group("toy")
    group.add_argument(
        "--kind",
        registry=_REGISTRY,
        default="bare",
        help="which kind to use",
    )
    group.add_argument("--kinds", nargs="*", registry=_REGISTRY, default=[])
    group.add_argument(
        "--width",
        type=int,
        default=8,
        help="How wide",
        doc="The long form of --width, for the config file only.",
    )
    group.add_argument("--port", type=int, default=1, help="Port", exclude_hash=True)
    return parser


def test_registry_sets_the_choices():
    parser = _toy_parser()
    assert parser.parse_args(["--kind", "profile"]).kind == "profile"
    try:
        parser.parse_args(["--kind", "nope"])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown registry key was accepted")


def test_a_registry_flag_cannot_carry_its_own_doc():
    group = create_base_parser().add_argument_group("toy")
    with pytest.raises(ValueError, match="document it where the namespace"):
        group.add_argument("--kind", registry=_REGISTRY, doc="a second copy")


def test_doc_stays_out_of_help():
    assert "long form" not in _toy_parser().format_help()


# ------------------------------------------------------------------------------
# abstractinator_calm
# ------------------------------------------------------------------------------
# AbstractinatorCALM: a continuous CALM arm beside the discrete RVQ arm.
#
# The thesis under test is that the DISCRETE arm can pay for the CONTINUOUS one. CALM's
# energy score is a weak, high-variance signal that needs far more tokens than this line
# can afford; an RVQ code is a dense, low-variance, mode-seeking target, and predicting
# the next code from the same conditioning hidden the energy head reads is what should
# concentrate its conditional.
#
# These tests pin the mechanics, not the thesis - the run decides that.


def test_generation_mode_is_not_part_of_the_model_hash():
    """A registry profile would have forced a whole separate TRAINING run just
    to compare decoders. Both paths are trained by the same objectives, so the
    flag is inference-only and excluded from the hash."""
    from praxis.cli.core.hasher import compute_args_hash

    base = ["--encoder-type", "x", "--batch-size", "4"]
    assert compute_args_hash(base) == compute_args_hash(
        base + ["--generation-mode", "vote"]
    )
