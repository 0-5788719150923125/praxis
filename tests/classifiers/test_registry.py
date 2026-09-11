"""Sweeps over the ``classifiers`` registry: what each profile builds, and the
blueprint contract every classifier class honours."""

import importlib
import inspect
import pkgutil

import pytest

import praxis.classifiers as classifiers_pkg
from praxis import registry
from praxis.classifiers.base import BaseClassifier
from praxis.classifiers.harmonic import HarmonicClassifier
from praxis.classifiers.parallel import ParallelClassifier
from praxis.classifiers.sequential import SequentialClassifier
from tests.stubs import Cfg, Enc


def _describe(classifier):
    """What a built classifier is made of. A harmonic field reads as its envelope
    mode plus ``+fast`` (fast-weight overlay) and ``+linear`` (its own readout);
    a SequentialClassifier as the list of its stages; a ParallelClassifier as
    ``(stem, arms)``; any other classifier as its class name."""
    if isinstance(classifier, HarmonicClassifier):
        field = classifier.field
        return (
            field.amp_modulation
            + ("+fast" if field.fast_weights else "")
            + ("+linear" if classifier.scorer is not None else "")
        )
    if isinstance(classifier, SequentialClassifier):
        return [_describe(s) for s in classifier.stages]
    if isinstance(classifier, ParallelClassifier):
        stem = _describe(classifier.stem) if classifier.stem is not None else None
        return (stem, [_describe(b) for b in classifier.branches])
    return type(classifier).__name__


_PRISMATIC3_ARMS = [
    ["learned+fast+linear"],
    ["input+fast", "CrystalClassifier"],
    ["pure+fast+linear"],
]
_PRISMATIC4_ARMS = [
    ["learned+fast+linear"],
    ["input+fast", "CrystalVearClassifier"],
    ["pure+fast+linear"],
]
_STEM = "input+fast"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("crystal_harmonic", ["off", "CrystalClassifier"]),
        ("crystal_harmonic_static", ["static", "CrystalClassifier"]),
        # Bias arm (learned field, linear readout) vs variance arm (input field
        # into the crystal).
        ("prismatic", (None, [["learned+linear"], ["input", "CrystalClassifier"]])),
        ("prismatic3", (None, _PRISMATIC3_ARMS)),
        ("prismatic3_repel", (None, _PRISMATIC3_ARMS)),
        ("prismatic4", (None, _PRISMATIC4_ARMS)),
        ("prismatic5", (None, _PRISMATIC4_ARMS + ["HaloClassifier"])),
        # prismatic6 onward: one shared stem, arms that differ only in how they
        # read it. Each successor changes the geometric arm alone, so a delta
        # between neighbours attributes to that arm.
        (
            "prismatic6",
            (_STEM, ["CrystalVearClassifier", "LinearClassifier", "HaloClassifier"]),
        ),
        (
            "prismatic6_vear",
            (_STEM, ["CrystalVearClassifier", "LinearClassifier", "HaloClassifier"]),
        ),
        (
            "prismatic7",
            (_STEM, ["CrystalSmearClassifier", "LinearClassifier", "HaloClassifier"]),
        ),
        (
            "prismatic8",
            (_STEM, ["CrystalClassifier", "LinearClassifier", "HaloClassifier"]),
        ),
        (
            "prismatic9",
            (_STEM, ["CrystalClassifier", "LinearClassifier", "HaloClassifier"]),
        ),
    ],
)
def test_profile_wiring(name, expected):
    classifier = registry.lookup("classifiers", name)(Cfg(), encoder=Enc())
    assert _describe(classifier) == expected


def test_every_leaf_classifier_names_its_readout():
    """`compose_repr` is what the blueprint tab renders, and the base default
    is the CLASS name. That is only a readable label when the class name says
    what it computes (``CrystalClassifier``); anything else must override it,
    or it reads like a passthrough next to arms that name their function."""
    seen = set()
    for info in pkgutil.iter_modules(classifiers_pkg.__path__):
        m = importlib.import_module(f"praxis.classifiers.{info.name}")
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (
                issubclass(obj, BaseClassifier)
                and obj is not BaseClassifier
                and not inspect.isabstract(obj)
            ):
                seen.add(obj)

    missing = [
        c.__name__
        for c in seen
        if c.compose_repr is BaseClassifier.compose_repr
        and not c.__name__.endswith("Classifier")
    ]
    assert not missing, f"classifiers whose blueprint label names nothing: {missing}"
    assert seen, "no classifier classes discovered"
