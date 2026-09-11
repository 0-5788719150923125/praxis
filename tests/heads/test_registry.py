"""Sweeps over the ``heads`` registry: what each profile builds, and the blueprint
contract every head class honours."""

import importlib
import inspect
import pkgutil

import pytest

import praxis.heads as heads_pkg
from praxis import registry
from praxis.heads.base import BaseHead
from praxis.heads.harmonic import HarmonicHead
from praxis.heads.parallel import ParallelHead
from praxis.heads.stacked import SequentialHead
from tests.stubs import Cfg, Enc


def _describe(head):
    """What a built head is made of. A harmonic field reads as its envelope
    mode plus ``+fast`` (fast-weight overlay) and ``+linear`` (its own readout);
    a SequentialHead as the list of its stages; a ParallelHead as
    ``(stem, arms)``; any other head as its class name."""
    if isinstance(head, HarmonicHead):
        field = head.field
        return (
            field.amp_modulation
            + ("+fast" if field.fast_weights else "")
            + ("+linear" if head.lm_head is not None else "")
        )
    if isinstance(head, SequentialHead):
        return [_describe(h) for h in head.heads]
    if isinstance(head, ParallelHead):
        stem = _describe(head.stem) if head.stem is not None else None
        return (stem, [_describe(b) for b in head.branches])
    return type(head).__name__


_PRISMATIC3_ARMS = [
    ["learned+fast+linear"],
    ["input+fast", "CrystalHead"],
    ["pure+fast+linear"],
]
_PRISMATIC4_ARMS = [
    ["learned+fast+linear"],
    ["input+fast", "CrystalVearHead"],
    ["pure+fast+linear"],
]
_STEM = "input+fast"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("crystal_harmonic", ["off", "CrystalHead"]),
        ("crystal_harmonic_static", ["static", "CrystalHead"]),
        # Bias arm (learned field, linear readout) vs variance arm (input field
        # into the crystal).
        ("prismatic", (None, [["learned+linear"], ["input", "CrystalHead"]])),
        ("prismatic3", (None, _PRISMATIC3_ARMS)),
        ("prismatic3_repel", (None, _PRISMATIC3_ARMS)),
        ("prismatic4", (None, _PRISMATIC4_ARMS)),
        ("prismatic5", (None, _PRISMATIC4_ARMS + ["HaloHead"])),
        # prismatic6 onward: one shared stem, arms that differ only in how they
        # read it. Each successor changes the geometric arm alone, so a delta
        # between neighbours attributes to that arm.
        ("prismatic6", (_STEM, ["CrystalVearHead", "ForwardHead", "HaloHead"])),
        ("prismatic6_vear", (_STEM, ["CrystalVearHead", "ForwardHead", "HaloHead"])),
        ("prismatic7", (_STEM, ["CrystalSmearHead", "ForwardHead", "HaloHead"])),
        ("prismatic8", (_STEM, ["CrystalHead", "ForwardHead", "HaloHead"])),
        ("prismatic9", (_STEM, ["CrystalHead", "ForwardHead", "HaloHead"])),
    ],
)
def test_profile_wiring(name, expected):
    head = registry.lookup("heads", name)(Cfg(), encoder=Enc())
    assert _describe(head) == expected


def test_every_leaf_head_names_its_readout():
    """`compose_repr` is what the blueprint tab renders, and the base default
    falls back to the CLASS name - which reads like a passthrough or a leftover
    default next to arms that name their function."""
    seen = set()
    for info in pkgutil.iter_modules(heads_pkg.__path__):
        m = importlib.import_module(f"praxis.heads.{info.name}")
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if (
                issubclass(obj, BaseHead)
                and obj is not BaseHead
                and not inspect.isabstract(obj)
            ):
                seen.add(obj)

    missing = [c.__name__ for c in seen if c.compose_repr is BaseHead.compose_repr]
    assert not missing, f"leaf heads falling back to their class name: {missing}"
    assert seen, "no head classes discovered"
