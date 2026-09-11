import pytest

from praxis import registry
from praxis.heads import ParallelHead
from praxis.heads.harmonic import HarmonicHead

# ------------------------------------------------------------------------------
# harmonic_modulation
# ------------------------------------------------------------------------------
# Amplitude modulation envelope on the harmonic field (off|static|learned).


def test_head_type_keys_compose_sequential_heads():
    # The single-field harmonic+crystal keys are functools.partial over
    # SequentialHead, composing [HarmonicHead(mode, transform-only), CrystalHead]
    # dynamically - no bespoke subclass. The mode lives in the harmonic builder's
    # keywords.
    import functools

    from praxis.heads import CrystalHead, HarmonicHead
    from praxis.heads.stacked import SequentialHead

    for key, mode in [
        ("crystal_harmonic", "off"),
        ("crystal_harmonic_static", "static"),
    ]:
        entry = registry.lookup("heads", key)
        assert isinstance(entry, functools.partial)
        assert entry.func is SequentialHead
        harmonic_spec, crystal_spec = entry.keywords["heads"]
        assert crystal_spec is CrystalHead
        assert harmonic_spec.func is HarmonicHead
        assert harmonic_spec.keywords["amp_modulation"] == mode
        assert harmonic_spec.keywords["build_classifier"] is False


def test_prismatic_is_top_level_parallel_split():
    # prismatic is a top-level Parallel of two arms balancing bias vs variance:
    #   Parallel(Sequential(HarmonicField), Sequential(HarmonicField, CrystalClassifier))
    import functools

    from praxis.heads import CrystalHead, HarmonicHead, ParallelHead
    from praxis.heads.stacked import SequentialHead

    entry = registry.lookup("heads", "prismatic")
    assert isinstance(entry, functools.partial) and entry.func is ParallelHead
    arm0, arm1 = entry.keywords["branches"]
    assert arm0.func is SequentialHead and arm1.func is SequentialHead

    # arm 0 (bias): a single harmonic field with its own linear readout.
    (field0,) = arm0.keywords["heads"]
    assert field0.func is HarmonicHead
    assert field0.keywords["amp_modulation"] == "learned"
    assert field0.keywords["build_classifier"] is True

    # arm 1 (variance): a transform-only field feeding the crystal classifier.
    field1, crystal = arm1.keywords["heads"]
    assert field1.func is HarmonicHead
    assert field1.keywords["build_classifier"] is False
    assert crystal is CrystalHead


# ------------------------------------------------------------------------------
# parallel_head
# ------------------------------------------------------------------------------
# ParallelHead: gated parallel branches + namespaced per-branch dashboards.


# ── the blueprint repr ─────────────────────────────────────────────────────


def test_every_leaf_head_names_its_readout():
    """`compose_repr` is what the blueprint tab renders, and the base default
    falls back to the CLASS name. Two leaves never overrode it, so prismatic6-9
    rendered as `[CrystalClassifier, ForwardHead, HaloClassifier]` - one arm
    naming its class where the others name their function, which reads like a
    passthrough or a leftover default instead of the linear readout that is the
    deliberate control arm."""
    import importlib
    import inspect
    import pkgutil

    import praxis.heads as heads_pkg
    from praxis.heads.base import BaseHead

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
