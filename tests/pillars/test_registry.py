"""Sweeps over the pillars registries: every projection field renders, and
every paper thread is well formed."""

import pytest

from praxis import registry
from praxis.pillars.projections import render_card
from praxis.registry import Namespace


@pytest.mark.parametrize("name", list(registry.namespace("projections")))
def test_every_projection_renders(name, monkeypatch):
    """A card picks its field at random, so pin the namespace to this one."""
    field = registry.lookup("projections", name)
    monkeypatch.setitem(
        registry._NAMESPACES, "projections", Namespace("projections", {name: field})
    )
    for side in ("front", "back"):
        out = render_card(side, 7, "dark", 200, ["Ryan J. Brooks"], "", "abc")
        assert out.startswith(b"<?xml")


def test_the_named_threads_are_discovered():
    assert {"blind_watchmaking", "good_get_gooder"} <= set(registry.namespace("threads"))


@pytest.mark.parametrize("key", list(registry.namespace("threads")))
def test_every_thread_is_well_formed(key):
    from praxis.pillars.build import STEPS

    thread = registry.lookup("threads", key)
    assert thread.title and thread.pillars
    unknown = set(thread.pillars) - set(STEPS)
    assert not unknown, f"{key} names unknown steps: {unknown}"
