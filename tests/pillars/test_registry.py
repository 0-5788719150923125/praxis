import pytest

from praxis import registry
from praxis.pillars.projections import render_card

# ------------------------------------------------------------------------------
# cards
# ------------------------------------------------------------------------------


AUTHORS = ["Ryan J. Brooks"]
DONATE = "https://example.com/donate"


def test_every_field_renders():
    import praxis.pillars.projections as P

    full = dict(registry.namespace("projections"))
    try:
        for name, fn in full.items():
            registry.namespace("projections").clear()
            registry.namespace("projections")[name] = fn
            for side in ("front", "back"):
                out = render_card(side, 7, "dark", 200, AUTHORS, DONATE, "abc")
                assert out.startswith(b"<?xml"), name
    finally:
        registry.namespace("projections").clear()
        registry.namespace("projections").update(full)


# ------------------------------------------------------------------------------
# thread
# ------------------------------------------------------------------------------
# Paper threads: yaml-document layouts behind --title.


def test_registry_discovered_from_yaml_documents():
    assert "blind_watchmaking" in registry.namespace("threads")
    assert "good_get_gooder" in registry.namespace("threads")
    for thread in registry.namespace("threads").values():
        assert thread.title and thread.pillars


def test_pillars_reference_real_steps():
    from praxis.pillars.build import STEPS

    for thread in registry.namespace("threads").values():
        unknown = set(thread.pillars) - set(STEPS)
        assert not unknown, f"{thread.key} names unknown steps: {unknown}"
