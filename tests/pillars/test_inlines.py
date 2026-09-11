"""The measured values the paper drops into its own prose.

Each of these reports a reading the body promises: the copy gain a frozen
attention geometry is predicted not to earn, the bottleneck's second pole, and
the density profile's steepening. A format string is prose, so it is pinned -
a typo here reaches the PDF - and so is the rule that an unmeasured run leaves
its paragraph alone.
"""

import pytest

from praxis.pillars.inlines import INLINES, PROVIDERS

MEASURED = {
    "copy-gain": ("copy_gain", 0.37),
    "trunk-swap": ("trunk_swap", 0.0006),
    "depth-tilt": ("depth_tilt_slope", -0.12),
}


@pytest.mark.parametrize("edit_id", sorted(MEASURED))
def test_a_measured_value_lands_in_the_sentence(edit_id, monkeypatch):
    provider, value = MEASURED[edit_id]
    monkeypatch.setitem(PROVIDERS, provider, lambda: value)
    body = INLINES[edit_id].resolve()
    assert body.startswith(" ")  # joins the sentence before it
    assert "{value" not in body  # the format string was applied
    assert "current run" in body


@pytest.mark.parametrize("edit_id", sorted(MEASURED))
def test_an_unmeasured_run_leaves_the_paragraph_alone(edit_id, monkeypatch):
    provider, _ = MEASURED[edit_id]
    monkeypatch.setitem(PROVIDERS, provider, lambda: None)
    assert INLINES[edit_id].resolve() is None
    assert INLINES[edit_id].fallback == ""


def test_the_field_instance_names_who_builds_the_field(monkeypatch):
    """Section 3.2 describes a multiplicative field. The sentence that says
    whose instance it is has to follow the registry, not the profile's family."""
    import praxis.pillars.framing as framing

    monkeypatch.setattr(framing, "newest_experiment", lambda: "run")
    for carrier, expected in (
        ("classifier", "builds that field"),
        ("encoder", "builds no such field"),
        ("none", "builds neither"),
    ):
        monkeypatch.setattr(
            framing, "resolve_config", lambda _e, c=carrier: {"field_carrier": c}
        )
        assert expected in INLINES["field-instance"].resolve()
