"""Freezing the dashboard so a terminal text selection survives.

The terminal owns the selection, not this process: there is no escape sequence
that preserves one across a rewrite of the selected cells. Differential
rendering does not help in the LOGS panel either, because that panel SCROLLS -
every line in it changes on every frame even when the text is the same. So the
only thing that actually keeps a selection alive is not writing at all, and
these tests pin that: frozen means zero writes, and nothing is lost while it
holds.
"""

import types


class _Key(str):
    """A blessed keystroke, near enough: a str with a ``name``."""

    @property
    def name(self):
        return None


def _capture(dashboard):
    writes = []
    dashboard.dashboard_output = types.SimpleNamespace(
        write=lambda s: writes.append(s), flush=lambda: None
    )
    return writes


def test_f_toggles_frozen(dashboard):
    assert dashboard.frozen is False
    _capture(dashboard)
    dashboard._handle_keyboard_input(_Key("f"))
    assert dashboard.frozen is True
    dashboard._handle_keyboard_input(_Key("F"))
    assert dashboard.frozen is False


def test_nothing_is_written_while_frozen(dashboard):
    """The whole feature. One write is allowed at the moment of freezing - the
    banner, on the bottom border row - and none after it."""
    writes = _capture(dashboard)
    dashboard._handle_keyboard_input(_Key("f"))
    settled = len(writes)

    for i in range(25):
        dashboard.add_log(f"line {i}")

    assert len(writes) == settled, f"{len(writes) - settled} writes while frozen"


def test_logs_still_accumulate_while_frozen(dashboard):
    """Freezing holds the SCREEN still, not the run. A frozen dashboard that
    dropped log lines would trade one copying problem for a worse one."""
    _capture(dashboard)
    dashboard._handle_keyboard_input(_Key("f"))
    before = len(dashboard.log_buffer)
    for i in range(25):
        dashboard.add_log(f"line {i}")
    assert len(dashboard.log_buffer) > before


def test_unfreezing_resyncs_the_screen(dashboard):
    """The screen is stale by definition after a freeze, so it is repainted in
    full - without a clear, so there is no flicker."""
    _capture(dashboard)
    dashboard._handle_keyboard_input(_Key("f"))
    dashboard.differential_renderer.repaint_pending = False
    dashboard._handle_keyboard_input(_Key("f"))
    assert dashboard.frozen is False
    assert dashboard.differential_renderer.repaint_pending is True


def test_freeze_works_from_fullscreen_log_mode(dashboard):
    """Copying happens in the log view, so the key has to work there - it is
    checked before the fullscreen branch swallows everything."""
    _capture(dashboard)
    dashboard.fullscreen_log_mode = True
    dashboard._handle_keyboard_input(_Key("f"))
    assert dashboard.frozen is True
    assert dashboard.fullscreen_log_mode is True, "freezing changed the mode"
