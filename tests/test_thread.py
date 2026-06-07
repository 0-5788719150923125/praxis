"""Paper threads: yaml-document layouts behind --title."""

import pytest

from praxis.pillars.thread import (
    DEFAULT_THREAD,
    THREAD_REGISTRY,
    resolve_thread,
)


def test_registry_discovered_from_yaml_documents():
    assert "blind_watchmaker" in THREAD_REGISTRY
    assert "good_get_gooder" in THREAD_REGISTRY
    for thread in THREAD_REGISTRY.values():
        assert thread.title and thread.pillars


def test_default_is_blind_watchmaker():
    s = resolve_thread(None)
    assert s.key == DEFAULT_THREAD
    assert s.title == "The Blind Watchmaker"
    assert "framing" in s.pillars and "proofs" in s.pillars
    assert s.abstract.startswith("Do you exist")


def test_gooder_stub_is_minimal():
    s = resolve_thread("good_get_gooder")
    assert s.pillars == ("runs", "evolution")
    assert "pretty good" in s.abstract
    assert s.theory  # plugs the body's \paperThreadTheory hook


def test_unknown_thread_raises():
    with pytest.raises(KeyError):
        resolve_thread("nonexistent_theory")


def test_pillars_reference_real_steps():
    from praxis.pillars.build import STEPS

    for thread in THREAD_REGISTRY.values():
        unknown = set(thread.pillars) - set(STEPS)
        assert not unknown, f"{thread.key} names unknown steps: {unknown}"


def test_write_thread_emits_component_macros(tmp_path, monkeypatch):
    import praxis.pillars.thread as sp

    monkeypatch.setattr(sp, "TITLE_TEX", str(tmp_path / "title.tex"))
    monkeypatch.setattr(sp, "THREAD_TEX", str(tmp_path / "thread.tex"))
    out = sp.write_thread(resolve_thread("good_get_gooder"))
    assert out["components"] == ["abstract", "theory"]
    assert "The Good Get Gooder Theorem" in (tmp_path / "title.tex").read_text()
    body = (tmp_path / "thread.tex").read_text()
    assert "\\newcommand{\\paperThreadAbstract}" in body
    assert "\\newcommand{\\paperThreadTheory}" in body
