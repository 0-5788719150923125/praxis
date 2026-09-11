"""Paper threads: yaml-document layouts behind --title."""

import pytest

from praxis.pillars.thread import DEFAULT_THREAD, resolve_thread


def test_default_is_blind_watchmaking():
    s = resolve_thread(None)
    assert s.key == DEFAULT_THREAD
    assert s.title == "Blind Watchmaking"
    assert "framing" in s.pillars and "proofs" in s.pillars
    assert s.component("abstract").startswith("Do you exist")
    assert s.component("introduction").startswith("The dominant recipe")
    assert s.component("conclusion").startswith("We began with scale")


def test_unknown_thread_raises():
    with pytest.raises(KeyError):
        resolve_thread("nonexistent_theory")


def test_write_thread_emits_component_macros(tmp_path, monkeypatch):
    import praxis.pillars.thread as sp

    monkeypatch.setattr(sp, "TITLE_TEX", str(tmp_path / "title.tex"))
    monkeypatch.setattr(sp, "THREAD_TEX", str(tmp_path / "thread.tex"))
    out = sp.write_thread(resolve_thread("good_get_gooder"))
    assert out["components"] == ["abstract", "introduction", "theory", "conclusion"]
    assert "The Good Get Gooder Theorem" in (tmp_path / "title.tex").read_text()
    body = (tmp_path / "thread.tex").read_text()
    assert "\\newcommand{\\paperThreadAbstract}" in body
    assert "\\newcommand{\\paperThreadIntroduction}" in body
    assert "\\newcommand{\\paperThreadTheory}" in body
    assert "\\newcommand{\\paperThreadConclusion}" in body
