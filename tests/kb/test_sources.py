"""Tests for praxis/kb/sources.py: crawled pages from the spider store and the
code source."""

import sqlite3

import praxis.kb.sources as sources
import praxis.spider.store as store_mod
from praxis.kb.sources import PagesSource


def _spider_db(tmp_path, monkeypatch, rows):
    """A spider.db holding ``rows`` of (url, site, title, text, summary, fetched)."""
    db = tmp_path / "spider.db"
    conn = sqlite3.connect(db)
    conn.executescript(store_mod._SCHEMA)
    conn.executemany(
        "INSERT INTO pages (url, site, title, text, summary, fetched, etag, "
        "last_modified) VALUES (?,?,?,?,?,?,'','')",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(store_mod, "DEFAULT_SPIDER_DB", db)


def test_pages_source_since_filter(tmp_path, monkeypatch):
    rows = [
        (f"https://a.com/{i}", "https://a.com", f"t{i}", "x", "s", float(i))
        for i in (1, 2, 3)
    ]
    _spider_db(tmp_path, monkeypatch, rows)
    assert len(list(PagesSource().iter_items())) == 3
    assert [i.title for i in PagesSource().iter_items(since=2.0)] == ["t3"]


def test_pages_boilerplate_dedup(tmp_path, monkeypatch):
    chrome = "Skip to main content\nDonate\nAbout Help Contact"
    rows = [
        (
            f"https://a.com/{i}",
            "https://a.com",
            f"t{i}",
            f"{chrome}\nUnique abstract {i}",
            "s",
            float(i + 1),
        )
        for i in range(4)
    ]
    _spider_db(tmp_path, monkeypatch, rows)
    items = list(PagesSource().iter_items())
    assert all("Donate" not in i.body for i in items)
    assert all(f"Unique abstract {n}" in items[3 - n].body for n in range(4))


def test_code_source_indexes_package_files_and_skips_secrets(tmp_path, monkeypatch):
    pkg = tmp_path / "praxis" / "classifiers"
    pkg.mkdir(parents=True)
    (pkg / "ok.py").write_text(
        '"""A linear classifier."""\n\nclass Classifier:\n    pass\n'
    )
    (pkg / "bad.py").write_text('API_KEY = "abcdef123456789"\n')
    (tmp_path / "outside.py").write_text("x = 1\n")
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)

    items = {i.id: i for i in sources.CodeSource().iter_items()}
    assert set(items) == {"code:praxis/classifiers/ok.py"}
    item = items["code:praxis/classifiers/ok.py"]
    assert item.type == "code" and "class Classifier" in item.body
    assert item.summary == "A linear classifier."
