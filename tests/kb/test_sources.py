"""KB-as-dataset sampler and incremental page indexing."""


def test_pages_source_since_filter(tmp_path, monkeypatch):
    import sqlite3

    import praxis.spider.store as store_mod
    from praxis.kb.sources import PagesSource

    db = tmp_path / "spider.db"
    conn = sqlite3.connect(db)
    conn.executescript(store_mod._SCHEMA)
    rows = [
        (f"https://a.com/{i}", "https://a.com", f"t{i}", "x", "s", float(i))
        for i in (1, 2, 3)
    ]
    conn.executemany(
        "INSERT INTO pages (url, site, title, text, summary, fetched, etag, "
        "last_modified) VALUES (?,?,?,?,?,?,'','')",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("praxis.spider.store.DEFAULT_SPIDER_DB", db)
    assert len(list(PagesSource().iter_items())) == 3
    assert [i.title for i in PagesSource().iter_items(since=2.0)] == ["t3"]


def test_code_source_indexes_main_paths():
    from praxis.kb.sources import CodeSource

    items = {i.id: i for i in CodeSource().iter_items()}
    assert "code:praxis/heads/energy.py" in items
    it = items["code:praxis/heads/energy.py"]
    assert it.type == "code" and "class" in it.body
    assert all(i.id.startswith("code:praxis/") for i in items.values())


def test_code_source_skips_secretish_files(tmp_path, monkeypatch):
    import praxis.kb.sources as src

    pkg = tmp_path / "praxis"
    pkg.mkdir()
    (pkg / "ok.py").write_text("x = 1\n")
    (pkg / "bad.py").write_text('API_KEY = "abcdef123456789"\n')
    monkeypatch.setattr(src, "REPO_ROOT", tmp_path)
    ids = {i.id for i in src.CodeSource().iter_items()}
    assert ids == {"code:praxis/ok.py"}


def test_pages_boilerplate_dedup(tmp_path, monkeypatch):
    import sqlite3

    import praxis.spider.store as store_mod
    from praxis.kb.sources import PagesSource

    db = tmp_path / "spider.db"
    conn = sqlite3.connect(db)
    conn.executescript(store_mod._SCHEMA)
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
    conn.executemany(
        "INSERT INTO pages (url, site, title, text, summary, fetched, etag, "
        "last_modified) VALUES (?,?,?,?,?,?,'','')",
        rows,
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("praxis.spider.store.DEFAULT_SPIDER_DB", db)
    items = list(PagesSource().iter_items())
    assert all("Donate" not in i.body for i in items)
    assert all(f"Unique abstract {n}" in items[3 - n].body for n in range(4))
