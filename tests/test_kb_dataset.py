"""KB-as-dataset sampler and incremental page indexing."""

from praxis.data.datasets.kb import KBDataset


def test_kb_dataset_yields_docs():
    ds = KBDataset(tokenizer=None, seed=0, config={"sources": ["docs"]})
    seqs = ds.get_sequences(3)
    assert all(isinstance(s, str) and s for s in seqs)


def test_kb_dataset_is_seeded():
    a = KBDataset(None, 7, {"sources": ["docs"]}).get_sequences(5)
    b = KBDataset(None, 7, {"sources": ["docs"]}).get_sequences(5)
    assert a == b


def test_kb_dataset_reloads_on_exhaustion():
    ds = KBDataset(None, 0, {"sources": ["docs"]})
    ds._load_epoch()
    n = len(ds._epoch)
    assert all(ds.get_sequences(n + 2))  # one wrap, no stall


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
    conn.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,'','')", rows)
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


def test_search_groups_widen_and_overlap_ranks_first(tmp_path):
    from praxis.kb.index import KBIndex
    from praxis.kb.item import KBItem

    idx = KBIndex(db_path=tmp_path / "kb.db")
    idx.upsert([
        KBItem(id="a", type="doc", label="W", title="alpha only", body="alpha", uri=""),
        KBItem(id="b", type="doc", label="W", title="beta only", body="beta", uri=""),
        KBItem(id="c", type="doc", label="W", title="alpha beta both", body="alpha beta", uri=""),
    ])
    hits = idx.search("alpha, beta", limit=10)
    ids = [h.item.id for h in hits]
    assert set(ids) == {"a", "b", "c"}  # OR widens to every group's matches
    assert ids[0] == "c"  # the AND-overlap ranks first
    idx.close()


def test_pages_boilerplate_dedup(tmp_path, monkeypatch):
    import sqlite3

    import praxis.spider.store as store_mod
    from praxis.kb.sources import PagesSource

    db = tmp_path / "spider.db"
    conn = sqlite3.connect(db)
    conn.executescript(store_mod._SCHEMA)
    chrome = "Skip to main content\nDonate\nAbout Help Contact"
    rows = [
        (f"https://a.com/{i}", "https://a.com", f"t{i}",
         f"{chrome}\nUnique abstract {i}", "s", float(i + 1))
        for i in range(4)
    ]
    conn.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,'','')", rows)
    conn.commit(); conn.close()
    monkeypatch.setattr("praxis.spider.store.DEFAULT_SPIDER_DB", db)
    items = list(PagesSource().iter_items())
    assert all("Donate" not in i.body for i in items)
    assert all(f"Unique abstract {n}" in items[3 - n].body for n in range(4))
