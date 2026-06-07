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
    rows = [(f"https://a.com/{i}", "https://a.com", f"t{i}", "x", "s", float(i)) for i in (1, 2, 3)]
    conn.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?,'','')", rows)
    conn.commit()
    conn.close()
    monkeypatch.setattr("praxis.spider.store.DEFAULT_SPIDER_DB", db)
    assert len(list(PagesSource().iter_items())) == 3
    assert [i.title for i in PagesSource().iter_items(since=2.0)] == ["t3"]
