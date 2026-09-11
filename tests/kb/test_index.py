import pytest

from praxis.kb import KBIndex, KBItem

# ------------------------------------------------------------------------------
# kb
# ------------------------------------------------------------------------------
# KB provenance schema and index roundtrip.


@pytest.fixture
def index(tmp_path):
    ix = KBIndex(db_path=tmp_path / "kb.db")
    yield ix
    ix.close()


def _item(**overrides):
    base = dict(
        id="doc:test",
        type="doc",
        label="Wiki",
        title="Test Doc",
        body="# Heading\nFirst real line of content.\nSecond line.",
        uri="docs/test.md",
        updated=100.0,
    )
    base.update(overrides)
    return KBItem(**base)


def test_index_roundtrip_preserves_provenance(index):
    index.upsert([_item(source="docs", origin="docs/test.md", summary="a summary")])
    item = index.get("doc:test")
    assert item.source == "docs"
    assert item.origin == "docs/test.md"
    assert item.summary == "a summary"


def test_search_and_feeds_carry_summary(index):
    index.upsert([_item(source="docs", summary="the one-liner")])
    hit = index.search("content")[0]
    assert hit.item.summary == "the one-liner"
    assert index.list_all()[0].snippet == "the one-liner"
    assert index.recent()[0].snippet == "the one-liner"


def test_old_schema_is_dropped_and_recreated(tmp_path):
    import sqlite3

    db = tmp_path / "kb.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE VIRTUAL TABLE kb USING fts5(id UNINDEXED, type, label, title, "
        "body, uri UNINDEXED, meta UNINDEXED, updated UNINDEXED)"
    )
    conn.commit()
    conn.close()

    ix = KBIndex(db_path=db)
    ix.upsert([_item(origin="docs/test.md")])
    assert ix.get("doc:test").origin == "docs/test.md"
    ix.close()


# ------------------------------------------------------------------------------
# kb_dataset
# ------------------------------------------------------------------------------
# KB-as-dataset sampler and incremental page indexing.


def test_search_groups_widen_and_overlap_ranks_first(tmp_path):
    from praxis.kb.index import KBIndex
    from praxis.kb.item import KBItem

    idx = KBIndex(db_path=tmp_path / "kb.db")
    idx.upsert(
        [
            KBItem(
                id="a", type="doc", label="W", title="alpha only", body="alpha", uri=""
            ),
            KBItem(
                id="b", type="doc", label="W", title="beta only", body="beta", uri=""
            ),
            KBItem(
                id="c",
                type="doc",
                label="W",
                title="alpha beta both",
                body="alpha beta",
                uri="",
            ),
        ]
    )
    hits = idx.search("alpha, beta", limit=10)
    ids = [h.item.id for h in hits]
    assert set(ids) == {"a", "b", "c"}  # OR widens to every group's matches
    assert ids[0] == "c"  # the AND-overlap ranks first
    idx.close()
