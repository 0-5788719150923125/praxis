"""KBIndex: the searchable backend behind the bus.

SQLite FTS5 for v1. The method surface (rebuild / upsert / search) is the seam
where a Postgres + pgvector backend slots in later without touching callers.
"""

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from praxis.kb.item import KBHit, KBItem
from praxis.kb.sources import KB_SOURCE_REGISTRY, REPO_ROOT

DEFAULT_DB_PATH = REPO_ROOT / "build" / "kb.db"


class KBIndex:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        if read_only:
            uri = f"file:{self.db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        # External-content-less FTS5: id/type/title/uri/meta are searchable
        # columns too, but only body and title carry real ranking weight.
        self._conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS kb USING fts5(
                id UNINDEXED, type, label, title, body, uri UNINDEXED,
                meta UNINDEXED, updated UNINDEXED
            );
            """
        )
        self._conn.commit()

    def rebuild(self, sources: Optional[List[str]] = None) -> int:
        """Full reindex from the named sources (default: all registered)."""
        names = sources or list(KB_SOURCE_REGISTRY)
        self._conn.execute("DELETE FROM kb")
        count = 0
        for name in names:
            source = KB_SOURCE_REGISTRY[name]()
            count += self._insert(source.iter_items())
        self._conn.commit()
        return count

    def upsert(self, items: Iterable[KBItem]) -> int:
        """Incremental update - replace any items sharing an id, then insert."""
        items = list(items)
        ids = [it.id for it in items]
        self._conn.executemany("DELETE FROM kb WHERE id = ?", [(i,) for i in ids])
        count = self._insert(items)
        self._conn.commit()
        return count

    def _insert(self, items: Iterable[KBItem]) -> int:
        rows = [
            (it.id, it.type, it.label, it.title, it.body, it.uri,
             _encode_meta(it.meta), str(int(it.updated or 0)))
            for it in items
        ]
        self._conn.executemany(
            "INSERT INTO kb (id, type, label, title, body, uri, meta, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def search(
        self, query: str, types: Optional[List[str]] = None, limit: int = 20
    ) -> List[KBHit]:
        query = (query or "").strip()
        if not query:
            return []
        match = _to_match(query)
        sql = (
            "SELECT id, type, label, title, body, uri, meta, "
            "bm25(kb) AS score, snippet(kb, 4, '\x01', '\x02', '...', 12) AS snip "
            "FROM kb WHERE kb MATCH ?"
        )
        params: list = [match]
        if types:
            placeholders = ",".join("?" * len(types))
            sql += f" AND type IN ({placeholders})"
            params.extend(types)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        hits = []
        for row in self._conn.execute(sql, params):
            id_, type_, label, title, body, uri, meta, score, snip = row
            hits.append(
                KBHit(
                    item=KBItem(
                        id=id_,
                        type=type_,
                        label=label,
                        title=title,
                        body=body,
                        uri=uri,
                        meta=_decode_meta(meta),
                    ),
                    score=score,
                    snippet=snip or "",
                )
            )
        return hits

    def recent(
        self, limit: int = 20, types: Optional[List[str]] = None
    ) -> List[KBHit]:
        """Most-recently-updated items, newest first. Powers the empty-query
        default feed. Items with no timestamp (cards, agents) are excluded."""
        sql = (
            "SELECT id, type, label, title, body, uri, meta "
            "FROM kb WHERE CAST(updated AS INTEGER) > 0"
        )
        params: list = []
        if types:
            placeholders = ",".join("?" * len(types))
            sql += f" AND type IN ({placeholders})"
            params.extend(types)
        sql += " ORDER BY CAST(updated AS INTEGER) DESC LIMIT ?"
        params.append(limit)

        hits = []
        for row in self._conn.execute(sql, params):
            id_, type_, label, title, body, uri, meta = row
            hits.append(
                KBHit(
                    item=KBItem(
                        id=id_,
                        type=type_,
                        label=label,
                        title=title,
                        body=body,
                        uri=uri,
                        meta=_decode_meta(meta),
                    ),
                    score=0.0,
                    snippet="",
                )
            )
        return hits

    def get(self, item_id: str) -> Optional[KBItem]:
        """Fetch one item by id (for inline rendering of its full body)."""
        row = self._conn.execute(
            "SELECT id, type, label, title, body, uri, meta FROM kb WHERE id = ? LIMIT 1",
            (item_id,),
        ).fetchone()
        if not row:
            return None
        id_, type_, label, title, body, uri, meta = row
        return KBItem(
            id=id_,
            type=type_,
            label=label,
            title=title,
            body=body,
            uri=uri,
            meta=_decode_meta(meta),
        )

    def close(self) -> None:
        self._conn.close()


def _to_match(query: str) -> str:
    """Turn raw input into a prefix MATCH so search-as-you-type works.

    Each whitespace token becomes a quoted prefix term; the trailing token
    matches partial words (the keystroke in progress).
    """
    tokens = [t for t in query.replace('"', " ").split() if t]
    if not tokens:
        return query
    return " ".join(f'"{t}"*' for t in tokens)


def _encode_meta(meta: dict) -> str:
    import json

    return json.dumps(meta or {})


def _decode_meta(blob: str) -> dict:
    import json

    try:
        return json.loads(blob) if blob else {}
    except json.JSONDecodeError:
        return {}
