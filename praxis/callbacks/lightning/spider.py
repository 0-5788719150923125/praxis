"""Mirror the spider's spider.db counters into the metrics stream.

The spider worker runs on its own slow clock in the API server thread; this
callback samples its store at logging intervals so the crawl shows up as
Research-tab cards (spider_pages / _revisits / _frontier / _sites). Read-only:
the worker thread stays the sole writer of spider.db.
"""

import sqlite3

from lightning.pytorch.callbacks import Callback

from praxis.spider.store import DEFAULT_SPIDER_DB


class SpiderCallback(Callback):
    def __init__(self, metrics_every: int = 50):
        self.metrics_every = metrics_every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.metrics_every != 0:
            return
        counts = self._counts()
        if counts is None:
            return
        scalars = {
            "spider_pages": float(counts.get("pages", 0)),
            "spider_new_pages": float(counts.get("new_page", 0)),
            "spider_revisits": float(
                counts.get("revisit", 0) + counts.get("unchanged", 0)
            ),
            "spider_frontier": float(counts.get("frontier", 0)),
            "spider_sites": float(counts.get("sites", 0)),
            "kb_size_mb": self._kb_size_mb(),
        }
        try:
            pl_module.log_dict(scalars, on_step=True, on_epoch=False, logger=True)
        except Exception:
            pass

    @staticmethod
    def _kb_size_mb() -> float:
        """On-disk footprint of the knowledge base in MB: the FTS index plus the
        spider's store of record, including their write-ahead logs."""
        from praxis.kb.index import DEFAULT_DB_PATH

        total = 0
        for db in (DEFAULT_DB_PATH, DEFAULT_SPIDER_DB):
            for path in (db, db.with_suffix(db.suffix + "-wal")):
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total / (1024 * 1024)

    @staticmethod
    def _counts():
        """Table sizes + cumulative event counts, or None before the worker
        has created the db. A fresh read-only connection per sample keeps this
        free of cross-thread connection sharing."""
        if not DEFAULT_SPIDER_DB.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{DEFAULT_SPIDER_DB}?mode=ro", uri=True)
            out = {}
            for table in ("pages", "frontier"):
                out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            out["sites"] = conn.execute(
                "SELECT COUNT(*) FROM sites WHERE enabled = 1"
            ).fetchone()[0]
            for event, n in conn.execute(
                "SELECT event, COUNT(*) FROM events GROUP BY event"
            ):
                out[event] = n
            conn.close()
            return out
        except sqlite3.Error:
            return None
