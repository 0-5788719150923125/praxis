"""SpiderStore: the spider's store of record (``build/spider.db``).

Unlike kb.db, this database is never dropped on reindex - crawled pages must
survive. Three tables: the watchlist (sites), each site's crawl queue
(frontier), and extracted page content (pages).
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

from praxis.kb.sources import REPO_ROOT

DEFAULT_SPIDER_DB = REPO_ROOT / "build" / "spider.db"

# Consecutive failures before a site is disabled instead of retried forever.
MAX_ERROR_STREAK = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    url TEXT PRIMARY KEY,
    added REAL NOT NULL,
    last_fetch REAL NOT NULL DEFAULT 0,
    error_streak INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    -- Seeded from the repo's own documents; never evicted by promotion churn.
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS frontier (
    url TEXT PRIMARY KEY,
    site TEXT NOT NULL,
    depth INTEGER NOT NULL DEFAULT 0,
    discovered REAL NOT NULL,
    next_due REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    site TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    fetched REAL NOT NULL,
    etag TEXT NOT NULL DEFAULT '',
    last_modified TEXT NOT NULL DEFAULT ''
);
-- The link graph: who cites whom. Citation counts rank the frontier, so a
-- URL many pages point at is fetched before a one-off (likely bad) link.
CREATE TABLE IF NOT EXISTS refs (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);
-- Citations of sites we don't watch yet. Enough distinct referrers promotes
-- a site into the watchlist when there's free capacity.
CREATE TABLE IF NOT EXISTS external_refs (
    site TEXT NOT NULL,
    src TEXT NOT NULL,
    PRIMARY KEY (site, src)
);
-- One row per crawl outcome, the time series behind the dashboard cards.
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    site TEXT NOT NULL DEFAULT ''
);
"""

# Distinct referring pages before an unwatched site earns a watchlist slot.
PROMOTE_MIN_REFERRERS = 3


def site_of(url: str) -> str:
    """Canonical site key for a URL: scheme://host."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class SpiderStore:
    def __init__(self, db_path: Path = DEFAULT_SPIDER_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        # Pages must survive, so migrate in place rather than drop-and-rebuild.
        try:
            self._conn.execute(
                "ALTER TABLE sites ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # column already present
        self._conn.commit()

    # --- watchlist ---

    def add_site(self, url: str, max_sites: int, pinned: bool = False) -> bool:
        """Watch a site, evicting the stalest unpinned site when over the cap.
        The site root enters the frontier so the walk starts at the top."""
        site = site_of(url)
        if not urlsplit(site).netloc:
            return False
        now = time.time()
        deep = url.rstrip("/") != site  # seeded with a path, not a bare host
        existing = self._conn.execute(
            "SELECT 1 FROM sites WHERE url = ?", (site,)
        ).fetchone()
        if existing:
            if pinned:  # a seed re-encountered: protect it from churn
                self._conn.execute(
                    "UPDATE sites SET pinned = 1 WHERE url = ?", (site,)
                )
            if deep:  # make sure the cited page itself gets crawled
                self._conn.execute(
                    "INSERT OR IGNORE INTO frontier (url, site, depth, discovered) "
                    "VALUES (?, ?, 0, ?)",
                    (url, site, now),
                )
            self._conn.commit()
            return False
        count = self._conn.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
        if count >= max_sites:
            stale = self._conn.execute(
                "SELECT url FROM sites WHERE pinned = 0 ORDER BY last_fetch, added "
                "LIMIT 1"
            ).fetchone()
            if not stale:
                return False  # everything pinned; nothing to evict
            self.remove_site(stale[0])
        self._conn.execute(
            "INSERT INTO sites (url, added, pinned) VALUES (?, ?, ?)",
            (site, now, int(pinned)),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO frontier (url, site, depth, discovered) "
            "VALUES (?, ?, 0, ?)",
            (site + "/", site, now),
        )
        if deep:  # crawl the cited page itself, not just the site's root
            self._conn.execute(
                "INSERT OR IGNORE INTO frontier (url, site, depth, discovered) "
                "VALUES (?, ?, 0, ?)",
                (url, site, now),
            )
        self._conn.commit()
        return True

    def remove_site(self, site: str) -> None:
        site = site_of(site)
        for table in ("sites", "frontier", "pages"):
            col = "url" if table == "sites" else "site"
            self._conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (site,))
        self._conn.commit()

    def list_sites(self) -> List[tuple]:
        """(site, last_fetch, error_streak, enabled, pages_held) rows."""
        return self._conn.execute(
            "SELECT s.url, s.last_fetch, s.error_streak, s.enabled, "
            "(SELECT COUNT(*) FROM pages p WHERE p.site = s.url) "
            "FROM sites s ORDER BY s.added"
        ).fetchall()

    # --- frontier ---

    def next_url(self, domain_seconds: int, revisit_days: float) -> Optional[tuple]:
        """The best due (url, site, depth) to fetch, skipping sites hit within
        domain_seconds. Ranked by citation count - well-referenced URLs are
        fetched before one-off links, so bad links sink. Falls back to
        revisiting the stalest stored page once a site's frontier is dry -
        eventual consistency."""
        now = time.time()
        row = self._conn.execute(
            "SELECT f.url, f.site, f.depth, "
            "(SELECT COUNT(*) FROM refs r WHERE r.dst = f.url) AS cited "
            "FROM frontier f JOIN sites s "
            "ON s.url = f.site WHERE s.enabled = 1 AND f.next_due <= ? "
            "AND s.last_fetch <= ? "
            "ORDER BY cited DESC, f.next_due, f.discovered LIMIT 1",
            (now, now - domain_seconds),
        ).fetchone()
        if row:
            return row[:3]
        return self._conn.execute(
            "SELECT p.url, p.site, 0 FROM pages p JOIN sites s ON s.url = p.site "
            "WHERE s.enabled = 1 AND p.fetched <= ? AND s.last_fetch <= ? "
            "ORDER BY p.fetched LIMIT 1",
            (now - revisit_days * 86400, now - domain_seconds),
        ).fetchone()

    def extend_frontier(self, site: str, urls: List[str], depth: int, cap: int) -> int:
        """Queue same-site links breadth-first, bounded by the per-site cap
        across frontier + stored pages."""
        now = time.time()
        held = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM frontier WHERE site = ?) + "
            "(SELECT COUNT(*) FROM pages WHERE site = ?)",
            (site, site),
        ).fetchone()[0]
        added = 0
        for url in urls:
            if held + added >= cap:
                break
            known = self._conn.execute(
                "SELECT 1 FROM pages WHERE url = ? UNION "
                "SELECT 1 FROM frontier WHERE url = ?",
                (url, url),
            ).fetchone()
            if known:
                continue
            self._conn.execute(
                "INSERT INTO frontier (url, site, depth, discovered) "
                "VALUES (?, ?, ?, ?)",
                (url, site, depth, now),
            )
            added += 1
        self._conn.commit()
        return added

    # --- link graph ---

    def record_refs(self, src: str, dsts: List[str]) -> None:
        """Record citation edges from one fetched page. Same-site edges feed
        frontier ranking; edges to unwatched sites accrue toward promotion."""
        watched = {row[0] for row in self._conn.execute("SELECT url FROM sites")}
        for dst in dsts:
            dst_site = site_of(dst)
            if dst_site in watched:
                self._conn.execute(
                    "INSERT OR IGNORE INTO refs (src, dst) VALUES (?, ?)",
                    (src, dst),
                )
            else:
                self._conn.execute(
                    "INSERT OR IGNORE INTO external_refs (site, src) VALUES (?, ?)",
                    (dst_site, src),
                )
        self._conn.commit()

    def promote_sites(self, max_sites: int) -> List[str]:
        """Watch external sites cited by enough distinct pages - 'the spider
        crawled upon something it wanted to watch'. Free slots fill first;
        with a full watchlist a candidate must out-cite the least-interesting
        unpinned site, which it evicts. Pinned (seeded) sites never churn."""
        promoted = []
        while True:
            row = self._conn.execute(
                "SELECT site, COUNT(*) AS n FROM external_refs GROUP BY site "
                "HAVING n >= ? ORDER BY n DESC LIMIT 1",
                (PROMOTE_MIN_REFERRERS,),
            ).fetchone()
            if not row:
                return promoted
            site, citations = row
            free = (
                max_sites
                - self._conn.execute("SELECT COUNT(*) FROM sites").fetchone()[0]
            )
            if free <= 0:
                weakest = self._weakest_site()
                if not weakest or weakest[1] >= citations:
                    return promoted  # incumbents are more interesting
                self.remove_site(weakest[0])
                self.log_event("evicted", weakest[0])
            self._conn.execute("DELETE FROM external_refs WHERE site = ?", (site,))
            if self.add_site(site, max_sites):
                self.log_event("promoted", site)
                promoted.append(site)

    def _weakest_site(self) -> Optional[tuple]:
        """(site, score) for the least-interesting evictable site: inbound
        citations to its pages, zeroed for disabled (dead) sites."""
        return self._conn.execute(
            "SELECT s.url, CASE WHEN s.enabled = 0 THEN 0 ELSE "
            "(SELECT COUNT(*) FROM refs r JOIN pages p ON r.dst = p.url "
            " WHERE p.site = s.url) END AS score "
            "FROM sites s WHERE s.pinned = 0 ORDER BY score, s.last_fetch LIMIT 1"
        ).fetchone()

    def top_cited(self, limit: int = 5) -> List[tuple]:
        """(url, citations) of the most-referenced URLs."""
        return self._conn.execute(
            "SELECT dst, COUNT(*) AS n FROM refs GROUP BY dst "
            "ORDER BY n DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def top_referrers(self, limit: int = 5) -> List[tuple]:
        """(url, outbound_links) of the pages that cite the most URLs."""
        return self._conn.execute(
            "SELECT src, COUNT(*) AS n FROM refs GROUP BY src "
            "ORDER BY n DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # --- events / telemetry ---

    def log_event(self, event: str, site: str = "") -> None:
        self._conn.execute(
            "INSERT INTO events (ts, event, site) VALUES (?, ?, ?)",
            (time.time(), event, site),
        )
        self._conn.commit()

    def counts(self) -> dict:
        """Scalar snapshot for the dashboard: live table sizes plus cumulative
        crawl outcomes from the event log."""
        out = {
            "pages": self._conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
            "frontier": self._conn.execute("SELECT COUNT(*) FROM frontier").fetchone()[
                0
            ],
            "sites": self._conn.execute(
                "SELECT COUNT(*) FROM sites WHERE enabled = 1"
            ).fetchone()[0],
        }
        for event, n in self._conn.execute(
            "SELECT event, COUNT(*) FROM events GROUP BY event"
        ):
            out[event] = n
        return out

    # --- fetch outcomes ---

    def record_page(
        self,
        url: str,
        site: str,
        title: str,
        text: str,
        summary: str,
        etag: str = "",
        last_modified: str = "",
    ) -> None:
        now = time.time()
        known = self._conn.execute(
            "SELECT 1 FROM pages WHERE url = ?", (url,)
        ).fetchone()
        self._conn.execute(
            "INSERT INTO events (ts, event, site) VALUES (?, ?, ?)",
            (now, "revisit" if known else "new_page", site),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO pages "
            "(url, site, title, text, summary, fetched, etag, last_modified) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, site, title, text, summary, now, etag, last_modified),
        )
        self._conn.execute("DELETE FROM frontier WHERE url = ?", (url,))
        self._conn.execute(
            "UPDATE sites SET last_fetch = ?, error_streak = 0 WHERE url = ?",
            (now, site),
        )
        self._conn.commit()

    def record_unchanged(self, url: str, site: str) -> None:
        """A 304: refresh the page's clock without re-storing content."""
        now = time.time()
        self._conn.execute(
            "INSERT INTO events (ts, event, site) VALUES (?, 'unchanged', ?)",
            (now, site),
        )
        self._conn.execute("UPDATE pages SET fetched = ? WHERE url = ?", (now, url))
        self._conn.execute("DELETE FROM frontier WHERE url = ?", (url,))
        self._conn.execute(
            "UPDATE sites SET last_fetch = ?, error_streak = 0 WHERE url = ?",
            (now, site),
        )
        self._conn.commit()

    def record_error(self, url: str, site: str) -> None:
        """Exponential backoff per site; a long streak disables it."""
        now = time.time()
        self._conn.execute(
            "INSERT INTO events (ts, event, site) VALUES (?, 'error', ?)",
            (now, site),
        )
        streak = (
            self._conn.execute(
                "SELECT error_streak FROM sites WHERE url = ?", (site,)
            ).fetchone()
            or (0,)
        )[0] + 1
        enabled = 0 if streak >= MAX_ERROR_STREAK else 1
        self._conn.execute(
            "UPDATE sites SET last_fetch = ?, error_streak = ?, enabled = ? "
            "WHERE url = ?",
            (now, streak, enabled, site),
        )
        self._conn.execute(
            "UPDATE frontier SET next_due = ? WHERE url = ?",
            (now + min(3600 * 2**streak, 7 * 86400), url),
        )
        self._conn.commit()

    def conditional_headers(self, url: str) -> dict:
        row = self._conn.execute(
            "SELECT etag, last_modified FROM pages WHERE url = ?", (url,)
        ).fetchone()
        headers = {}
        if row:
            if row[0]:
                headers["If-None-Match"] = row[0]
            if row[1]:
                headers["If-Modified-Since"] = row[1]
        return headers

    # --- KB surfacing ---

    def recent_pages(self, limit: int) -> List[tuple]:
        """(url, site, title, text, summary, fetched), newest first."""
        return self._conn.execute(
            "SELECT url, site, title, text, summary, fetched FROM pages "
            "ORDER BY fetched DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def close(self) -> None:
        self._conn.close()
