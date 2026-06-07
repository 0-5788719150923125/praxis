"""SpiderWorker: the slow background loop, one fetch per tick.

Runs as a daemon thread beside the API server (rank 0 only). It only ever
writes spider.db; pages reach the KB index through the normal reindex path
(the "pages" source), never from this thread.
"""

import threading
import traceback

from praxis.spider import SpiderSettings
from praxis.spider.fetch import fetch_page
from praxis.spider.store import SpiderStore


class SpiderWorker:
    def __init__(self, settings: SpiderSettings, logger=None):
        self.settings = settings
        self.logger = logger
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="praxis-spider", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(f"[SPIDER] {message}")

    def _run(self) -> None:
        # The sqlite connection must be created on this thread.
        store = SpiderStore()
        robots_cache: dict = {}
        self._seed(store)
        while not self._stop.wait(self.settings.tick_seconds):
            try:
                self._tick(store, robots_cache)
            except Exception:
                self._log(f"tick failed:\n{traceback.format_exc()}")
        store.close()

    def _seed(self, store: SpiderStore) -> None:
        """Ground the watchlist in sites cited by the README, docs/, and
        next/. README links are the project's curated identity, so those are
        pinned - promotion churn can never evict them; doc citations compete
        like everything else."""
        from praxis.kb.sources import LinksSource

        added = 0
        for item in LinksSource().iter_items():
            pinned = item.origin == "README.md"
            if store.add_site(item.uri, self.settings.max_sites, pinned=pinned):
                added += 1
        if added:
            self._log(f"watching {added} new site(s) seeded from cited links")

    def _tick(self, store: SpiderStore, robots_cache: dict) -> None:
        s = self.settings
        due = store.next_url(s.domain_seconds, s.revisit_days)
        if not due:
            return
        url, site, depth = due
        try:
            result = fetch_page(
                url,
                max_bytes=s.max_page_bytes,
                conditional=store.conditional_headers(url),
                robots_cache=robots_cache,
            )
        except Exception as exc:
            store.record_error(url, site)
            self._log(f"backoff {url}: {exc}")
            return
        if result.status == 304:
            store.record_unchanged(url, site)
            return
        store.record_page(
            url,
            site,
            result.title,
            result.text,
            result.summary,
            etag=result.etag,
            last_modified=result.last_modified,
        )
        # Track every citation: same-site edges rank the frontier, edges to
        # unwatched sites accrue toward promotion. Only this site's own links
        # extend its frontier (the top-to-bottom walk stays per-site).
        store.record_refs(url, result.links)
        own = [u for u in result.links if u.startswith(site + "/") or u == site]
        queued = store.extend_frontier(site, own, depth + 1, s.max_pages_per_site)
        for promoted in store.promote_sites(s.max_sites):
            self._log(f"promoted {promoted} to the watchlist (widely cited)")
        self._log(f"fetched {url} (+{queued} queued)")
