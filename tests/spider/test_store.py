"""Tests for praxis/spider/store.py: the watchlist, frontier, link graph,
promotion and event log."""

import pytest

from praxis.spider.store import SpiderStore, site_of


@pytest.fixture
def store(tmp_path):
    s = SpiderStore(db_path=tmp_path / "spider.db")
    yield s
    s.close()


def test_site_of_normalizes():
    assert site_of("https://example.com/a/b?q=1") == "https://example.com"


def test_add_site_seeds_frontier(store):
    assert store.add_site("https://example.com/page", max_sites=4)
    url, site, depth = store.next_url(domain_seconds=0, revisit_days=7)
    assert (url, site, depth) == ("https://example.com/", "https://example.com", 0)


def test_watchlist_cap_evicts_stalest(store):
    store.add_site("https://a.com", max_sites=2)
    store.add_site("https://b.com", max_sites=2)
    store.add_site("https://c.com", max_sites=2)
    sites = [row[0] for row in store.list_sites()]
    assert len(sites) == 2 and "https://c.com" in sites


def test_domain_spacing_blocks_recent_site(store):
    store.add_site("https://a.com", max_sites=4)
    store.record_page("https://a.com/", "https://a.com", "t", "x", "s")
    assert store.next_url(domain_seconds=3600, revisit_days=7) is None


def test_frontier_cap_and_dedupe(store):
    store.add_site("https://a.com", max_sites=4)
    added = store.extend_frontier(
        "https://a.com",
        [f"https://a.com/p{i}" for i in range(10)] + ["https://a.com/"],
        depth=1,
        cap=5,
    )
    assert added == 4  # root already queued; cap covers frontier + pages


def test_error_backoff_disables_after_streak(store):
    store.add_site("https://a.com", max_sites=4)
    for _ in range(8):
        store.record_error("https://a.com/", "https://a.com")
    _, _, streak, enabled, _ = store.list_sites()[0]
    assert streak == 8 and enabled == 0


def test_revisit_when_frontier_dry(store):
    store.add_site("https://a.com", max_sites=4)
    store.record_page("https://a.com/", "https://a.com", "t", "x", "s")
    # Frontier is empty; with a zero revisit window the stored page is due.
    url, site, _ = store.next_url(domain_seconds=0, revisit_days=0)
    assert url == "https://a.com/"


def test_conditional_headers_roundtrip(store):
    store.add_site("https://a.com", max_sites=4)
    store.record_page(
        "https://a.com/",
        "https://a.com",
        "t",
        "x",
        "s",
        etag='W/"abc"',
        last_modified="Mon, 01 Jan 2026 00:00:00 GMT",
    )
    headers = store.conditional_headers("https://a.com/")
    assert headers["If-None-Match"] == 'W/"abc"'
    assert "If-Modified-Since" in headers


# --- link graph / promotion / events ---


def test_citations_rank_the_frontier(store):
    store.add_site("https://a.com", max_sites=4)
    store.extend_frontier(
        "https://a.com",
        ["https://a.com/popular", "https://a.com/oneoff"],
        depth=1,
        cap=50,
    )
    for src in ("https://a.com/p1", "https://a.com/p2"):
        store.record_refs(src, ["https://a.com/popular"])
    url, _, _ = store.next_url(domain_seconds=0, revisit_days=7)
    assert url == "https://a.com/popular"


def test_external_refs_promote_into_free_slots(store):
    store.add_site("https://a.com", max_sites=8)
    store.add_site("https://b.com", max_sites=8)
    store.add_site("https://c.com", max_sites=8)
    for src in ("https://a.com/1", "https://b.com/2", "https://c.com/3"):
        store.record_refs(src, ["https://news.example/story"])
    promoted = store.promote_sites(max_sites=8)
    assert promoted == ["https://news.example"]
    assert "https://news.example" in [row[0] for row in store.list_sites()]


def test_same_site_referrers_do_not_promote(store):
    """A single site's footer boilerplate citing the same external site on
    every page must not promote it - that's the churn-ring failure mode."""
    store.add_site("https://a.com", max_sites=8)
    for src in ("https://a.com/1", "https://a.com/2", "https://a.com/3"):
        store.record_refs(src, ["https://social.example/share"])
    assert store.promote_sites(max_sites=8) == []


def test_evicted_sites_wait_out_a_cooldown(store):
    store.add_site("https://a.com", max_sites=8)
    store.add_site("https://b.com", max_sites=8)
    store.add_site("https://c.com", max_sites=8)
    store.log_event("evicted", "https://news.example")
    for src in ("https://a.com/1", "https://b.com/2", "https://c.com/3"):
        store.record_refs(src, ["https://news.example/story"])
    assert store.promote_sites(max_sites=8) == []


def test_events_and_counts(store):
    store.add_site("https://a.com", max_sites=4)
    store.record_page("https://a.com/", "https://a.com", "t", "x", "s")
    store.record_page("https://a.com/", "https://a.com", "t", "x2", "s")
    store.record_unchanged("https://a.com/", "https://a.com")
    counts = store.counts()
    assert counts["new_page"] == 1
    assert counts["revisit"] == 1
    assert counts["unchanged"] == 1
    assert counts["pages"] == 1
    assert counts["sites"] == 1


def test_top_cited_and_referrers(store):
    store.add_site("https://a.com", max_sites=4)
    store.record_refs("https://a.com/hub", ["https://a.com/x", "https://a.com/y"])
    store.record_refs("https://a.com/p", ["https://a.com/x"])
    assert store.top_cited(1) == [("https://a.com/x", 2)]
    assert store.top_referrers(1) == [("https://a.com/hub", 2)]


# --- pinning / competitive promotion ---


def test_pinned_sites_survive_eviction(store):
    store.add_site("https://pinned.com", max_sites=2, pinned=True)
    store.add_site("https://stale.com", max_sites=2)
    store.add_site("https://new.com", max_sites=2)  # cap hit: evicts unpinned
    sites = {row[0] for row in store.list_sites()}
    assert "https://pinned.com" in sites and "https://stale.com" not in sites


def test_add_site_refuses_when_all_pinned(store):
    store.add_site("https://a.com", max_sites=1, pinned=True)
    assert not store.add_site("https://b.com", max_sites=1)


def test_promotion_evicts_weakest_when_full(store):
    store.add_site("https://pinned.com", max_sites=2, pinned=True)
    store.add_site("https://boring.com", max_sites=2)  # zero inbound citations
    for i in range(3):
        store.record_refs(f"https://x{i}.com/p", ["https://hot.com/a"])
    promoted = store.promote_sites(max_sites=2)
    sites = {row[0] for row in store.list_sites()}
    assert promoted == ["https://hot.com"]
    assert "https://boring.com" not in sites and "https://pinned.com" in sites


def test_promotion_spares_more_interesting_incumbents(store):
    store.add_site("https://liked.com", max_sites=1)
    store.record_page("https://liked.com/a", "https://liked.com", "t", "x", "s")
    for i in range(9):
        store.record_refs(f"https://r.com/{i}", ["https://liked.com/a"])
    for i in range(3):
        store.record_refs(f"https://x{i}.com/p", ["https://meh.com/a"])
    assert store.promote_sites(max_sites=1) == []
    assert {row[0] for row in store.list_sites()} == {"https://liked.com"}


def test_deep_seed_queues_cited_page(store):
    store.add_site("https://yt.com/@Chan", max_sites=4)
    urls = {r[0] for r in store._conn.execute("SELECT url FROM frontier")}
    assert "https://yt.com/" in urls and "https://yt.com/@Chan" in urls
    store.add_site("https://yt.com/@Other", max_sites=4)  # existing site, new path
    urls = {r[0] for r in store._conn.execute("SELECT url FROM frontier")}
    assert "https://yt.com/@Other" in urls
