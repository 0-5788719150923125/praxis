"""Spider settings resolution, store behavior, and HTML extraction."""

import pytest

from praxis.spider import SpiderSettings, spider_settings
from praxis.spider.fetch import _Extractor, _normalize_link
from praxis.spider.store import SpiderStore, site_of

# --- settings ---


def test_disabled_when_flag_absent():
    assert spider_settings(None) is None


def test_bare_flag_uses_gentle_defaults():
    settings = spider_settings([])
    assert settings.profile == "gentle"
    assert settings.tick_seconds == 300


def test_key_value_overrides():
    settings = spider_settings(["profile=brisk", "max_sites=4"])
    assert settings.profile == "brisk"
    assert settings.max_sites == 4
    assert settings.tick_seconds == 15  # from brisk


def test_dict_entries_from_yml():
    settings = spider_settings({"tick_seconds": "600"})
    assert settings.tick_seconds == 600


def test_fractional_revisit_days():
    settings = spider_settings(["tick_seconds=3600", "revisit_days=0.05"])
    assert settings.revisit_days == 0.05
    assert settings.tick_seconds == 3600


@pytest.mark.parametrize("bad", [["nope"], ["profile=missing"], ["bogus=1"]])
def test_invalid_entries_raise(bad):
    with pytest.raises(ValueError):
        spider_settings(bad)


# --- store ---


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


# --- extraction ---

_HTML = """
<html><head><title>A Page</title>
<meta name="description" content="What this page is about.">
<style>.x{color:red}</style></head>
<body><script>var hidden = 1;</script>
<h1>Welcome</h1><p>Some prose here.</p>
<a href="/local">in</a>
<a href="https://other.com/away">out</a>
<a href="/local#frag">dupe</a>
</body></html>
"""


def test_extractor_title_text_description():
    ex = _Extractor()
    ex.feed(_HTML)
    assert ex.title == "A Page"
    assert ex.description == "What this page is about."
    assert "Some prose here." in ex.text
    assert "hidden" not in ex.text and "color:red" not in ex.text


def test_normalize_link_keeps_cross_site():
    base = "https://example.com/dir/page"
    assert _normalize_link("/local", base) == "https://example.com/local"
    assert _normalize_link("/local#frag", base) == "https://example.com/local"
    assert _normalize_link("https://other.com/x", base) == "https://other.com/x"
    assert _normalize_link("mailto:x@y.z", base) is None


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
    store.add_site("https://a.com", max_sites=4)
    for src in ("https://a.com/1", "https://a.com/2", "https://a.com/3"):
        store.record_refs(src, ["https://news.example/story"])
    promoted = store.promote_sites(max_sites=4)
    assert promoted == ["https://news.example"]
    assert "https://news.example" in [row[0] for row in store.list_sites()]


def test_promotion_never_evicts_when_full(store):
    store.add_site("https://a.com", max_sites=1)
    for src in ("https://a.com/1", "https://a.com/2", "https://a.com/3"):
        store.record_refs(src, ["https://news.example/story"])
    assert store.promote_sites(max_sites=1) == []
    assert [row[0] for row in store.list_sites()] == ["https://a.com"]


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
