"""Tests for praxis/spider/enrichers.py: the YouTube enricher."""

from praxis.kb.sources import LinksSource
from praxis.spider.enrichers import enricher_for


def test_youtube_enricher_mines_inline_json():
    html = (
        '{"videoId":"dQw4w9WgXcQ",'
        '"canonicalBaseUrl":"/@OtherChannel",'
        '"shortDescription":"A song.\\nMore: https://example.com"}'
    )
    e = enricher_for("https://youtube.com/@AnyChannel")
    assert e is not None
    out = e.enrich_html("https://youtube.com/@AnyChannel", html)
    assert "https://youtube.com/watch?v=dQw4w9WgXcQ" in out.links
    assert "https://youtube.com/@OtherChannel" in out.links
    assert "A song.\nMore: https://example.com" == out.text
    assert enricher_for("https://example.com/") is None


def test_youtube_link_veto():
    e = enricher_for("https://www.youtube.com/")
    # The point of the pipeline: whatever channel the README cites must pass
    # the veto - discovered, not hardcoded.
    readme_channels = [
        i.uri
        for i in LinksSource().iter_items()
        if i.origin == "README.md" and e.matches(i.uri)
    ]
    assert readme_channels, "README should cite at least one YouTube channel"
    assert all(e.link_allowed(u) for u in readme_channels)
    assert e.link_allowed("https://www.youtube.com/watch?v=abcdefghijk")
    assert e.link_allowed("https://www.youtube.com/channel/UCx")
    assert e.link_allowed("https://www.youtube.com/")
    assert not e.link_allowed("https://www.youtube.com/t/terms")
    assert not e.link_allowed("https://www.youtube.com/about/")
    assert not e.link_allowed("https://www.youtube.com/jobs/")
    assert not e.link_allowed("https://www.youtube.com/premium?x=1")


def test_blocked_shell_pages_are_errors():
    """A YouTube page with no minable JSON is a consent/blocked shell."""

    e = enricher_for("https://youtube.com/watch?v=abcdefghijk")
    out = e.enrich_html(
        "https://youtube.com/watch?v=abcdefghijk", "<html>About Press Copyright</html>"
    )
    assert not out.links and not out.text  # fetch_page treats this as an error
