"""Tests for praxis/spider/fetch.py: HTML extraction and link normalization."""

from praxis.spider.fetch import _Extractor, _normalize_link

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
