"""Polite single-page fetching and stdlib HTML extraction.

One conditional GET per call: robots.txt is honored (and cached), responses
are capped by size and restricted to text/html, and a 304 is a first-class
result so watching a stable page costs the site almost nothing.
"""

import gzip
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

from praxis.spider.enrichers import enricher_for

USER_AGENT = "praxis-spider/0.1 (+https://github.com/0-5788719150923125/praxis)"
TIMEOUT = 30

# Tags whose text content is noise, not prose.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}


@dataclass
class FetchResult:
    status: int  # HTTP status; 304 = unchanged
    title: str = ""
    text: str = ""
    summary: str = ""
    links: List[str] = field(default_factory=list)
    etag: str = ""
    last_modified: str = ""


class _Extractor(HTMLParser):
    """Title, readable text, meta description, and hrefs from one page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.links: List[str] = []
        self._chunks: List[str] = []
        self._stack: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP_TAGS:
            self._stack.append(tag)
        elif tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        elif tag == "meta" and attrs.get("name") == "description":
            self.description = (attrs.get("content") or "").strip()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif self._stack and self._stack[-1] == tag:
            self._stack.pop()

    def handle_data(self, data):
        text = data.strip()
        if self._in_title:
            if text and not self.title:
                self.title = text
            return
        if self._stack or not text:
            return
        self._chunks.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self._chunks)


def _robots_allowed(url: str, cache: Dict[str, object]) -> bool:
    site = urlsplit(url)
    base = f"{site.scheme}://{site.netloc}"
    parser = cache.get(base)
    if parser is None:
        parser = urllib.robotparser.RobotFileParser(f"{base}/robots.txt")
        try:
            parser.read()
        except (OSError, ValueError):
            # Unreachable robots.txt = no stated policy; proceed politely.
            parser.allow_all = True
        cache[base] = parser
    return parser.can_fetch(USER_AGENT, url)


def _normalize_link(href: str, base_url: str) -> Optional[str]:
    """Resolve an href to a fragment-free absolute http(s) URL, or None.
    Cross-site links are kept: the worker splits them into frontier edges
    (same site) vs citation counts toward watching a new site."""
    url = urljoin(base_url, href)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def fetch_page(
    url: str,
    max_bytes: int,
    conditional: Optional[dict] = None,
    robots_cache: Optional[Dict[str, object]] = None,
) -> FetchResult:
    """Fetch and extract one page. Raises on network/HTTP errors so the
    caller can record backoff; a 304 returns a result with no content."""
    if robots_cache is not None and not _robots_allowed(url, robots_cache):
        raise PermissionError(f"robots.txt disallows {url}")

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    headers.update(conditional or {})
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return FetchResult(status=304)
        raise

    enricher = enricher_for(url)
    with response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and enricher is None:
            raise ValueError(f"not html: {content_type}")
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"page exceeds {max_bytes} bytes")
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        charset = response.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")
        etag = response.headers.get("ETag", "")
        last_modified = response.headers.get("Last-Modified", "")

    if enricher is not None and "text/html" not in content_type:
        feed = enricher.parse_feed(url, content_type, html)
        if feed is None:
            raise ValueError(f"not html: {content_type}")
        title, text, hrefs = feed.title, feed.text, feed.links
        description = ""
    else:
        extractor = _Extractor()
        try:
            extractor.feed(html)
            extractor.close()
        except Exception:
            pass  # keep whatever was extracted before the parse hiccup
        title, text, description = extractor.title, extractor.text, extractor.description
        hrefs = list(extractor.links)
        if enricher is not None:
            extra = enricher.enrich_html(url, html)
            if extra is not None:
                hrefs.extend(extra.links)
                if extra.text:
                    text = f"{text}\n{extra.text}".strip()
                title = title or extra.title

    links, seen = [], set()
    for href in hrefs:
        normalized = _normalize_link(href, url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        # The TARGET's enricher can veto boilerplate paths on its own site.
        target = enricher_for(normalized)
        if target is not None and not target.link_allowed(normalized):
            continue
        links.append(normalized)

    summary = description or text[:140].replace("\n", " ").strip()
    return FetchResult(
        status=200,
        title=title or url,
        text=text,
        summary=summary,
        links=links,
        etag=etag,
        last_modified=last_modified,
    )
