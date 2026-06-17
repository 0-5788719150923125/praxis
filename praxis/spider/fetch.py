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
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

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
    image: str = ""  # preview image (og:image), if the page carries one
    richness: float = 0.0  # content-conformance score in [0, 1]


def content_richness(title: str, text: str, summary: str, image: str) -> float:
    """Host-agnostic conformance score in [0, 1]: how well a fetched page meets
    the marks of substantive content - a preview image, a body of prose, a real
    summary, a title. Videos rise because they carry thumbnails and
    descriptions, not because the score knows what a video is."""
    body = (text or "").strip()
    marks = (
        (0.40, 1.0 if image else 0.0),
        (0.30, min(len(body) / 1200.0, 1.0)),
        (0.20, 1.0 if len((summary or "").strip()) >= 40 else 0.0),
        (0.10, 1.0 if (title or "").strip() else 0.0),
    )
    return round(sum(w * m for w, m in marks), 4)


class _Extractor(HTMLParser):
    """Title, readable text, meta description, and hrefs from one page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.description = ""
        self.image = ""
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
        elif (
            tag == "meta"
            and not self.image
            and (
                attrs.get("property") == "og:image"
                or attrs.get("name") == "twitter:image"
            )
        ):
            self.image = (attrs.get("content") or "").strip()

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


# Query keys that only track a click; they never change which page is served,
# so dropping them collapses URL variants that would otherwise be stored and
# indexed as separate (duplicate) pages.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_social", "utm_brand",
        "fbclid", "gclid", "dclid", "msclkid", "yclid", "mc_eid", "mc_cid",
        "igshid", "igsh", "ref", "ref_src", "ref_url", "spm",
        "_hsenc", "_hsmi", "oly_enc_id", "oly_anon_id",
    }
)  # fmt: skip


def canonical_url(url: str) -> str:
    """A stable identity for a page: lowercased scheme/host, no default port, no
    trailing slash, no fragment, and tracking-only query params dropped. This is
    the dedup key - it collapses the trailing-slash / utm variants that would
    otherwise store and index the same page several times. Meaningful params
    (e.g. ``?v=`` on a watch URL) and ``www`` are preserved (``www`` is part of
    the site key the crawler matches on)."""
    parts = urlsplit(url)
    netloc = parts.netloc.lower()
    # Drop the port when it's the scheme's default.
    if (parts.scheme, parts.port) in (("http", 80), ("https", 443)):
        netloc = (parts.hostname or "").lower()
    query = "&".join(
        f"{k}={v}" if v else k
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    )
    # Drop trailing slashes, including a bare "/" root, so the slash variants
    # of a page collapse to one key.
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def _normalize_link(href: str, base_url: str) -> Optional[str]:
    """Resolve an href to a canonical absolute http(s) URL, or None.
    Cross-site links are kept: the worker splits them into frontier edges
    (same site) vs citation counts toward watching a new site."""
    url = urljoin(base_url, href)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    return canonical_url(url)


def _absolute_image(image: str, base_url: str) -> str:
    """Resolve an og:image reference to an absolute http(s) URL the dashboard
    can actually load. Relative (``/img/x.jpg``) and protocol-relative
    (``//cdn/x.jpg``) references are the common cause of broken thumbnails -
    the browser would otherwise resolve them against the dashboard's own
    origin. Anything that isn't http(s) is dropped."""
    if not image:
        return ""
    url = urljoin(base_url, image.strip())
    if urlsplit(url).scheme not in ("http", "https"):
        return ""
    return url


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

    image = ""
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
        title, text, description = (
            extractor.title,
            extractor.text,
            extractor.description,
        )
        image = _absolute_image(extractor.image, url)
        hrefs = list(extractor.links)
        if enricher is not None:
            extra = enricher.enrich_html(url, html)
            if extra is not None:
                if not extra.links and not extra.text:
                    # An enriched site that yields nothing minable is a
                    # consent/blocked shell (common on remote IPs), not
                    # content. Treat as an error so it backs off and retries
                    # instead of becoming an empty KB husk.
                    raise ValueError("blocked or empty shell page")
                hrefs.extend(extra.links)
                if extra.text:
                    # Mined text IS the content (e.g. a video description);
                    # the JS shell around it is footer boilerplate.
                    text = extra.text
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
        image=image,
        richness=content_richness(title, text, summary, image),
    )
