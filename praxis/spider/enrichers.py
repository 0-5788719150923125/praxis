"""Per-site content enrichers for the spider.

Some sites are JS-rendered: <a href> extraction yields nothing, but the data
the spider wants (links, text) sits in inline JSON or an alternate machine
feed. An enricher recognizes its site, mines the raw document for extra links
and text, and may handle a non-HTML feed format outright. The generic walk
stays unchanged; sites get smarter by adding an entry to ENRICHER_REGISTRY.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlsplit


@dataclass
class Enriched:
    links: List[str] = field(default_factory=list)
    text: str = ""
    title: str = ""


class Enricher:
    """Base: site recognition + hooks. Either hook may return None (no-op)."""

    def matches(self, url: str) -> bool:
        raise NotImplementedError

    def enrich_html(self, url: str, html: str) -> Optional[Enriched]:
        return None

    def parse_feed(self, url: str, content_type: str, body: str) -> Optional[Enriched]:
        """Handle a non-HTML body (e.g. an Atom feed) the fetcher would
        otherwise reject."""
        return None


class YouTubeEnricher(Enricher):
    """Channel/watch pages are JS-rendered; mine their inline JSON for video
    links, cited channels, and descriptions - the path that lets discovery
    spill from one channel into the ones it cites. (The Atom feeds would be
    cleaner, but robots.txt disallows them.)"""

    _VIDEO_ID = re.compile(r'"videoId":"([0-9A-Za-z_-]{11})"')
    _HANDLE = re.compile(r'"(?:url|canonicalBaseUrl)":"(/@[A-Za-z0-9._\-]+)"')
    _DESCRIPTION = re.compile(r'"shortDescription":"((?:[^"\\]|\\.)*)"')

    _MAX_VIDEOS = 30
    _MAX_CHANNELS = 5

    def matches(self, url: str) -> bool:
        return urlsplit(url).netloc.lower().endswith("youtube.com")

    def enrich_html(self, url: str, html: str) -> Enriched:
        # Stay on the fetched host so links land in this site's frontier.
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        links = []
        for vid in dict.fromkeys(self._VIDEO_ID.findall(html)[: self._MAX_VIDEOS]):
            links.append(f"{base}/watch?v={vid}")
        for handle in dict.fromkeys(self._HANDLE.findall(html)[: self._MAX_CHANNELS]):
            links.append(f"{base}{handle}")
        text = ""
        match = self._DESCRIPTION.search(html)
        if match:
            try:
                text = json.loads(f'"{match.group(1)}"')
            except ValueError:
                pass
        return Enriched(links=links, text=text)


ENRICHER_REGISTRY: Dict[str, Enricher] = {
    "youtube": YouTubeEnricher(),
}


def enricher_for(url: str) -> Optional[Enricher]:
    for enricher in ENRICHER_REGISTRY.values():
        if enricher.matches(url):
            return enricher
    return None
