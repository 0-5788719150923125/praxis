"""Praxis spider: a slow, bounded, eventually-consistent web crawler.

The spider grounds itself in a small watchlist of sites (seeded from links
cited in docs/ and next/), then walks each site top-to-bottom in pieces - one
HTTP request per tick, spread over days. Crawled pages land in
``build/spider.db`` (the store of record) and surface in the KB through the
"pages" source on reindex.

Profiles live in ``SPIDER_REGISTRY``; ``--spider`` selects and overrides them
with KEY=VALUE entries, e.g. ``--spider profile=gentle tick_seconds=600``.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Union

# Fixed, model-agnostic pacing constants. "gentle" is the default profile;
# "brisk" is for development, where waiting 5 minutes per fetch is unusable.
SPIDER_REGISTRY: Dict[str, dict] = {
    "gentle": dict(
        max_sites=16,
        max_pages_per_site=50,
        max_kb_pages=200,
        tick_seconds=300,
        domain_seconds=1800,
        max_page_bytes=2 * 1024 * 1024,
        revisit_days=7.0,
    ),
    "ghost": dict(
        max_sites=23,
        max_pages_per_site=23,
        max_kb_pages=230,
        tick_seconds=15,
        domain_seconds=60,
        max_page_bytes=2 * 1024 * 1024,
        revisit_days=0.01,  # 14.4 minutes; for testing re-fetching of stale pages
    ),
}


@dataclass(frozen=True)
class SpiderSettings:
    profile: str = "gentle"
    max_sites: int = 16  # watchlist cap; adding past it evicts the stalest
    max_pages_per_site: int = 50  # frontier + stored pages cap per site
    max_kb_pages: int = 200  # cap on "page" items surfaced into the KB
    tick_seconds: int = 300  # one HTTP request per tick, globally
    domain_seconds: int = 1800  # minimum spacing between hits to one host
    max_page_bytes: int = 2 * 1024 * 1024  # response size cap, text/html only
    revisit_days: float = 7.0  # re-fetch the stalest page when the frontier is
    # dry; fractional = sub-day watching (0.05 ≈ every 72 minutes)


def spider_settings(
    entries: Optional[Union[List[str], dict]],
) -> Optional[SpiderSettings]:
    """Resolve ``--spider`` KEY=VALUE entries to settings; None = disabled.

    ``profile`` picks a base from ``SPIDER_REGISTRY`` (default "gentle");
    remaining keys override individual fields. Accepts a dict too, for
    experiment yml configs.
    """
    if entries is None:
        return None
    if isinstance(entries, dict):
        overrides = dict(entries)
    else:
        overrides = {}
        for entry in entries:
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError(f"--spider entries must be KEY=VALUE, got {entry!r}")
            overrides[key.strip()] = value.strip()

    profile = str(overrides.pop("profile", "gentle"))
    if profile not in SPIDER_REGISTRY:
        raise ValueError(
            f"Unknown spider profile {profile!r}. Available: {sorted(SPIDER_REGISTRY)}"
        )
    spec = dict(SPIDER_REGISTRY[profile], profile=profile)

    valid = {f.name for f in fields(SpiderSettings)}
    for key, value in overrides.items():
        if key not in valid:
            raise ValueError(
                f"Unknown spider setting {key!r}. Available: {sorted(valid)}"
            )
        spec[key] = _cast(key, value)
    return SpiderSettings(**spec)


def _cast(key: str, value):
    if key == "profile":
        return str(value)
    if key == "revisit_days":
        return float(value)
    return int(value)


__all__ = ["SPIDER_REGISTRY", "SpiderSettings", "spider_settings"]
