"""Normalized payloads on the KB bus.

Every source emits ``KBItem``s of a single shape; search returns ``KBHit``s
(an item plus its ranking score and a highlighted snippet).
"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class KBItem:
    id: str  # stable, source-namespaced, e.g. "doc:attention", "run:008b0cb75"
    type: str  # machine kind, drives behavior: "doc" | "run" | "note" | "link" | "card"
    label: str  # human display chip, source-specific: "Roadmap", "Run", "Research"
    title: str
    body: str  # searchable text
    uri: str  # where Read opens it: tab route, file path, or URL
    source: str = ""  # producing source's registry name; stamped by the bus
    origin: str = ""  # document/event that yielded it: "next/roadmap.md", a site
    summary: str = ""  # one-line supporting detail; bus derives one if empty
    meta: dict = field(default_factory=dict)
    updated: float = 0.0  # unix mtime; 0 = no natural timestamp (cards, agents)


def with_provenance(item: KBItem, source: str) -> KBItem:
    """Stamp bus-level defaults: the producing source's name, and a derived
    summary (first legible body line) when the source didn't provide one."""
    summary = item.summary or _derive_summary(item.body)
    if item.source == source and item.summary == summary:
        return item
    return replace(item, source=source, summary=summary)


def _derive_summary(body: str, width: int = 140) -> str:
    for line in (body or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:width]
    return ""


@dataclass(frozen=True)
class KBHit:
    item: KBItem
    score: float  # lower is better (FTS5 bm25 rank)
    snippet: str  # query-highlighted excerpt
