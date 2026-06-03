"""Normalized payloads on the KB bus.

Every source emits ``KBItem``s of a single shape; search returns ``KBHit``s
(an item plus its ranking score and a highlighted snippet).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KBItem:
    id: str  # stable, source-namespaced, e.g. "doc:attention", "run:008b0cb75"
    type: str  # machine kind, drives behavior: "doc" | "run" | "note" | "link" | "card"
    label: str  # human display chip, source-specific: "Roadmap", "Run", "Research"
    title: str
    body: str  # searchable text
    uri: str  # where Read opens it: tab route, file path, or URL
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KBHit:
    item: KBItem
    score: float  # lower is better (FTS5 bm25 rank)
    snippet: str  # query-highlighted excerpt
