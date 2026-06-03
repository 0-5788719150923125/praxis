"""KB sources: typed producers of ``KBItem``s, registered on a bus.

A source knows how to walk one corner of the corpus (the wiki, run history,
research notes, external links) and yield normalized items. New corpora plug
in by subclassing ``KBSource`` and adding to ``KB_SOURCE_REGISTRY``.
"""

import json
import re
import sqlite3
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from praxis.kb.item import KBItem

# Repo root: praxis/kb/sources.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

# A link embedded in markdown: [text](http://...). Used to seed LinksSource
# until a real spider populates external documents.
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


class KBSource(ABC):
    name: str

    @abstractmethod
    def iter_items(self) -> Iterable[KBItem]: ...


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
    return fallback


class DocsSource(KBSource):
    """The auto-generated wiki under ``docs/`` - one item per markdown file."""

    name = "docs"

    def iter_items(self) -> Iterable[KBItem]:
        docs_dir = REPO_ROOT / "docs"
        for path in sorted(docs_dir.glob("*.md")):
            text = _read(path)
            if not text:
                continue
            slug = path.stem
            yield KBItem(
                id=f"doc:{slug}",
                type="doc",
                label="Wiki",
                title=_first_heading(text, slug),
                body=text,
                uri=f"docs/{path.name}",
            )


class NotesSource(KBSource):
    """Research notes and the roadmap under ``next/``.

    Long files are split on ``##`` / top-level list items so hits land on a
    granular section rather than a whole document.
    """

    name = "notes"

    def iter_items(self) -> Iterable[KBItem]:
        notes_dir = REPO_ROOT / "next"
        for path in sorted(notes_dir.glob("*.md")):
            text = _read(path)
            if not text:
                continue
            doc_title = _first_heading(text, path.stem)
            # Label by source document (e.g. "Roadmap", "Oscillatory Axes") so
            # notes don't all collapse to one generic chip.
            label = path.stem.replace("_", " ").title()
            for i, (title, body) in enumerate(_split_sections(text)):
                yield KBItem(
                    id=f"note:{path.stem}#{i}",
                    type="note",
                    label=label,
                    title=title or doc_title,
                    body=body,
                    uri=f"next/{path.name}",
                    meta={"document": doc_title},
                )


class RunsSource(KBSource):
    """Experiments under ``build/runs/<hash>/`` keyed by run hash."""

    name = "runs"

    def iter_items(self) -> Iterable[KBItem]:
        runs_dir = REPO_ROOT / "build" / "runs"
        if not runs_dir.is_dir():
            return
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            hash_id = run_dir.name
            config = _read(run_dir / "config.json")
            if not config:
                continue
            label = _run_label(config) or hash_id
            yield KBItem(
                id=f"run:{hash_id}",
                type="run",
                label="Run",
                title=f"run {hash_id} ({label})" if label != hash_id else f"run {hash_id}",
                body=config,
                uri=f"build/runs/{hash_id}/config.json",
                meta={"hash": hash_id},
            )


class LinksSource(KBSource):
    """External documents, seeded from links embedded in docs and notes.

    The spider that autonomously ingests these is deferred; this source makes
    the links searchable now and is the natural home for crawled content later.
    """

    name = "links"

    def iter_items(self) -> Iterable[KBItem]:
        seen = set()
        for sub in ("docs", "next"):
            for path in sorted((REPO_ROOT / sub).glob("*.md")):
                text = _read(path)
                for label, url in _MD_LINK.findall(text):
                    if url in seen:
                        continue
                    seen.add(url)
                    yield KBItem(
                        id=f"link:{url}",
                        type="link",
                        label="Link",
                        title=label.strip(),
                        body=f"{label}\n{url}",
                        uri=url,
                        meta={"found_in": f"{sub}/{path.name}"},
                    )


class CardsSource(KBSource):
    """Dashboard chart cards (Research / Dynamics / Identity tabs).

    Enumerates the chart registries so a card is findable by name in Read mode;
    opening one navigates to its tab and slides the deck to it (see the web
    OPEN_KB_ITEM handler). The uri encodes the target tab; meta carries the
    title used to locate the rendered card.
    """

    name = "cards"

    # Identity tab "sheets" are built in JS, not a registry; mirror them here.
    _IDENTITY_SHEETS = [
        ("Identity & Commands", "Run hashes, reproduce commands, parameter counts"),
        ("Architecture", "Instantiated model module tree"),
        ("Arguments", "Resolved run configuration"),
    ]

    def iter_items(self) -> Iterable[KBItem]:
        from praxis.metrics import (
            COMPOSITE_METRIC_REGISTRY,
            DYNAMICS_CHART_REGISTRY,
            TRAINING_METRIC_REGISTRY,
        )

        # A Research card only renders once its metric has data; only surface
        # cards we can actually land on. None = no run found yet, so include all
        # (discovery on a fresh checkout). Reindex to refresh as a run logs more.
        data_cols = _metric_columns_with_data()

        def has_data(key, key_pattern=None):
            if data_cols is None:
                return True
            if key_pattern is not None:
                return any(key_pattern.search(c) for c in data_cols)
            return key in data_cols

        for key, spec in TRAINING_METRIC_REGISTRY.items():
            chart = spec.get("chart") or {}
            title = chart.get("title")
            if title and has_data(key):
                yield self._card("research", "Research", key, title,
                                 spec.get("description") or chart.get("y_label", ""))

        for entry in COMPOSITE_METRIC_REGISTRY:
            pattern = entry.get("key_pattern")
            compiled = re.compile(pattern) if pattern else None
            present = has_data(entry.get("key", ""), compiled) if compiled else has_data(entry.get("key", ""))
            if entry.get("title") and present:
                yield self._card("research", "Research", entry.get("key", ""),
                                 entry["title"], entry.get("y_label", ""))

        # Dynamics + Identity are structural (gradient families / fixed sheets /
        # module-emitted scalars); surfaced unconditionally - the deck self-skips
        # any whose metric has no data yet, and navigation degrades gracefully.
        seen_dynamics = set()
        for entry in DYNAMICS_CHART_REGISTRY:
            key = entry.get("key", "")
            if entry.get("title") and key not in seen_dynamics:
                seen_dynamics.add(key)
                yield self._card("dynamics", "Dynamics", key,
                                 entry["title"], entry.get("subtitle", ""))

        for key, title, desc in _module_chart_metrics():
            if key not in seen_dynamics:
                seen_dynamics.add(key)
                yield self._card("dynamics", "Dynamics", key, title, desc)

        for title, subtitle in self._IDENTITY_SHEETS:
            yield self._card("spec", "Identity", title, title, subtitle)

    @staticmethod
    def _card(tab: str, label: str, key: str, title: str, body: str) -> KBItem:
        return KBItem(
            id=f"card:{tab}:{key}",
            type="card",
            label=label,
            title=title,
            body=body or title,
            uri=f"card:{tab}",
            meta={"tab": tab, "card_title": title},
        )


class AgentsSource(KBSource):
    """Peer agents = this repo's git remotes (the Stage tab's fleet).

    Indexes each remote's identity; live online/offline status is resolved on
    the tab, not here. URLs are masked so a remote carrying an ngrok secret
    never lands in the index.
    """

    name = "agents"

    def iter_items(self) -> Iterable[KBItem]:
        try:
            from praxis.utils import mask_git_url
        except ImportError:
            mask_git_url = lambda u: u  # noqa: E731
        try:
            out = subprocess.run(
                ["git", "remote", "-v"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        seen = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0] in seen:
                continue
            name, url = parts[0], mask_git_url(parts[1])
            seen.add(name)
            yield KBItem(
                id=f"agent:{name}",
                type="agent",
                label="Agent",
                title=name,
                body=f"{name}\n{url}",
                uri="tab:agents",
                meta={"name": name, "url": url},
            )


def _module_chart_metrics() -> List[tuple]:
    """(key, title, description) for every module-emitted chart/snapshot metric.

    Modules (encoders, heads, ...) declare a ``metric_descriptions`` class attr
    that renders on the Dynamics tab via the scalar/snapshot manifest - they're
    not in the central registries. Discover them statically by walking every
    ``*_REGISTRY`` praxis exposes and reading each registered class's attribute
    (no model instantiation). Deduped by metric key.
    """
    import functools
    import inspect

    try:
        import praxis
    except Exception:
        return []

    def resolve(value):
        if inspect.isclass(value):
            return value
        if isinstance(value, functools.partial):
            return resolve(value.func)
        return None

    out: dict = {}
    seen_titles = set()
    for name in dir(praxis):
        if not name.endswith("_REGISTRY"):
            continue
        registry = getattr(praxis, name, None)
        if not isinstance(registry, dict):
            continue
        for value in registry.values():
            cls = resolve(value)
            descriptions = getattr(cls, "metric_descriptions", None) if cls else None
            if not isinstance(descriptions, dict):
                continue
            for key, entry in descriptions.items():
                if key in out or not isinstance(entry, dict):
                    continue
                hint = entry.get("chart") or entry.get("snapshot")
                title = hint.get("title") if isinstance(hint, dict) else None
                # Series-group companions share a title and render as one card
                # under the first (lead) key; dedup so KB shows it once.
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                out[key] = (title, entry.get("description", ""))
    return [(k, t, d) for k, (t, d) in out.items()]


def _metric_columns_with_data() -> Optional[set]:
    """Scalar metric columns that hold at least one value in the newest run's
    metrics.db. Returns None when no run/db exists (caller includes all)."""
    runs_dir = REPO_ROOT / "build" / "runs"
    if not runs_dir.is_dir():
        return None
    dbs = [
        run / "metrics.db" for run in runs_dir.iterdir() if (run / "metrics.db").exists()
    ]
    if not dbs:
        return None
    db_path = max(dbs, key=lambda p: p.stat().st_mtime)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(metrics)")]
        candidate = [c for c in cols if c not in ("step", "ts", "id")]
        with_data = set()
        if candidate:
            select = ", ".join(f'COUNT("{c}")' for c in candidate)
            row = conn.execute(f"SELECT {select} FROM metrics").fetchone()
            with_data = {c for c, n in zip(candidate, row) if n}
        conn.close()
        return with_data
    except sqlite3.Error:
        return None


def _split_sections(text: str) -> List[tuple]:
    """Split markdown into (heading, body) chunks on ## headers and list items."""
    sections, title, buf = [], "", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        is_break = line.startswith("##") or re.match(r"^\s*[-*]\s+\*\*", line)
        if is_break and buf:
            flush()
            title, buf = _first_heading(line, line.strip()), [line]
        else:
            buf.append(line)
    flush()
    return sections or [("", text)]


def _run_label(config_json: str) -> str:
    """Pull a human label (model_name / experiment) from a run config blob."""
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError:
        return ""
    for key in ("experiment", "model_name", "name"):
        val = cfg.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


KB_SOURCE_REGISTRY = {
    "docs": DocsSource,
    "notes": NotesSource,
    "runs": RunsSource,
    "links": LinksSource,
    "cards": CardsSource,
    "agents": AgentsSource,
}
