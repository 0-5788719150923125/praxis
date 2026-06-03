"""Inline edits: single-value substitutions the paper splices into its prose.

Where a :mod:`~praxis.research.framing` fragment injects a whole section, an
inline edit replaces one token - a count, a rate, a metric - with a value
computed from the repository. Each edit is one YAML file in ``inlines/`` (data
only); the value comes from a named *provider* registered here in code.

An inline file::

    macro: paperRunsLogged      # the \\paperRunsLogged command in main.tex
    provider: run_count         # a @provider function below
    args: {}                    # optional kwargs passed to the provider
    format: "{value}"           # how to render a non-None value
    fallback: "Thousands of"    # macro body when the provider yields None

Resolution: the provider is called with ``args``; if it returns ``None`` the
edit is *absent* and the paper's ``\\providecommand`` fallback stands, so a
metric that does not exist for a given run simply leaves the prose unchanged.
Otherwise the value is rendered through ``format`` and emitted as a
``\\newcommand`` that overrides the fallback.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

INLINES_DIR = Path(__file__).parent / "inlines"
REPO_ROOT = Path(__file__).parents[2]

PROVIDERS: Dict[str, Callable[..., Optional[Any]]] = {}


def provider(name: str):
    """Register a value provider. A provider returns a value, or ``None`` to
    signal "no value for this run" (the paper keeps its fallback)."""

    def wrap(fn):
        PROVIDERS[name] = fn
        return fn

    return wrap


@provider("run_count")
def _run_count() -> Optional[int]:
    """Number of recorded runs (run dirs carrying a config.json). Always
    defined while any run exists; ``None`` only on an empty checkout."""
    n = len(glob.glob(str(REPO_ROOT / "build" / "runs" / "*" / "config.json")))
    return n or None


@dataclass(frozen=True)
class InlineEdit:
    """One macro substitution, loaded from an inlines/*.yml file."""

    id: str
    macro: str
    provider: str
    fallback: str
    fmt: str = "{value}"
    args: Dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Optional[str]:
        """Rendered macro body, or ``None`` if the provider yields no value."""
        fn = PROVIDERS.get(self.provider)
        if fn is None:
            raise KeyError(f"{self.id}: unknown provider '{self.provider}'")
        value = fn(**self.args)
        if value is None:
            return None
        return self.fmt.format(value=value)


def _load_inline(path: Path) -> InlineEdit:
    raw = yaml.safe_load(path.read_text()) or {}
    for key in ("macro", "provider", "fallback"):
        if key not in raw:
            raise ValueError(f"{path.name}: inline must define '{key}'")
    return InlineEdit(
        id=path.stem,
        macro=raw["macro"],
        provider=raw["provider"],
        fallback=raw["fallback"],
        fmt=raw.get("format", "{value}"),
        args=raw.get("args") or {},
    )


def load_inlines() -> Dict[str, InlineEdit]:
    """All inline edits from ``inlines/``, keyed by id (filename stem)."""
    return {p.stem: _load_inline(p) for p in sorted(INLINES_DIR.glob("*.yml"))}


INLINES: Dict[str, InlineEdit] = load_inlines()
