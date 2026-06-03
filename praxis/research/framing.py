"""Loader for the paper's conditionally-rendered sections.

Each fragment is one YAML file in ``framings/`` (data only - id is the filename
stem); all logic lives here. A fragment declares a ``requires`` predicate over
the resolved experiment config and carries one self-contained chunk of LaTeX
(typically a ``\\subsection``), so the paper never claims geometry it didn't
build.

A fragment file::

    section: harmonic           # paper anchor it slots into
    order: 20                   # sort within the section
    requires:                   # AND across keys, OR (fnmatch) within a key
      head_type: [crystal, prismatic]
    body: >                     # folded: wrap lines freely, blank line = new para
      \\subsection{...}

      The prose wraps across as many
      lines as you like ...

Matching: a fragment is active when, for every key in ``requires``, the run's
value matches at least one accepted value (``fnmatch`` patterns allowed, e.g.
``calm*``). An empty/absent ``requires`` always renders. The export tool groups
active fragments by ``section`` and concatenates each group's bodies (in
``order``) into one ``\\paperFraming<Section>`` macro that ``main.tex`` drops in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List

import yaml

FRAMINGS_DIR = Path(__file__).parent / "framings"


@dataclass(frozen=True)
class Fragment:
    """One gated chunk of paper prose, loaded from a framings/*.yml file."""

    id: str
    section: str
    body: str
    requires: Dict[str, List[str]] = field(default_factory=dict)
    order: int = 100

    def matches(self, config: Dict) -> bool:
        """True if every required key in ``config`` hits an accepted value."""
        for key, accepted in self.requires.items():
            value = config.get(key)
            if value is None:
                return False
            if not any(fnmatch(str(value), str(pat)) for pat in accepted):
                return False
        return True


def _load_fragment(path: Path) -> Fragment:
    raw = yaml.safe_load(path.read_text()) or {}
    if "section" not in raw or "body" not in raw:
        raise ValueError(f"{path.name}: framing must define 'section' and 'body'")
    requires = raw.get("requires") or {}
    # Normalize scalar values to single-element lists so requires entries may be
    # written as either `key: value` or `key: [a, b]`.
    requires = {k: v if isinstance(v, list) else [v] for k, v in requires.items()}
    return Fragment(
        id=path.stem,
        section=raw["section"],
        body=_to_tex_paragraphs(raw["body"]),
        requires=requires,
        order=raw.get("order", 100),
    )


def _to_tex_paragraphs(body: str) -> str:
    """Bodies are written as YAML folded scalars (``>``): authors wrap lines
    freely and folding collapses each block to one line. TeX needs a blank line
    between paragraphs, so promote every newline run to a paragraph break."""
    return re.sub(r"\n+", "\n\n", body.strip())


def load_framings() -> Dict[str, Fragment]:
    """All fragments from ``framings/``, keyed by id (filename stem)."""
    return {
        p.stem: _load_fragment(p)
        for p in sorted(FRAMINGS_DIR.glob("*.yml"))
    }


FRAMING: Dict[str, Fragment] = load_framings()


def active_fragments(config: Dict) -> List[Fragment]:
    """Fragments whose ``requires`` are satisfied by ``config``, sorted for
    rendering (by section, then ``order``, then ``id`` for stability)."""
    active = [f for f in FRAMING.values() if f.matches(config)]
    return sorted(active, key=lambda f: (f.section, f.order, f.id))
