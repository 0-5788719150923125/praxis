"""Registry namespace - the eventual home of a unified ``Registry`` type.

Praxis currently spreads behavior across ~40 ad-hoc ``*_REGISTRY`` dicts (see
the roadmap item on unifying them). This package is where that consolidation
will live. The first piece is discovery: counting the registries so the docs
can report a real number instead of a guess, rather than hand-maintaining it.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import List, Optional

_PKG_ROOT = Path(__file__).resolve().parent.parent  # the praxis/ package dir


@functools.lru_cache(maxsize=None)
def discover_registries(root: Optional[Path] = None) -> tuple:
    """Module-level names ending in ``_REGISTRY`` defined under the praxis
    package, sorted and de-duplicated.

    AST-based, so it neither imports the code nor matches names that only
    contain the substring rather than ending in ``_REGISTRY``."""
    root = Path(root) if root else _PKG_ROOT
    names = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:  # module scope only
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for t in targets:
                if isinstance(t, ast.Name) and t.id.endswith("_REGISTRY"):
                    names.add(t.id)
    return tuple(sorted(names))


def count_registries(root: Optional[Path] = None) -> int:
    """Number of distinct ``*_REGISTRY`` dicts in the codebase."""
    return len(discover_registries(root))
