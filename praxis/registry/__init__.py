"""The Praxis registry: one place every pluggable choice is declared and found.

The registry holds named namespaces - ``attention``, ``memory``, ``optimizers``,
... - and each namespace maps a key to an implementation or profile, with what
each key means. A module declares its namespace where the entries live::

    from praxis import registry

    registry.declare(
        "attention",
        {
            "causal": CausalAttention,
            "arc_dropoff": Entry(
                partial(ArcAttention, dropoff="warp"),
                "Arc with the causal tip withheld at one recurrent pass.",
            ),
            "old_name": Alias("causal"),
        },
        title="Attention mechanisms",
        doc="Self-attention variants, selected with ``--attention-type``.",
    )

and everything else reads it by name, never by importing a constant::

    registry.lookup("attention", config.attention_type)(config)
    for key in registry.namespace("routers"): ...
    registry.describe("memory", "mag_energy")

A namespace is a ``dict`` of its listed entries. Documentation is optional: an
entry without a doc is described by the lead paragraph of its class or function
docstring. Asking for a namespace whose module has not been imported yet
imports it, so lookups do not depend on import order.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

_PKG_ROOT = Path(__file__).resolve().parent.parent  # the praxis/ package dir

# Every declared namespace, by name.
_NAMESPACES: Dict[str, "Namespace"] = {}

_MISSING = object()


class Entry:
    """A registry value with its documentation.

    ``listed=False`` keeps the name resolvable (configs and checkpoints may
    carry it) without offering it as a choice or giving it a docs entry."""

    __slots__ = ("value", "doc", "listed")

    def __init__(self, value: Any, doc: Optional[str] = None, *, listed: bool = True):
        self.value = value
        self.doc = doc
        self.listed = listed


class Alias:
    """An unlisted name that resolves to a listed entry."""

    __slots__ = ("target",)

    def __init__(self, target: str):
        self.target = target


class Namespace(dict):
    """One namespace of the registry: its listed entries, plus what each one
    is. Built by :func:`declare`; constructed directly only in tests."""

    def __init__(
        self,
        name: str,
        entries: Optional[Mapping[str, Any]] = None,
        *,
        title: Optional[str] = None,
        doc: Optional[str] = None,
        module: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.title = title
        self.doc = doc
        self.module = module
        self._docs: Dict[str, str] = {}
        self._unlisted: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}
        for key, value in (entries or {}).items():
            self.register(key, value)

    def register(self, key: str, value: Any, doc: Optional[str] = None) -> None:
        """Add ``key``. ``value`` may be an :class:`Entry` or :class:`Alias`."""
        if isinstance(value, Alias):
            self._aliases[key] = value.target
            return
        listed = True
        if isinstance(value, Entry):
            doc = value.doc if doc is None else doc
            listed = value.listed
            value = value.value
        if doc:
            self._docs[key] = " ".join(doc.split())
        if listed:
            self[key] = value
        else:
            self._unlisted[key] = value

    def canonical(self, key: str) -> str:
        """The name ``key`` resolves to: itself, or an alias's target."""
        return self._aliases.get(key, key)

    def aliases(self) -> Dict[str, str]:
        """``{alias: target}`` for every alias."""
        return dict(self._aliases)

    def unlisted(self) -> Dict[str, Any]:
        """``{name: value}`` for entries that resolve but are not listed."""
        return dict(self._unlisted)

    def describe(self, key: str) -> Optional[str]:
        """What ``key`` is, in one paragraph, or None when nothing says.

        The entry's own doc when it has one; otherwise the lead paragraph of
        the docstring of the class or function it resolves to."""
        key = self.canonical(key)
        if key in self._docs:
            return self._docs[key]
        if key not in self:
            return None
        return docstring_lead(self[key])

    def has_doc(self, key: str) -> bool:
        """Whether ``key`` carries a doc of its own (not a docstring fallback)."""
        return self.canonical(key) in self._docs

    def __missing__(self, key: str) -> Any:
        if key in self._aliases:
            return self[self._aliases[key]]
        return self._unlisted[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return (
            dict.__contains__(self, key)
            or key in self._aliases
            or key in self._unlisted
        )

    def __repr__(self) -> str:
        return f"Namespace(name={self.name!r}, entries={len(self)})"


def declare(
    name: str,
    entries: Optional[Mapping[str, Any]] = None,
    *,
    title: Optional[str] = None,
    doc: Optional[str] = None,
) -> Namespace:
    """Create namespace ``name``. ``title`` gives it a page under ``docs/``.

    A name can be declared once. Re-running the declaring module (a reload)
    replaces its own namespace; a second module claiming the name is an error."""
    module = sys._getframe(1).f_globals.get("__name__")
    existing = _NAMESPACES.get(name)
    if existing is not None and existing.module != module:
        raise ValueError(
            f"registry namespace {name!r} is declared by {existing.module}; "
            f"{module} cannot declare it again"
        )
    ns = Namespace(name, entries, title=title, doc=doc, module=module)
    _NAMESPACES[name] = ns
    return ns


def namespace(name: str) -> Namespace:
    """Namespace ``name``, importing the module that declares it if needed."""
    ns = _NAMESPACES.get(name)
    if ns is not None:
        return ns
    home = _home_modules().get(name)
    if home is not None:
        importlib.import_module(home)
        ns = _NAMESPACES.get(name)
        if ns is not None:
            return ns
    raise KeyError(
        f"no registry namespace {name!r}. Declared: {', '.join(sorted(_home_modules()))}"
    )


def lookup(name: str, key: str) -> Any:
    """The value ``key`` names in namespace ``name`` (aliases resolve)."""
    ns = namespace(name)
    try:
        return ns[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not in the {name!r} registry. Choices: {', '.join(ns)}"
        ) from None


def describe(name: str, key: str) -> Optional[str]:
    """What ``key`` is in namespace ``name`` (see :meth:`Namespace.describe`)."""
    return namespace(name).describe(key)


def register(
    name: str, key: str, value: Any = _MISSING, doc: Optional[str] = None
) -> Any:
    """Add ``key`` to namespace ``name``. Without ``value`` it is a decorator::

    @registry.register("sorting", "decay_bias")
    class DecayBiasSort(NoSort): ...
    """
    ns = namespace(name)
    if value is not _MISSING:
        ns.register(key, value, doc)
        return value

    def decorate(obj: Any) -> Any:
        ns.register(key, obj, doc)
        return obj

    return decorate


def namespaces() -> Tuple[Namespace, ...]:
    """Every namespace in the package, sorted by name. Imports every module
    that declares one."""
    for name in _home_modules():
        namespace(name)
    return tuple(ns for _, ns in sorted(_NAMESPACES.items()))


def count_namespaces() -> int:
    """Number of namespaces declared in the package source."""
    return len(_home_modules())


@functools.lru_cache(maxsize=None)
def _home_modules(root: Optional[Path] = None) -> Dict[str, str]:
    """``{namespace: module}`` for every module-level ``registry.declare("x",
    ...)`` in the package source. AST-based, so it imports nothing."""
    root = Path(root) if root else _PKG_ROOT
    found: Dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root.parent).with_suffix("")
        module = ".".join(p for p in rel.parts if p != "__init__")
        for node in tree.body:  # module scope only
            call = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
            if (
                isinstance(call, ast.Call)
                and _is_declare(call.func)
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            ):
                found.setdefault(call.args[0].value, module)
    return found


def _is_declare(func: ast.AST) -> bool:
    return (isinstance(func, ast.Name) and func.id == "declare") or (
        isinstance(func, ast.Attribute)
        and func.attr == "declare"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def unwrap(value: Any) -> Tuple[Any, tuple, dict]:
    """Follow a ``functools.partial`` chain to its target.

    Returns ``(target, args, keywords)`` with the bound arguments of every
    layer, outermost last so later bindings win as they do on a call."""
    args: tuple = ()
    keywords: dict = {}
    chain = []
    while isinstance(value, functools.partial):
        chain.append(value)
        value = value.func
    for layer in reversed(chain):
        args += layer.args
        keywords.update(layer.keywords)
    return value, args, keywords


_SECTION_HEADERS = (
    "Args:",
    "Arguments:",
    "Returns:",
    "Raises:",
    "Yields:",
    "Example:",
    "Examples:",
    "Note:",
    "Notes:",
    "Attributes:",
)


def docstring_body(obj: Any) -> str:
    """The docstring of ``obj`` (unwrapped through partials) up to its first
    section header, falling back to its module's docstring."""
    target, _, _ = unwrap(obj)
    # A (class, kwargs) pair, the shape transformers' ACT2FN stores presets in.
    if isinstance(target, tuple) and target and inspect.isclass(target[0]):
        target = target[0]
    if not (inspect.isclass(target) or inspect.isroutine(target)):
        return ""
    for source in (inspect.getdoc(target), inspect.getdoc(inspect.getmodule(target))):
        if not source:
            continue
        lines = source.splitlines()
        cut = next(
            (i for i, line in enumerate(lines) if line.strip() in _SECTION_HEADERS),
            len(lines),
        )
        body = "\n".join(lines[:cut]).strip()
        if body:
            return body
    return ""


def docstring_lead(obj: Any) -> Optional[str]:
    """The first paragraph of :func:`docstring_body`, on one line."""
    body = docstring_body(obj)
    if not body:
        return None
    return " ".join(body.split("\n\n", 1)[0].split())


__all__ = [
    "Alias",
    "Entry",
    "Namespace",
    "count_namespaces",
    "declare",
    "describe",
    "docstring_body",
    "docstring_lead",
    "lookup",
    "namespace",
    "namespaces",
    "register",
    "unwrap",
]
