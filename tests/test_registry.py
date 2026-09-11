"""Tests for praxis/registry: namespaces, lookup and descriptions, plus the
package-wide lints that keep every pluggable choice in the registry and keep
background threads off the process-global stdout."""

import ast
import re
from functools import partial
from pathlib import Path

import pytest

import praxis
from praxis import registry
from praxis.registry import Alias, Entry, Namespace

PKG = Path(praxis.__file__).resolve().parent

# Chart declarations for the web dashboard: ordered lists of specs, each with
# its own description, rather than named choices.
_CHART_LISTS = {
    "COMPOSITE_METRIC_REGISTRY",
    "DYNAMICS_CHART_REGISTRY",
    "TRAINING_METRIC_REGISTRY",
    "X_AXIS_REGISTRY",
}


class _Documented:
    """A documented class.

    Second paragraph."""


class _Bare:
    pass


def _toy():
    return Namespace(
        "toy",
        {
            "documented": _Documented,
            "preset": Entry(partial(_Documented, x=1), "The documented class, x=1."),
            "bare": _Bare,
            "profile": Entry(dict(a=1), "A profile."),
            "hidden": Entry(_Bare, "Not offered.", listed=False),
            "old_name": Alias("documented"),
        },
    )


def test_namespace_is_a_dict_of_its_listed_entries():
    ns = _toy()
    assert isinstance(ns, dict)
    assert list(ns) == ["documented", "preset", "bare", "profile"]
    assert ns["profile"] == {"a": 1}
    assert ns["preset"].keywords == {"x": 1}


def test_unlisted_names_resolve_without_being_listed():
    ns = _toy()
    assert "hidden" in ns and "old_name" in ns
    assert ns["old_name"] is _Documented
    assert ns.get("hidden") is _Bare
    assert ns.get("missing") is None
    assert ns.canonical("old_name") == "documented"
    assert ns.aliases() == {"old_name": "documented"}
    assert ns.unlisted() == {"hidden": _Bare}
    with pytest.raises(KeyError):
        ns["missing"]


def test_describe_prefers_the_entry_doc_and_falls_back_to_the_docstring():
    ns = _toy()
    assert ns.describe("preset") == "The documented class, x=1."
    assert ns.describe("documented") == "A documented class."
    assert ns.describe("old_name") == "A documented class."
    assert ns.describe("hidden") == "Not offered."
    assert ns.has_doc("profile") and not ns.has_doc("documented")
    assert ns.describe("missing") is None


def test_lookup_names_the_choices_when_a_key_is_missing():
    assert registry.lookup("memory", "none") is None
    with pytest.raises(KeyError, match="not in the 'memory' registry"):
        registry.lookup("memory", "no_such_profile")
    with pytest.raises(KeyError, match="no registry namespace"):
        registry.namespace("no_such_namespace")


def test_a_second_module_cannot_claim_a_declared_name():
    registry.namespace("memory")
    with pytest.raises(ValueError, match="is declared by praxis.memory"):
        registry.declare("memory", {})


def test_every_declared_namespace_loads_and_is_found():
    """Each ``registry.declare`` in the source is found without importing its
    module first, and declares the namespace it names."""
    for ns in registry.namespaces():
        assert registry.namespace(ns.name) is ns
        assert ns.module and ns.module.startswith("praxis"), ns.name


def test_no_module_keeps_its_own_registry_constant():
    """Pluggable choices are declared into ``praxis.registry`` and looked up
    by name. A module-level ``*_REGISTRY`` or ``*_PROFILES`` dict is a second,
    private registry that nothing else can enumerate or document."""
    found = []
    for path in PKG.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            for t in targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id.endswith(("_REGISTRY", "_PROFILES"))
                    and t.id not in _CHART_LISTS
                ):
                    found.append(f"{path.relative_to(PKG.parent)}: {t.id}")
    assert not found, f"declare these with registry.declare instead: {found}"


# ------------------------------------------------------------------------------
# stdout safety
# ------------------------------------------------------------------------------
# ``contextlib.redirect_stdout`` mutates a PROCESS-GLOBAL. Used from the Flask API
# thread or a build thread, it silently redirects every other thread's output for
# the width of the block, and any thread that reads ``sys.stdout`` before the block
# ends and writes to it after gets ``ValueError: I/O operation on closed file``.
# That killed abstractinator-m at its first step, when the snapshot publisher
# printed the model repr under a redirect while the compute profiler flushed
# stdout on the training thread.


def test_no_background_thread_redirects_stdout():
    """redirect_stdout must not reappear off the main thread.

    paper.py is the known remaining offender - ``_build`` runs in a daemon
    thread and holds the global for an entire LaTeX build - so it is listed
    explicitly rather than silently tolerated.
    """
    known = {"praxis/callbacks/lightning/paper.py"}
    # CALLS only - the fix in spec_data.py names the hazard in a comment, and
    # matching prose would make this test un-passable by documenting itself.
    call = re.compile(r"^(?!\s*#).*\bredirect_std(out|err)\s*\(", re.M)
    offenders = {
        str(path.relative_to(PKG.parent))
        for path in PKG.rglob("*.py")
        if call.search(path.read_text())
    }
    assert offenders <= known, (
        f"new global-stdout redirect(s): {sorted(offenders - known)}. "
        "Render to a string instead; see praxis/web/spec_data.py."
    )
