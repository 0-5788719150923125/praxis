"""Auto-docs generator for Praxis.

Writes one markdown page under ``docs/`` for every registry namespace (see
``praxis.registry``) that carries a title, from the docs declared on its entries (falling back to
class docstrings).
Also documents the third-party ``integrations/`` from their spec.yaml files,
and patches the project README between two AUTODOC markers.

Runs at every launch (see ``main.py`` ``setup_environment``). Writes are
idempotent: a file is only touched when its content actually changes, so
this never produces spurious git diffs.

To skip generation: ``./launch --no-docs``.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from praxis.registry import (
    Namespace,
    count_namespaces,
    docstring_body,
    namespaces,
    unwrap,
)


def registry_pages() -> List[Namespace]:
    """Every registry namespace with a docs page (a ``title``), by name."""
    return [ns for ns in namespaces() if ns.title]


def page_slug(ns: Namespace) -> str:
    """The docs filename stem for a namespace: its name, dashed."""
    return ns.name.replace("_", "-")


# Hand-curated one-liners for packages that don't have a registry.
# Listed in docs/index.md as "core infrastructure" pointers.
INFRASTRUCTURE_PACKAGES: List[Tuple[str, str]] = [
    (
        "callbacks",
        "Lightning callbacks (terminal interface, dynamics logging, periodic eval, ...).",
    ),
    (
        "cli",
        "Argparse-based CLI with experiment YAMLs, environment YAMLs, and PRAXIS_* env var overrides.",
    ),
    (
        "configuration",
        "``PraxisConfig`` - the central model config object passed everywhere.",
    ),
    ("containers", "Small typed containers (LossContainer, OutputContainer)."),
    (
        "data",
        "Dataset collections, the interleaving data manager and its samplers "
        "(the ``samplers`` and ``seq_curriculum`` namespaces).",
    ),
    ("environments", "Per-environment feature flags layered on top of experiments."),
    ("experimental", "Modules that are not yet promoted to a registry."),
    ("functional", "Stateless functional ops."),
    ("inference", "Text-generation entry point (``Generator``)."),
    ("integrations", "Pluggable third-party integrations (Discord, hivemind, ...)."),
    ("interface", "Terminal dashboard."),
    (
        "kb",
        "Knowledge base: sources emit normalized items, ``KBIndex`` answers "
        "ranked queries per keystroke. Backed by the spider's page store.",
    ),
    ("layers", "Shared low-level layer building blocks."),
    ("logging", "Logging utilities and formatters."),
    (
        "models",
        "Foreign models: loading a published ``transformers`` checkpoint and "
        "training it under the Praxis objectives (the ``model_tasks``, "
        "``model_adapters`` and ``peft_profiles`` namespaces).",
    ),
    ("metrics", "Metrics descriptions and bookkeeping for the dashboards."),
    (
        "modeling",
        "``PraxisModel`` / ``PraxisForCausalLM`` - the top-level transformers-compatible wrappers.",
    ),
    (
        "optimization",
        "Optimizer construction: named profiles, composite (per-parameter-group) "
        "optimizers, and the wrapper stack.",
    ),
    (
        "pillars",
        "Living research paper: builds ``research/`` from the current run via "
        "``python -m praxis.pillars.build``. Subpackages: ``framing`` (config-gated "
        "prose), ``proofs`` (consistency-checked lemmas, some verifiable computations), "
        "``inlines`` (single-value edits); plus runs/geometries/halting/ghostmax figures.",
    ),
    (
        "policies",
        "Reinforcement-learning policy losses and weight-editing controllers "
        "(the ``rl_policies`` and ``rl_profiles`` namespaces).",
    ),
    (
        "registry",
        "The registry: every pluggable choice, declared by namespace "
        "(``registry.declare``) with what each entry means, and read by name "
        "(``registry.lookup``) by the model, the CLI, these docs and the "
        "annotated config download.",
    ),
    ("schedulers", "Learning-rate schedulers."),
    ("tasks", "Training task abstractions used by ``strategies``."),
    (
        "tokenizers",
        "Tokenizer creation and registry, plus the ``chat_formats`` registry "
        "(``--chat-format``): each entry pairs a chat template with the turn "
        "boundaries, assistant mask, halting contract and tool-call layout that "
        "have to agree with it. ``default`` is ChatML with control tokens; "
        "``prose`` uses plain-text role boundaries and halts on stop strings.",
    ),
    (
        "tools",
        "Tool-calling support. ``default`` splices in-band "
        "``[TOOL_CALL]``/``[/TOOL_CALL]``; text-boundary formats promote the call "
        "and its result to turns of their own, so role boundaries delimit them.",
    ),
    (
        "trainers",
        "Lightning training loop construction (backprop, layer-wise, mono-forward, ...).",
    ),
    ("utils", "Misc helpers (system info, launch animation, lazy-module init, ...)."),
    ("web", "Flask + SocketIO dashboard and ``/input`` / ``/messages`` inference API."),
]


# Standalone subsystem docs (not registry-backed). Each tuple is
# (title, repo-relative doc path, one-line description). Single source of
# truth for both docs/index.md and the README "subsystems" block.
SUBSYSTEMS: List[Tuple[str, str, str]] = [
    ("CLI arguments", "docs/cli.md", "every `./launch` flag, grouped as in `--help`."),
    (
        "Web stack",
        "docs/web.md",
        "dashboard, JSON API routes, and inference endpoints.",
    ),
    (
        "ghost",
        "axis/ghost/README.md",
        "spectral audio visualizer and video masking editor (Godot, standalone), "
        "with its own generated reference under `axis/ghost/docs/`.",
    ),
    (
        "Vortex mobile app",
        "axis/vortex/README.md",
        "archived Godot companion app for controlling Praxis.",
    ),
    (
        "nuTube",
        "axis/nutube/README.md",
        "local-first YouTube explorer with an on-device recommender (Godot, in development).",
    ),
]


# Each tuple: (directory name, fallback description if no README is present).
# Order is the rendering order in the README "project layout" block.
TOP_LEVEL_DIRS: List[Tuple[str, str]] = [
    (
        "praxis",
        "The model framework itself. See [docs/index.md](docs/index.md) for the per-registry feature map.",
    ),
    ("experiments", ""),
    ("environments", ""),
    ("integrations", ""),
    ("staging", ""),
    ("tests", "Unit tests. Run with ``pytest tests -x``."),
    (
        "next",
        "Long-form research notes, exploratory writing, and the project [roadmap](next/roadmap.md).",
    ),
    ("evaluation", "Evaluation harnesses and helpers."),
    (
        "tools",
        "Small CLI utilities, callable by both human and assistant. See [tools/README.md](tools/README.md).",
    ),
    ("research", "The research paper, in LaTeX."),
    ("axis", "Standalone side projects with little or no Praxis integration."),
    ("docs", "Auto-generated per-registry docs. Regenerated at every launch."),
    ("static", "Images used in the README and the web dashboard."),
]


def regenerate_docs(repo_root: Optional[Path] = None) -> None:
    """Regenerate ``docs/`` and patch the README. Idempotent."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    written: List[Tuple[str, str, int, str]] = []
    for ns in registry_pages():
        content = _render_registry(ns, repo_root)
        _write_if_changed(docs_dir / f"{page_slug(ns)}.md", content)
        written.append((page_slug(ns), ns.title, len(ns), ns.doc or ""))
    _remove_stale_pages(docs_dir, {slug for slug, *_ in written})

    integration_specs = _discover_integration_specs(repo_root)
    _write_if_changed(
        docs_dir / "integrations.md",
        _render_integrations_doc(integration_specs, repo_root),
    )

    _write_if_changed(docs_dir / "web.md", _render_web_doc(repo_root))
    _write_if_changed(docs_dir / "cli.md", _render_cli_doc())
    _write_if_changed(
        docs_dir / "index.md", _render_index(written, integration_specs, repo_root)
    )

    readme_path = repo_root / "README.md"
    _patch_readme_block(readme_path, "FEATURES", _render_features_block(written))
    _patch_readme_block(readme_path, "LAYOUT", _render_layout_block(repo_root))
    _patch_readme_block(readme_path, "SUBSYSTEMS", _render_subsystems_block())

    _sort_roadmap(repo_root)
    _regenerate_terminal_webp(repo_root)
    _regenerate_web_webps(repo_root)


def _sort_roadmap(repo_root: Path) -> None:
    """Partition next/roadmap.md so open items sit above closed ones, then
    sort each group by description length: open ascending (shortest first),
    closed descending (longest first). Stable, so equal-length items keep
    their original order. Idempotent."""
    path = repo_root / "next" / "roadmap.md"
    if not path.exists():
        return
    text = path.read_text()
    first = re.search(r"^- \[[ xX]\] ", text, re.M)
    if not first:
        return
    preamble, body = text[: first.start()], text[first.start() :]
    starts = [m.start() for m in re.finditer(r"^- \[[ xX]\] ", body, re.M)]
    blocks = [
        body[s : (starts[i + 1] if i + 1 < len(starts) else len(body))].rstrip("\n")
        for i, s in enumerate(starts)
    ]
    opens = sorted((b for b in blocks if re.match(r"- \[ \] ", b)), key=len)
    closed = sorted(
        (b for b in blocks if re.match(r"- \[[xX]\] ", b)), key=len, reverse=True
    )
    rebuilt = preamble.rstrip("\n") + "\n\n" + "\n\n".join(opens + closed) + "\n"
    _write_if_changed(path, rebuilt)


# Files whose content determines the rendered terminal.webp.
def _terminal_render_sources(repo_root: Path) -> List[Path]:
    paths = [
        p
        for p in (repo_root / "praxis" / "interface").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    paths.append(repo_root / "tools" / "render_dashboard_webp.py")
    paths.append(repo_root / "tools" / "assets" / "DejaVuSansMono.ttf")
    return paths


def _terminal_webp_stale(repo_root: Path) -> bool:
    """True if any rendering source is newer than the webp (detected on the fly
    from mtimes - no stored fingerprint). The renderer bumps the webp's mtime
    even on a no-op render, so an unchanged result won't re-trigger."""
    out = repo_root / "static" / "terminal.webp"
    if not out.exists():
        return True
    cutoff = out.stat().st_mtime
    return any(
        p.exists() and p.stat().st_mtime > cutoff
        for p in _terminal_render_sources(repo_root)
    )


def _regenerate_terminal_webp(repo_root: Path) -> None:
    """Re-render static/terminal.webp when the dashboard rendering changes.

    Detached background process: the render takes ~40s and the renderer also
    hijacks logging/warnings/atexit, so we neither block the launch nor run it
    in-process. The webp's own mtime becoming newest is what clears staleness."""
    if not _terminal_webp_stale(repo_root):
        return

    out = repo_root / "static" / "terminal.webp"
    # Claim the artifact now (bump its mtime) so rapid re-launches during the
    # ~40s render don't each spawn a duplicate.
    if out.exists():
        os.utime(out, None)

    script = repo_root / "tools" / "render_dashboard_webp.py"
    log = repo_root / "static" / ".terminal_render.log"
    try:
        logf = open(log, "w")
        subprocess.Popen(
            [sys.executable, str(script), "--out", str(out)],
            cwd=str(repo_root),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            "[DOCS] Dashboard changed - rendering static/terminal.webp in background."
        )
    except OSError as e:
        print(f"[DOCS] Skipped terminal.webp: {e}")


# The web-app stills (chat, dashboard, architecture) all come from one tool.
_WEB_WEBPS = ("chat.webp", "dashboard.webp", "architecture.webp")


# Files whose content determines the rendered web-app webps.
def _web_render_sources(repo_root: Path) -> List[Path]:
    web = repo_root / "praxis" / "web"
    paths = [
        p
        for p in (web / "src").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    paths.append(web / "templates" / "index.html")
    paths.append(repo_root / "tools" / "render_web.py")
    paths.append(repo_root / "tools" / "assets" / "inter" / "inter-latin.woff2")
    return paths


def _web_webps_stale(repo_root: Path) -> bool:
    """Stale if any still is missing, or any rendering source is newer than the
    oldest still (the tool re-renders all three in one boot, so any drift
    re-fires the lot). Same on-the-fly mtime check as the terminal webp."""
    outs = [repo_root / "static" / name for name in _WEB_WEBPS]
    if any(not out.exists() for out in outs):
        return True
    cutoff = min(out.stat().st_mtime for out in outs)
    return any(
        p.exists() and p.stat().st_mtime > cutoff
        for p in _web_render_sources(repo_root)
    )


def _regenerate_web_webps(repo_root: Path) -> None:
    """Re-render the web-app stills when the web frontend changes.

    Same shape as the terminal webp: detached background process (this one
    boots the API server and a headless browser), claim via mtime bump."""
    if not _web_webps_stale(repo_root):
        return

    # Claim every output now so rapid re-launches during the render don't each
    # spawn a duplicate.
    for name in _WEB_WEBPS:
        out = repo_root / "static" / name
        if out.exists():
            os.utime(out, None)

    script = repo_root / "tools" / "render_web.py"
    log = repo_root / "static" / ".web_render.log"
    try:
        logf = open(log, "w")
        subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(
            "[DOCS] Web app changed - rendering chat/dashboard/architecture "
            "webps in background."
        )
    except OSError as e:
        print(f"[DOCS] Skipped web webps: {e}")


def _render_registry(registry: Namespace, repo_root: Path) -> str:
    lines = [
        f"<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        f"# {registry.title}",
        "",
        registry.doc or "",
        "",
        f'Namespace: ``registry.namespace("{registry.name}")``, declared in '
        f"``{registry.module}`` ({len(registry)} entries)",
        "",
    ]
    selected = _selecting_flags().get(registry.name)
    if selected:
        lines[-1:-1] = ["", f"Selected with {selected}."]
    for entry in _grouped_entries(registry):
        if "cls" in entry:
            lines.extend(_render_class_entry(registry, entry, repo_root))
        else:
            key = entry["keys"][0]
            lines.extend(_render_value_entry(registry, key, entry["value"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@functools.lru_cache(maxsize=None)
def _selecting_flags() -> Dict[str, str]:
    """``{namespace: "``--flag`` (default: x)"}`` for every namespace a CLI
    flag selects from, read off the parser so the page cannot drift from it."""
    from praxis.cli.core.parser import resolve_namespace

    out: Dict[str, List[str]] = {}
    for action in _build_cli_parser()._actions:
        bound = getattr(action, "registry", None)
        if bound is None:
            continue
        default = "unset" if action.default is None else f"``{action.default}``"
        text = f"``{action.option_strings[0]}`` (default: {default})"
        for ref in bound if isinstance(bound, tuple) else (bound,):
            out.setdefault(resolve_namespace(ref).name, []).append(text)
    return {name: ", ".join(texts) for name, texts in out.items()}


def _resolve_class(value: Any) -> Tuple[Optional[type], Optional[str]]:
    """The class a ``functools.partial`` chain builds, and its bound arguments
    as a string (``None`` for a bare class). ``(None, None)`` when the target
    isn't a class - e.g. a partial wrapping a plain factory function."""
    target, args, keywords = unwrap(value)
    if not inspect.isclass(target):
        return None, None
    parts = [_format_value(a) for a in args]
    parts += [f"{k}={_format_value(v)}" for k, v in sorted(keywords.items())]
    return target, ", ".join(parts) if parts else None


def _grouped_entries(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group entries by the class they resolve to - including ``partial``
    presets that share a base class. Values that don't resolve to a class
    (plain functions, factories, string enums) stay one-per-key."""
    class_groups: Dict[int, Dict[str, Any]] = {}
    others: List[Dict[str, Any]] = []
    for key in sorted(registry):
        value = registry[key]
        cls, config = _resolve_class(value)
        if cls is not None:
            group = class_groups.setdefault(
                id(cls), {"cls": cls, "keys": [], "presets": []}
            )
            group["keys"].append(key)
            group["presets"].append((key, config))
        else:
            others.append({"keys": [key], "value": value})
    return sorted(list(class_groups.values()) + others, key=lambda e: e["keys"][0])


def _render_class_entry(
    registry: Namespace, entry: Dict[str, Any], repo_root: Path
) -> List[str]:
    cls = entry["cls"]
    keys = entry["keys"]
    header = ", ".join(f"`{k}`" for k in keys)
    qualname = getattr(cls, "__qualname__", cls.__name__)
    link = _source_link(cls, repo_root)

    out = [f"## {header} - {qualname}", ""]
    # One key with a doc of its own: that doc says more than the class does.
    if len(keys) == 1 and registry.has_doc(keys[0]):
        summary = _fill(registry.describe(keys[0]))
    else:
        summary = _extract_summary(cls)
    if summary:
        out.extend([summary, ""])
    if link:
        out.append(f"Source: [{link.display}]({link.url})")
    else:
        module = getattr(cls, "__module__", "") or ""
        out.append(
            f"Source: `{module}` (external dependency)"
            if module and not module.startswith("praxis")
            else "Source: (unknown)"
        )

    presets = entry["presets"]
    documented = len(keys) > 1 and any(registry.has_doc(k) for k in keys)
    if any(config for _, config in presets) or documented:
        out.extend(["", "Presets:"])
        for key, config in presets:
            doc = (
                registry.describe(key)
                if len(keys) > 1 and registry.has_doc(key)
                else None
            )
            if doc:
                where = f" (`{config}`)" if config else ""
                out.append(f"- `{key}`{where} - {doc}")
            else:
                out.append(
                    f"- `{key}` - {f'`{config}`' if config else 'class defaults'}"
                )
    return out


def _render_value_entry(registry: Namespace, key: str, value: Any) -> List[str]:
    out = [f"## `{key}`", ""]
    if registry.has_doc(key):
        out.append(_fill(registry.describe(key)))
    else:
        out.append(f"Value: `{_format_value(value)}`")
    return out


def _fill(text: str) -> str:
    return textwrap.fill(text, width=88, replace_whitespace=True)


def _discover_integration_specs(repo_root: Path) -> List[Any]:
    """Discover the integrations under ``integrations/`` from their spec.yaml.
    Returns IntegrationSpec objects sorted by name; empty list on failure."""
    try:
        from praxis.integrations.loader import IntegrationLoader
    except Exception:
        return []
    loader = IntegrationLoader(str(repo_root / "integrations"))
    return sorted(loader.discover_integrations(), key=lambda s: s.name)


def _render_integrations_doc(specs: List[Any], repo_root: Path) -> str:
    """docs/integrations.md: the third-party, non-core add-ons in this repo."""
    lines = [
        "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        "# Integrations",
        "",
        textwrap.fill(
            "**Third-party, non-core add-ons.** These live in the repo for "
            "convenience, but they are not part of the Praxis framework proper. "
            "Each one is optional and self-contained under `integrations/<name>/`, "
            "loads only when its activation flag is set, and installs its own "
            "dependencies on demand. Praxis runs fine with none of them enabled.",
            width=88,
        ),
        "",
        "See [`integrations/README.md`](../integrations/README.md) for the "
        "directory layout and how to add one.",
        "",
    ]
    if not specs:
        lines.append("_(no integrations found)_")
        return "\n".join(lines).rstrip() + "\n"
    for spec in specs:
        lines.extend(_render_integration_entry(spec, repo_root))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_integration_entry(spec: Any, repo_root: Path) -> List[str]:
    out = [f"## `{spec.name}` (v{spec.version})", ""]
    if spec.description:
        out.extend([spec.description, ""])
    out.append(f"- Activation: {_integration_flags(spec.conditions)}")
    capabilities = _integration_capabilities(spec)
    if capabilities:
        out.append(f"- Provides: {capabilities}")
    out.append(
        f"- Source: [`integrations/{spec.path.name}/`]({_integration_link(spec, repo_root)})"
    )
    return out


def _integration_flags(conditions: List[str]) -> str:
    """Translate spec conditions (``args.foo``) into ``--foo`` activation flags."""
    flags: List[str] = []
    for cond in conditions:
        for name in re.findall(r"args\.(\w+)", str(cond)):
            flag = "--" + name.replace("_", "-")
            if flag not in flags:
                flags.append(flag)
    if not flags:
        return "always on when installed"
    return ", ".join(f"`{f}`" for f in flags)


def _integration_capabilities(spec: Any) -> str:
    """Sorted, deduplicated capability names from ``provides`` + ``integrations``."""
    caps = set(spec.provides) | set(spec.integrations.keys())
    return ", ".join(f"`{c}`" for c in sorted(caps))


def _integration_link(spec: Any, repo_root: Path) -> str:
    """Link from docs/ to an integration's README (or its directory)."""
    try:
        rel = spec.path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = Path("integrations") / spec.path.name
    if (spec.path / "README.md").exists():
        rel = rel / "README.md"
    return (Path("..") / rel).as_posix()


# Endpoints whose name should be excluded from the API table.
_ROUTE_SKIP_BLUEPRINTS = {"static_files"}

# Friendly headings for each blueprint group. Anything not listed falls
# back to the blueprint name title-cased.
_BLUEPRINT_TITLES = {
    "core": "Core",
    "generation": "Generation",
    "agents": "Agents",
    "metrics": "Metrics",
    "dynamics": "Dynamics",
    "data_metrics": "Data metrics",
    "git": "Git HTTP backend",
}


def _render_web_doc(repo_root: Path) -> str:
    """Top-level docs/web.md: architecture, route table, inference examples."""
    lines = [
        "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        "# Web stack",
        "",
        "The dashboard at <http://localhost:2100> and the JSON API that backs it.",
        "Two halves:",
        "",
        "- **Backend** (Flask + Socket.IO, [`praxis/web/`](../praxis/web/)) - serves the dashboard and exposes a JSON API for inference, metrics, runs, and live training dynamics.",
        "- **Frontend** (vanilla JavaScript, [`praxis/web/src/`](../praxis/web/src/)) - no frameworks, no npm, no build chain beyond a small Python concatenator.",
        "",
        "## Design philosophy",
        "",
        "The frontend is intentionally data-driven. The whole UI is a pure function of state:",
        "",
        "```",
        "events → update state → render(state) → DOM",
        "```",
        "",
        "Components are reusable building blocks that take data and emit HTML; they hold no state of their own. New tabs, new chart types, new agent panels are added by extending the state object and writing one more pure component - not by adding another framework abstraction. See [`praxis/web/src/README.md`](../praxis/web/src/README.md) for the frontend conventions in detail.",
        "",
        "The backend mirrors this: each Flask blueprint is a thin shim over a data source (a SQLite run database, the active model, a git repo). Routes return JSON; the frontend handles all presentation.",
        "",
        "## API routes",
        "",
        "Introspected from the live Flask app at every launch. Each summary is the first paragraph of the view function's docstring.",
        "",
    ]
    lines.extend(_render_route_table(repo_root))
    lines.extend(
        [
            "",
            "## Sending prompts",
            "",
            "Both endpoints accept any argument from the [Transformers text generation API](https://huggingface.co/docs/transformers/en/main_classes/text_generation).",
            "",
            "### String-based generation",
            "",
            "```py",
            "import requests",
            "",
            'url = "http://localhost:2100/input"',
            'payload = {"prompt": "Once upon a time, ", "do_sample": True, "temperature": 0.7}',
            "",
            "response = requests.post(url, json=payload)",
            "print(response.json())",
            "```",
            "",
            "### Message-based generation (recommended)",
            "",
            "Supports system prompts and tool calls.",
            "",
            "```py",
            "import requests",
            "",
            'url = "http://localhost:2100/messages"',
            "payload = {",
            '    "messages": [',
            '        {"role": "system", "content": "You are a helpful assistant."},',
            '        {"role": "user", "content": "Hello, how are you?"},',
            "    ],",
            '    "do_sample": True,',
            '    "temperature": 0.7,',
            "}",
            "",
            "response = requests.post(url, json=payload)",
            "print(response.json())",
            "```",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_route_table(repo_root: Path) -> List[str]:
    """Spin up a throwaway Flask app, register routes, enumerate. Grouped
    by blueprint, deduplicated by view function so `/foo` and `/foo/` show
    up as one row with both paths."""
    try:
        from flask import Flask

        from praxis.web.routes import register_routes
    except Exception as e:
        return [f"_(route introspection failed: {e})_"]

    app = Flask(__name__)
    register_routes(app)

    by_view: Dict[str, Dict[str, Any]] = {}
    for rule in app.url_map.iter_rules():
        blueprint = rule.endpoint.split(".", 1)[0]
        if blueprint in _ROUTE_SKIP_BLUEPRINTS or rule.endpoint == "static":
            continue
        view = app.view_functions.get(rule.endpoint)
        if not view:
            continue
        methods = set(rule.methods) - {"HEAD", "OPTIONS"}
        entry = by_view.setdefault(
            rule.endpoint,
            {
                "blueprint": blueprint,
                "paths": set(),
                "methods": set(),
                "view": view,
            },
        )
        entry["paths"].add(rule.rule)
        entry["methods"] |= methods

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for endpoint, info in by_view.items():
        info["endpoint"] = endpoint
        groups.setdefault(info["blueprint"], []).append(info)

    out: List[str] = []
    for bp in sorted(groups, key=lambda b: _BLUEPRINT_TITLES.get(b, b).lower()):
        out.append(f"### {_BLUEPRINT_TITLES.get(bp, bp.replace('_', ' ').title())}")
        out.append("")
        for info in sorted(groups[bp], key=lambda i: sorted(i["paths"])[0]):
            paths = " / ".join(f"`{p}`" for p in sorted(info["paths"]))
            methods = ", ".join(sorted(info["methods"]))
            # First paragraph, collapsed to one line - so a summary that
            # wraps across source lines isn't cut mid-sentence.
            doc = " ".join(
                (inspect.getdoc(info["view"]) or "").split("\n\n", 1)[0].split()
            )
            link = _source_link(info["view"], repo_root)
            row = f"- **{methods}** {paths}"
            if doc:
                row += f" - {doc}"
            if link:
                row += f" ([source]({link.url}))"
            out.append(row)
        out.append("")
    return out


def _build_cli_parser():
    """The static CLI parser (see ``build_static_parser``)."""
    from praxis.cli.groups import build_static_parser

    return build_static_parser()


# Subcommands handled by the `./launch` bash wrapper before Python runs, so
# they never reach argparse and can't be introspected. Maintained by hand;
# the launch script changes rarely.
_LAUNCH_COMMANDS_SECTION = [
    "### launch script",
    "",
    "Handled by the `./launch` wrapper itself (before Python), so they do not "
    "appear in `--help`. Maintained by hand.",
    "",
    "| Command | Description |",
    "| --- | --- |",
    "| `./launch` | Set up / reuse the virtualenv and start a normal training run. |",
    "| `./launch stop` | Tear down a running Docker Compose stack. |",
    "| `./launch compose [args...]` | Run inside Docker Compose (auto-detects ARM vs x86_64); forwards all remaining args to a containerized `./launch`. |",
    "| `./launch test [pytest args...]` | Run the test suite via pytest; forwards all remaining args (e.g. `./launch test -x`). |",
    "",
]


def _render_cli_doc() -> str:
    """docs/cli.md: every ./launch flag, grouped as in --help."""
    import argparse

    try:
        parser = _build_cli_parser()
        # Some defaults are computed at build time and differ every run
        # (e.g. --seed uses random.random()). Build a second parser and
        # diff defaults to flag those, so the doc doesn't churn.
        second = {a.dest: a.default for a in _build_cli_parser()._actions}
        nondeterministic = {
            a.dest
            for a in parser._actions
            if a.dest in second and repr(a.default) != repr(second[a.dest])
        }
    except Exception as e:
        return (
            "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->\n"
            f"# CLI arguments\n\n_(CLI introspection failed: {e})_\n"
        )

    lines = [
        "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        "# CLI arguments",
        "",
        "Every flag accepted by `./launch`, grouped as in `./launch --help`.",
        "Each flag is also settable via a `PRAXIS_<DEST>` environment variable "
        "(e.g. `--batch-size` -> `PRAXIS_BATCH_SIZE`); precedence is explicit CLI "
        "> active environment YAML > env var > experiment YAML > default.",
        "",
        "> Experiment YAMLs (`experiments/`) and environment YAMLs "
        "(`environments/`) each add their own `--<name>` toggle flag, and "
        "integrations add their own arguments. Those are local to your checkout "
        "and so are not listed here - run `./launch --help` to see everything "
        "active in your setup.",
        "",
    ]
    lines.extend(_LAUNCH_COMMANDS_SECTION)
    for group in parser._action_groups:
        actions = [
            a
            for a in group._group_actions
            if not isinstance(a, argparse._HelpAction) and a.option_strings
        ]
        if not actions:
            continue
        lines.append(f"### {group.title or 'options'}")
        lines.append("")
        lines.append("| Flag | Type | Default | Description |")
        lines.append("| --- | --- | --- | --- |")
        for action in sorted(actions, key=lambda a: a.option_strings[0]):
            lines.append(_format_cli_row(action, action.dest in nondeterministic))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _md_cell(text: str) -> str:
    """Escape a string for use inside a markdown table cell."""
    return text.replace("\n", " ").replace("|", "\\|")


def _format_cli_row(action, default_varies: bool) -> str:
    import argparse

    flags = ", ".join(f"`{o}`" for o in action.option_strings)

    type_str = ""
    if action.type is not None and hasattr(action.type, "__name__"):
        type_str = action.type.__name__
        if type_str == "<lambda>":
            type_str = "str"
    elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        type_str = "bool"

    if default_varies:
        default_str = "_varies_"
    elif action.default is argparse.SUPPRESS:
        default_str = ""
    else:
        default_str = f"`{action.default}`"

    # Drop a trailing "(default: ...)" baked into some help strings - the
    # Default column already carries it.
    help_text = " ".join((action.help or "").split()).strip()
    help_text = re.sub(r"\s*\(default:[^)]*\)\s*$", "", help_text)
    if action.choices:
        help_text += " (choices: " + ", ".join(str(c) for c in action.choices) + ")"

    cells = [_md_cell(c) for c in (flags, type_str, default_str, help_text)]
    return f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"


def _render_index(
    written: List[Tuple[str, str, int, str]],
    integration_specs: List[Any],
    repo_root: Path,
) -> str:
    lines = [
        "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        "# Praxis docs index",
        "",
        f"Praxis is built around {count_namespaces()} registry namespaces. The "
        "feature categories below link to a page listing the registered "
        "implementations and their source.",
        "",
        "## Feature registries",
        "",
    ]
    for slug, title, count, description in sorted(written, key=lambda e: e[1].lower()):
        lines.append(f"- [{title}]({slug}.md) ({count}) - {description}")
    lines.extend(["", "## Subsystems", ""])
    for title, path, description in SUBSYSTEMS:
        lines.append(f"- [{title}]({_docs_relative(path)}) - {description}")
    lines.extend(["", "## Project layout", ""])
    lines.append(_render_layout_block(repo_root, link_prefix="../"))
    lines.extend(["", "## Core infrastructure", ""])
    lines.append(
        "These packages don't expose a registry yet; they are the infrastructure"
    )
    lines.append("the registries plug into. One-liner per package; consult")
    lines.append("the package directory for details.")
    lines.append("")
    for slug, description in INFRASTRUCTURE_PACKAGES:
        lines.append(f"- `praxis/{slug}/` - {description}")
    if integration_specs:
        lines.extend(["", "## Integrations", ""])
        lines.append(
            "Third-party, non-core add-ons that live in the repo but are not part"
        )
        lines.append(
            "of the framework. Optional and off by default - see "
            "[integrations](integrations.md)."
        )
        lines.append("")
        for spec in integration_specs:
            lines.append(f"- `{spec.name}` - {spec.description}")
    lines.extend(
        [
            "",
            "---",
            "",
            "This index is regenerated on every launch. To opt out, pass `--no-docs`.",
            "",
        ]
    )
    return "\n".join(lines)


def _patch_readme_block(readme_path: Path, name: str, body: str) -> None:
    """Replace content between ``<!-- AUTODOC:{name}:BEGIN -->`` and the
    matching END marker in the README. No-op if markers are missing."""
    if not readme_path.exists():
        return
    current = readme_path.read_text()
    begin = f"<!-- AUTODOC:{name}:BEGIN -->"
    end = f"<!-- AUTODOC:{name}:END -->"
    if begin not in current or end not in current:
        return

    block = f"{begin}\n\n{body.rstrip()}\n\n{end}"
    before, _, rest = current.partition(begin)
    _, _, after = rest.partition(end)
    _write_if_changed(readme_path, before + block + after)


def _render_features_block(written: List[Tuple[str, str, int, str]]) -> str:
    lines = [
        f"Praxis is organized as {count_namespaces()} registry namespaces. The "
        "feature categories below link to a docs page listing the concrete "
        "implementations and their source. See [docs/index.md](docs/index.md) "
        "for the full map.",
        "",
    ]
    for slug, title, count, _description in sorted(written, key=lambda e: e[1].lower()):
        lines.append(f"- [{title}](docs/{slug}.md) ({count})")
    return "\n".join(lines)


def _render_subsystems_block() -> str:
    lines = [
        "Standalone subsystems, documented outside the registry map.",
        "",
    ]
    for title, path, description in SUBSYSTEMS:
        lines.append(f"- [{title}]({path}) - {description}")
    return "\n".join(lines)


def _docs_relative(path: str) -> str:
    """Rebase a repo-relative doc path to one relative to docs/index.md."""
    return path[len("docs/") :] if path.startswith("docs/") else f"../{path}"


def _render_layout_block(repo_root: Path, link_prefix: str = "") -> str:
    """Render the top-level directory map. ``link_prefix`` rebases the links
    for callers outside repo root (e.g. ``../`` for docs/index.md)."""
    lines = [
        "Top-level directories, with detail sourced from each one's README",
        "where present.",
        "",
    ]
    for name, fallback in sorted(TOP_LEVEL_DIRS, key=lambda e: e[0].lower()):
        path = repo_root / name
        if not path.is_dir():
            continue
        readme = path / "README.md"
        if readme.exists():
            summary = _first_readme_paragraph(readme.read_text())
            link = f"[`{name}/`]({link_prefix}{name}/README.md)"
        else:
            summary = fallback
            link = f"[`{name}/`]({link_prefix}{name}/)"
        if not summary:
            summary = "(undocumented)"
        lines.append(f"- **{link}** - {_reprefix_links(summary, link_prefix)}")
    return "\n".join(lines)


def _reprefix_links(text: str, link_prefix: str) -> str:
    """Rebase repo-root-relative markdown link targets embedded in summary
    text. Leaves absolute, anchor, and already-relative links untouched."""
    if not link_prefix:
        return text
    return re.sub(
        r"\]\((?!https?:|/|#|\.\.?/)([^)]+)\)",
        lambda m: f"]({link_prefix}{m.group(1)})",
        text,
    )


def _first_readme_paragraph(text: str) -> str:
    """First non-heading paragraph of a markdown README, single line, capped."""
    paragraphs = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        paragraphs.append(" ".join(line.strip() for line in chunk.splitlines()))
        break
    if not paragraphs:
        return ""
    out = paragraphs[0]
    if len(out) > 300:
        out = out[:300].rsplit(" ", 1)[0] + " ..."
    return out


# ---------- helpers ----------


class _Link:
    __slots__ = ("display", "url")

    def __init__(self, display: str, url: str) -> None:
        self.display = display
        self.url = url


def _format_value(value: Any) -> str:
    """Deterministic repr for a non-class registry entry. Stripping the
    ``<function foo at 0x...>`` address means docs/*.md stays stable
    across launches even when Python relocates the function in memory."""
    if isinstance(value, functools.partial):
        parts = [_format_value(value.func)]
        parts.extend(_format_value(a) for a in value.args)
        parts.extend(
            f"{k}={_format_value(v)}" for k, v in sorted(value.keywords.items())
        )
        return f"functools.partial({', '.join(parts)})"
    if inspect.isclass(value):
        module = getattr(value, "__module__", "") or ""
        return (
            f"<class '{module}.{value.__qualname__}'>"
            if module
            else f"<class '{value.__qualname__}'>"
        )
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        module = getattr(value, "__module__", "") or ""
        qualname = getattr(value, "__qualname__", value.__name__)
        return f"<function {module}.{qualname}>" if module else f"<function {qualname}>"
    return repr(value)


def _source_link(cls: type, repo_root: Path) -> Optional[_Link]:
    try:
        path = Path(inspect.getsourcefile(cls) or inspect.getfile(cls)).resolve()
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return None
    # Don't link into installed dependencies: those paths carry a Python
    # version, churn across machines, and don't exist in the repo.
    if "site-packages" in path.parts or ".venv" in path.parts:
        return None
    try:
        rel = path.relative_to(repo_root.resolve())
    except ValueError:
        return None
    rel_from_docs = Path("..") / rel
    return _Link(display=f"{rel}:{line}", url=f"{rel_from_docs.as_posix()}#L{line}")


def _extract_summary(cls: type) -> str:
    """Lead of the class docstring (or its module's), trimmed at the first
    Args/Returns section and capped at ~500 chars."""
    body = docstring_body(cls)
    if len(body) > 500:
        body = body[:500].rsplit(" ", 1)[0] + " ..."
    paragraphs = []
    for p in body.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        # Preserve markdown lists; only fill prose paragraphs.
        if any(
            line.lstrip().startswith(("- ", "* ", "1.", "2.", "3."))
            for line in p.splitlines()
        ):
            paragraphs.append(p)
        else:
            paragraphs.append(_fill(p))
    return "\n\n".join(paragraphs)


def undocumented_registries() -> List[str]:
    """Namespaces a CLI flag selects from that have no docs page (no
    ``title``). The drift guard behind ``tests/test_docs.py``."""
    from praxis.cli.core.parser import resolve_namespace

    missing = set()
    for action in _build_cli_parser()._actions:
        bound = getattr(action, "registry", None)
        for ref in bound if isinstance(bound, tuple) else (bound,):
            ns = resolve_namespace(ref) if ref is not None else None
            if ns is not None and not ns.title:
                missing.add(f"{ns.name} ({action.option_strings[0]})")
    return sorted(missing)


# Generated pages that are not registry pages.
_FIXED_PAGES = {"cli", "index", "integrations", "web"}


def _remove_stale_pages(docs_dir: Path, current: set) -> None:
    """Delete generated pages whose namespace no longer exists (renamed or
    removed). Only files carrying the AUTOGENERATED marker are touched."""
    for path in docs_dir.glob("*.md"):
        if path.stem in current or path.stem in _FIXED_PAGES:
            continue
        if path.read_text().startswith("<!-- AUTOGENERATED by praxis/docs.py"):
            path.unlink()


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


if __name__ == "__main__":
    regenerate_docs()
    print("Regenerated docs/ and patched README.md.")
