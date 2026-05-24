"""Auto-docs generator for Praxis.

Walks every registry exposed from ``praxis.__init__``, introspects each
registered class, and writes one markdown file per category under ``docs/``.
Also patches the project README between two AUTODOC markers.

Runs at every launch (see ``main.py`` ``setup_environment``). Writes are
idempotent: a file is only touched when its content actually changes, so
this never produces spurious git diffs.

To skip generation: ``./launch --no-docs``.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import praxis


# Each tuple: (slug, title, registry, category description) and optionally
# a 5th element - a {key: description} dict that overrides docstring/value
# rendering for entries that aren't introspectable classes (e.g. string
# enums). Slug is the filename stem under docs/.
def _registries() -> List[Tuple]:
    return [
        (
            "activations",
            "Activation functions",
            praxis.ACTIVATION_REGISTRY,
            "Pointwise nonlinearities used inside blocks and heads.",
        ),
        (
            "attention",
            "Attention mechanisms",
            praxis.ATTENTION_REGISTRY,
            "Self-attention variants, from vanilla causal MHA to compressive-memory and per-depth-biased variants.",
        ),
        (
            "blocks",
            "Decoder block layouts",
            praxis.BLOCK_REGISTRY,
            "Top-level layer types the decoder stacks. Mix attention-based and recurrent designs freely.",
        ),
        (
            "compression",
            "Sequence compression",
            praxis.COMPRESSION_REGISTRY,
            "Strategies for reducing sequence length between layers.",
        ),
        (
            "controllers",
            "Layer-routing controllers",
            praxis.CONTROLLER_REGISTRY,
            "Decide which expert / block a token visits at each depth. Enables out-of-order layers and graph-style routing.",
        ),
        (
            "data",
            "Data sampler strategies",
            praxis.SAMPLER_REGISTRY,
            "How datasets are interleaved during training. Praxis trains on multiple "
            "datasets at once: at every step the trainer picks a dataset, draws a "
            "document, and tokenizes it (see ``InterleaveDataManager`` in "
            "``praxis/data/datasets/manager.py``). The sampler chosen here decides "
            "*how* that pick is biased - either statically from configured weights, "
            "or adaptively based on document length, novelty, or per-dataset loss. "
            "Set with ``--sampler``; default is ``novelty``.",
            praxis.data.SAMPLER_DESCRIPTIONS,
        ),
        (
            "decoders",
            "Block-stacking decoders",
            praxis.DECODER_REGISTRY,
            "How the stack of blocks is composed (sequential, parallel, weighted, ...).",
        ),
        (
            "embeddings",
            "Token embeddings",
            praxis.EMBEDDING_REGISTRY,
            "Input embedding layers, paired with the corresponding block type.",
        ),
        (
            "encoders",
            "Input encoders",
            praxis.ENCODER_REGISTRY,
            "Front-end encoders, including the byte-latent and abstractinator variants.",
        ),
        (
            "encoding",
            "Positional encoding",
            praxis.ENCODING_REGISTRY,
            "RoPE, ALiBi, NoPE and friends - the rotational / additive position priors injected into attention.",
        ),
        (
            "halting",
            "Halting / early exit",
            praxis.HALTING_REGISTRY,
            "Per-token mechanisms for early exit from recurrent depth loops.",
        ),
        (
            "heads",
            "Output heads",
            {**praxis.HEAD_REGISTRY, **praxis.MTP_REGISTRY},
            "LM heads (tied/untied, harmonic, crystal) and multi-token-prediction wrappers.",
        ),
        (
            "losses",
            "Loss functions",
            praxis.LOSS_REGISTRY,
            "Per-token criteria. Most accept optional ``loss_weights`` for task-weighted training.",
        ),
        (
            "normalization",
            "Normalization layers",
            praxis.NORMALIZATION_REGISTRY,
            "LayerNorm/RMSNorm variants, including SandwichNorm (required for stable recurrent-depth bias).",
        ),
        (
            "orchestration",
            "Expert orchestration",
            praxis.EXPERT_REGISTRY,
            "How a block's feedforward path is realized: MLP, GLU, KAN, PEER, ...",
        ),
        (
            "policies",
            "RL policies",
            praxis.RL_POLICIES_REGISTRY,
            "Reinforcement-learning policy losses (REINFORCE, GRPO, ...) for post-training.",
        ),
        (
            "residuals",
            "Residual connections",
            praxis.RESIDUAL_REGISTRY,
            "Standard residuals vs. hyper-connections.",
        ),
        (
            "routers",
            "Token routers",
            praxis.ROUTER_REGISTRY,
            "Token-routing mechanisms, including the Mixture-of-Depths family that skips a fraction of tokens per layer.",
        ),
        (
            "sorting",
            "Sequence sorting",
            praxis.SORTING_REGISTRY,
            "Optional reordering operations applied to the sequence.",
        ),
        (
            "strategies",
            "Training strategies",
            praxis.STRATEGIES_REGISTRY,
            "Multi-task / task-weighting strategies used by the trainer.",
        ),
    ]


# Hand-curated one-liners for packages that don't have a registry.
# Listed in docs/index.md as "core infrastructure" pointers.
NON_REGISTRY_PACKAGES: List[Tuple[str, str]] = [
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
    ("environments", "Per-environment feature flags layered on top of experiments."),
    ("experimental", "Modules that are not yet promoted to a registry."),
    ("functional", "Stateless functional ops."),
    ("generation", "Text-generation entry point (``Generator``)."),
    ("integrations", "Pluggable third-party integrations (Discord, hivemind, ...)."),
    ("interface", "Terminal dashboard."),
    ("layers", "Shared low-level layer building blocks."),
    ("logging", "Logging utilities and formatters."),
    ("metrics", "Metrics descriptions and bookkeeping for the dashboards."),
    (
        "modeling",
        "``PraxisModel`` / ``PraxisForCausalLM`` - the top-level transformers-compatible wrappers.",
    ),
    ("optimizers", "Optimizer registry and parameter-grouping helpers."),
    ("recurrent", "Recurrent-depth utilities (looped layer execution)."),
    ("schedulers", "Learning-rate schedulers."),
    ("tasks", "Training task abstractions used by ``strategies``."),
    ("tokenizers", "Tokenizer creation and registry."),
    (
        "tools",
        "Tool-calling support (in-band ``[TOOL_CALL]``/``[/TOOL_CALL]`` splices).",
    ),
    (
        "trainers",
        "Lightning training loop construction (backprop, layer-wise, mono-forward, ...).",
    ),
    ("utils", "Misc helpers (system info, launch animation, lazy-module init, ...)."),
    ("web", "Flask + SocketIO dashboard and ``/input`` / ``/messages`` inference API."),
]


# Each tuple: (directory name, fallback description if no README is present).
# Order is the rendering order in the README "project layout" block.
TOP_LEVEL_DIRS: List[Tuple[str, str]] = [
    ("praxis", "The model framework itself. See [docs/index.md](docs/index.md) for the per-registry feature map."),
    ("experiments", ""),
    ("environments", ""),
    ("integrations", ""),
    ("staging", ""),
    ("tests", "Unit tests. Run with ``pytest tests -x``."),
    ("next", "Long-form research notes, exploratory writing, and the project [roadmap](next/roadmap.md)."),
    ("evaluation", "Evaluation harnesses and helpers."),
    ("proofs", "Math / derivation notes backing the more unusual designs (harmonic head, ghostmax)."),
    ("research", "The research paper, in LaTeX."),
    ("axis", "Mobile companion app, built with Godot."),
    ("docs", "Auto-generated per-registry docs. Regenerated at every launch."),
    ("static", "Images used in the README and the web dashboard."),
]


def regenerate_docs(repo_root: Optional[Path] = None) -> None:
    """Regenerate ``docs/`` and patch the README. Idempotent."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    written: List[Tuple[str, str, int, str]] = []
    for entry in _registries():
        slug, title, registry, description = entry[:4]
        overrides = entry[4] if len(entry) > 4 else None
        content = _render_registry(
            slug, title, registry, description, repo_root, overrides
        )
        _write_if_changed(docs_dir / f"{slug}.md", content)
        written.append((slug, title, len(registry), description))

    _write_if_changed(docs_dir / "web.md", _render_web_doc(repo_root))
    _write_if_changed(docs_dir / "index.md", _render_index(written))

    readme_path = repo_root / "README.md"
    _patch_readme_block(readme_path, "FEATURES", _render_features_block(written))
    _patch_readme_block(readme_path, "LAYOUT", _render_layout_block(repo_root))


def _render_registry(
    slug: str,
    title: str,
    registry: Dict[str, Any],
    description: str,
    repo_root: Path,
    overrides: Optional[Dict[str, str]] = None,
) -> str:
    lines = [
        f"<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        f"# {title}",
        "",
        description,
        "",
        f"Registry: ``praxis.{_registry_attr(slug)}`` ({len(registry)} entries)",
        "",
    ]
    for keys, value in _grouped_entries(registry):
        override = overrides.get(keys[0]) if overrides else None
        lines.extend(_render_entry(keys, value, repo_root, override))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _grouped_entries(registry: Dict[str, Any]) -> List[Tuple[List[str], Any]]:
    """Collapse multiple keys that point to the same class into a single
    entry. Non-class values (partials, strings) stay one-per-key, since
    they may differ even when wrapping the same target."""
    class_groups: Dict[int, Tuple[Any, List[str]]] = {}
    others: List[Tuple[List[str], Any]] = []
    for key in sorted(registry):
        value = registry[key]
        if inspect.isclass(value):
            existing = class_groups.get(id(value))
            if existing:
                existing[1].append(key)
            else:
                class_groups[id(value)] = (value, [key])
        else:
            others.append(([key], value))
    grouped = [(keys, cls) for cls, keys in class_groups.values()]
    return sorted(grouped + others, key=lambda entry: entry[0][0])


def _render_entry(
    keys: List[str],
    value: Any,
    repo_root: Path,
    override: Optional[str] = None,
) -> List[str]:
    header = ", ".join(f"`{k}`" for k in keys)
    if not inspect.isclass(value):
        out = [f"## {header}", ""]
        if override:
            out.append(textwrap.fill(override, width=88, replace_whitespace=True))
        else:
            out.append(f"Value: `{value!r}`")
        return out

    qualname = getattr(value, "__qualname__", value.__name__)
    summary = override or _extract_summary(value)
    if override:
        summary = textwrap.fill(summary, width=88, replace_whitespace=True)
    link = _source_link(value, repo_root)

    out = [f"## {header} - {qualname}", ""]
    if summary:
        out.extend([summary, ""])
    if link:
        out.append(f"Source: [{link.display}]({link.url})")
    else:
        out.append("Source: (unknown)")
    return out


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
        "Introspected from the live Flask app at every launch. Each summary is the first line of the view function's docstring.",
        "",
    ]
    lines.extend(_render_route_table(repo_root))
    lines.extend([
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
    ])
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
        entry = by_view.setdefault(rule.endpoint, {
            "blueprint": blueprint,
            "paths": set(),
            "methods": set(),
            "view": view,
        })
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
            doc = (inspect.getdoc(info["view"]) or "").split("\n")[0].strip()
            link = _source_link(info["view"], repo_root)
            row = f"- **{methods}** {paths}"
            if doc:
                row += f" - {doc}"
            if link:
                row += f" ([source]({link.url}))"
            out.append(row)
        out.append("")
    return out


def _render_index(written: List[Tuple[str, str, int, str]]) -> str:
    lines = [
        "<!-- AUTOGENERATED by praxis/docs.py - do not edit by hand -->",
        "# Praxis docs index",
        "",
        "Praxis is built around ~20 pluggable registries. Each category below ",
        "links to a page listing the registered implementations and their source.",
        "",
        "## Feature registries",
        "",
    ]
    for slug, title, count, description in sorted(written, key=lambda e: e[1].lower()):
        lines.append(f"- [{title}]({slug}.md) ({count}) - {description}")
    lines.extend([
        "",
        "## Subsystems",
        "",
        "- [Web stack](web.md) - dashboard, JSON API routes, and inference endpoints.",
    ])
    lines.extend(["", "## Core infrastructure", ""])
    lines.append(
        "These packages don't expose a registry yet, but they're the load-bearing"
    )
    lines.append(
        "infrastructure the registries plug into. One-liner per package; consult"
    )
    lines.append("the package directory for details.")
    lines.append("")
    for slug, description in NON_REGISTRY_PACKAGES:
        lines.append(f"- `praxis/{slug}/` - {description}")
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
        "Praxis is organized as ~20 pluggable registries. Each category below",
        "links to a docs page listing the concrete implementations and their",
        "source. See [docs/index.md](docs/index.md) for the full map.",
        "",
    ]
    for slug, title, count, _description in sorted(written, key=lambda e: e[1].lower()):
        lines.append(f"- [{title}](docs/{slug}.md) ({count})")
    return "\n".join(lines)


def _render_layout_block(repo_root: Path) -> str:
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
            link = f"[`{name}/`]({name}/README.md)"
        else:
            summary = fallback
            link = f"[`{name}/`]({name}/)"
        if not summary:
            summary = "(undocumented)"
        lines.append(f"- **{link}** - {summary}")
    return "\n".join(lines)


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


def _source_link(cls: type, repo_root: Path) -> Optional[_Link]:
    try:
        path = Path(inspect.getsourcefile(cls) or inspect.getfile(cls))
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    rel_from_docs = Path("..") / rel
    return _Link(display=f"{rel}:{line}", url=f"{rel_from_docs.as_posix()}#L{line}")


_SECTION_HEADERS = ("Args:", "Arguments:", "Returns:", "Raises:", "Yields:",
                    "Example:", "Examples:", "Note:", "Notes:", "Attributes:")


def _extract_summary(cls: type) -> str:
    """Lead of the class docstring (or its module's), trimmed at the first
    Args/Returns section and capped at ~500 chars."""
    for source in (inspect.getdoc(cls), inspect.getdoc(inspect.getmodule(cls))):
        if not source:
            continue
        # Stop at the first section header (Args:, Returns:, etc).
        lines = source.splitlines()
        cut = len(lines)
        for i, line in enumerate(lines):
            if line.strip() in _SECTION_HEADERS:
                cut = i
                break
        body = "\n".join(lines[:cut]).strip()
        if len(body) > 500:
            body = body[:500].rsplit(" ", 1)[0] + " ..."
        if body:
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
                    paragraphs.append(
                        textwrap.fill(p, width=88, replace_whitespace=True)
                    )
            return "\n\n".join(paragraphs)
    return ""


def _registry_attr(slug: str) -> str:
    """Map a doc slug back to the registry variable name in ``praxis``."""
    return {
        "activations": "ACTIVATION_REGISTRY",
        "attention": "ATTENTION_REGISTRY",
        "blocks": "BLOCK_REGISTRY",
        "compression": "COMPRESSION_REGISTRY",
        "controllers": "CONTROLLER_REGISTRY",
        "data": "SAMPLER_REGISTRY",
        "decoders": "DECODER_REGISTRY",
        "embeddings": "EMBEDDING_REGISTRY",
        "encoders": "ENCODER_REGISTRY",
        "encoding": "ENCODING_REGISTRY",
        "halting": "HALTING_REGISTRY",
        "heads": "HEAD_REGISTRY + MTP_REGISTRY",
        "losses": "LOSS_REGISTRY",
        "normalization": "NORMALIZATION_REGISTRY",
        "orchestration": "EXPERT_REGISTRY",
        "policies": "RL_POLICIES_REGISTRY",
        "residuals": "RESIDUAL_REGISTRY",
        "routers": "ROUTER_REGISTRY",
        "sorting": "SORTING_REGISTRY",
        "strategies": "STRATEGIES_REGISTRY",
    }[slug]


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
