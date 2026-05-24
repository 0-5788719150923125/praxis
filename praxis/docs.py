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


# Each tuple: (slug, title, registry, one-line category description)
# Slug is used for the filename (docs/<slug>.md) and the README link.
def _registries() -> List[Tuple[str, str, Dict[str, Any], str]]:
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
            "Curriculum and difficulty-weighting strategies for batch sampling.",
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


_README_BEGIN = "<!-- AUTODOC:FEATURES:BEGIN -->"
_README_END = "<!-- AUTODOC:FEATURES:END -->"


def regenerate_docs(repo_root: Optional[Path] = None) -> None:
    """Regenerate ``docs/`` and patch the README. Idempotent."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    written: List[Tuple[str, str, int, str]] = []
    for slug, title, registry, description in _registries():
        content = _render_registry(slug, title, registry, description, repo_root)
        _write_if_changed(docs_dir / f"{slug}.md", content)
        written.append((slug, title, len(registry), description))

    _write_if_changed(docs_dir / "index.md", _render_index(written))
    _patch_readme(repo_root / "README.md", written)


def _render_registry(
    slug: str,
    title: str,
    registry: Dict[str, Any],
    description: str,
    repo_root: Path,
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
    for key in sorted(registry):
        value = registry[key]
        lines.extend(_render_entry(key, value, repo_root))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_entry(key: str, value: Any, repo_root: Path) -> List[str]:
    if not inspect.isclass(value):
        # Enum-style registry (e.g. SAMPLER_REGISTRY maps strings to strings).
        return [f"## `{key}`", "", f"Value: `{value!r}`"]

    qualname = getattr(value, "__qualname__", value.__name__)
    summary = _extract_summary(value)
    link = _source_link(value, repo_root)

    out = [f"## `{key}` - {qualname}", ""]
    if summary:
        out.extend([summary, ""])
    if link:
        out.append(f"Source: [{link.display}]({link.url})")
    else:
        out.append("Source: (unknown)")
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
    for slug, title, count, description in written:
        lines.append(f"- [{title}]({slug}.md) ({count}) - {description}")
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


def _patch_readme(readme_path: Path, written: List[Tuple[str, str, int, str]]) -> None:
    """Replace content between the AUTODOC markers in the README."""
    if not readme_path.exists():
        return
    current = readme_path.read_text()
    if _README_BEGIN not in current or _README_END not in current:
        return  # markers not present; user must add them manually

    block_lines = [_README_BEGIN, ""]
    block_lines.append(
        "Praxis is organized as ~20 pluggable registries. Each category below"
    )
    block_lines.append(
        "links to a docs page listing the concrete implementations and their"
    )
    block_lines.append("source. See [docs/index.md](docs/index.md) for the full map.")
    block_lines.append("")
    for slug, title, count, _description in written:
        block_lines.append(f"- [{title}](docs/{slug}.md) ({count})")
    block_lines.append("")
    block_lines.append(_README_END)
    block = "\n".join(block_lines)

    before, _, rest = current.partition(_README_BEGIN)
    _, _, after = rest.partition(_README_END)
    new = before + block + after
    _write_if_changed(readme_path, new)


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


def _extract_summary(cls: type) -> str:
    """First paragraph of the class docstring, falling back to the module docstring."""
    for source in (inspect.getdoc(cls), inspect.getdoc(inspect.getmodule(cls))):
        if not source:
            continue
        paragraph = source.split("\n\n", 1)[0].strip()
        if paragraph:
            return textwrap.fill(paragraph, width=88, replace_whitespace=True)
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
