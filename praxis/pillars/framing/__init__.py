"""Loader for the paper's conditionally-rendered sections.

Each fragment is one YAML file in this package's ``framing/`` directory (data
only - id is the filename stem); all logic lives here. A fragment declares a
``requires`` predicate over
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

import glob
import json
import os
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from praxis.cli.loaders.experiments import load_rendered_config
from praxis.pillars.proofs import PROOFS, render_proof
from praxis.pillars.runs import experiment_name, experiment_stems

# Fragment definitions live in this package's own directory, beside this file.
FRAMING_DIR = Path(__file__).parent
# This file is now praxis/pillars/framing/__init__.py, so repo root is parents[3].
REPO_ROOT = Path(__file__).parents[3]
RUNS_DIR = REPO_ROOT / "build" / "runs"
EXP_DIR = REPO_ROOT / "experiments"
OUT_PATH = REPO_ROOT / "research" / "framing.tex"


@dataclass(frozen=True)
class Fragment:
    """One gated chunk of paper prose, loaded from a framing/*.yml file."""

    id: str
    section: str
    body: str
    requires: Dict[str, List[str]] = field(default_factory=dict)
    order: int = 100
    proofs: List[str] = field(default_factory=list)

    def matches(self, config: Dict) -> bool:
        """True if every required key in ``config`` hits an accepted value."""
        for key, accepted in self.requires.items():
            value = config.get(key)
            if value is None:
                return False
            if not any(_match_one(value, pat) for pat in accepted):
                return False
        return True


# Comparison prefixes for numeric gating (e.g. requires: {recurrent_steps: [">1"]}).
# Longest operators first so ">=" is not read as ">".
_CMP = [
    (">=", lambda a, b: a >= b),
    ("<=", lambda a, b: a <= b),
    ("==", lambda a, b: a == b),
    (">", lambda a, b: a > b),
    ("<", lambda a, b: a < b),
]


def _match_one(value, pat) -> bool:
    """A config value matches a pattern by numeric comparison (``>1``, ``<=4``)
    when the pattern carries an operator, else by ``fnmatch`` on its string."""
    pat = str(pat)
    for op_str, op in _CMP:
        if pat.startswith(op_str):
            try:
                return op(float(value), float(pat[len(op_str) :]))
            except (TypeError, ValueError):
                return False
    return fnmatch(str(value), pat)


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
        proofs=list(raw.get("proofs") or []),
    )


def _to_tex_paragraphs(body: str) -> str:
    """Bodies are written as YAML folded scalars (``>``): authors wrap lines
    freely and folding collapses each block to one line. TeX needs a blank line
    between paragraphs, so promote every newline run to a paragraph break."""
    return re.sub(r"\n+", "\n\n", body.strip())


def load_framings() -> Dict[str, Fragment]:
    """All fragments from ``framing/``, keyed by id (filename stem)."""
    return {p.stem: _load_fragment(p) for p in sorted(FRAMING_DIR.glob("*.yml"))}


FRAMING: Dict[str, Fragment] = load_framings()


def active_fragments(config: Dict) -> List[Fragment]:
    """Fragments whose ``requires`` are satisfied by ``config``, sorted for
    rendering (by section, then ``order``, then ``id`` for stability)."""
    active = [f for f in FRAMING.values() if f.matches(config)]
    return sorted(active, key=lambda f: (f.section, f.order, f.id))


# ─── Rendering (which fragments the paper sees, for a given run) ──────────────


def newest_experiment() -> Optional[str]:
    """Experiment name of the most recently created run, or ``None``."""
    stems = experiment_stems()
    best_cmd, best_key = "", ""
    for cfg_path in glob.glob(str(RUNS_DIR / "*" / "config.json")):
        try:
            cfg = json.load(open(cfg_path))
        except (OSError, ValueError):
            continue
        key = cfg.get("created") or str(os.path.getmtime(cfg_path))
        if key > best_key:
            best_cmd, best_key = cfg.get("command", ""), key
    name = experiment_name(best_cmd, stems) if best_cmd else None
    return name or None


def resolve_config(experiment: str) -> Dict:
    """Flat config dict for an experiment (extends chain resolved), augmented
    with derived keys that fragments can gate on."""
    path = EXP_DIR / f"{experiment}.yml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment '{experiment}' at {path}")
    return _augment(load_rendered_config(path, experiments_dir=EXP_DIR))


def _augment(config: Dict) -> Dict:
    """Add config-derived gate keys the raw YAML doesn't carry, so a fragment can
    gate on a semantic ("is this run doing X?") rather than one literal value.
    Several of these are deliberately OR'd across keys, because one capability
    shows up under more than one config (the matcher itself only ORs within a
    single key)."""
    derived: Dict = {}

    # recurrent_steps: how many times the same physical layers are revisited. The
    # decoder loops ``depth`` times over ``num_layers`` experts, so depth>num_layers
    # means weights are reused across depth (e.g. calm-d's 2 layers x 4 steps).
    depth, layers = config.get("depth"), config.get("num_layers")
    if isinstance(depth, int) and isinstance(layers, int) and 0 < layers < depth:
        derived["recurrent"] = True
        derived["recurrent_steps"] = depth // layers
    else:
        derived["recurrent"] = False

    # Mono-Forward (detached, layer-wise local objectives) appears in two places:
    # the dedicated trainer (in-process or the Ray pipeline) and the remote-expert
    # swarm, whose layer-wise updates are Mono-Forward by construction.
    trainer = str(config.get("trainer_type", ""))
    derived["uses_mono_forward"] = (
        trainer.startswith("mono_forward")
        or config.get("orchestration_type") == "swarm"
    )

    # Harmonic latent space = a harmonic/crystal-bearing head OR the CALM codec.
    head = str(config.get("head_type", ""))
    encoder = str(config.get("encoder_type", ""))
    derived["uses_harmonic_latent"] = head in (
        "harmonic",
        "crystal",
        "crystal_harmonic",
        "prismatic",
    ) or encoder.startswith("calm")

    # codec_mode: which input representation the harmonic section opens on -
    # a CALM continuous-latent codec, a byte-latent (BLT) patching bottleneck,
    # or plain (sub)word tokenization with no codec. Exactly one, so the three
    # section-3.1 fragments are mutually exclusive.
    if encoder.startswith("calm"):
        derived["codec_mode"] = "calm"
    elif encoder.startswith("byte"):
        derived["codec_mode"] = "byte_latent"
    else:
        derived["codec_mode"] = "standard"

    return {**config, **derived}


def _section_macro(section: str) -> str:
    """Section anchor -> macro name, e.g. 'harmonic' -> 'paperFramingHarmonic'."""
    camel = "".join(p.capitalize() for p in section.replace("_", "-").split("-"))
    return f"paperFraming{camel}"


def render(config: Dict) -> Tuple[str, Dict]:
    """The framing.tex body and a summary, for one run's resolved config."""
    active = active_fragments(config)
    sections: Dict[str, List[str]] = {}
    rendered_proofs: set = set()
    for frag in active:
        block = frag.body
        # Inline proofs: render each cited proof once, beside its first active
        # framing ("you get what you give" - the live framing pulls in its proof).
        for pkey in frag.proofs:
            if pkey in rendered_proofs:
                continue
            proof = PROOFS.get(pkey)
            if proof is not None:
                block += "\n\n" + render_proof(proof)
                rendered_proofs.add(pkey)
        sections.setdefault(frag.section, []).append(block)

    lines = ["% Generated by praxis/pillars/framing - do not edit by hand."]
    for section, blocks in sorted(sections.items()):
        bodies = "\n\n".join(blocks)
        lines.append(f"\\newcommand{{\\{_section_macro(section)}}}{{%\n{bodies}\n}}")

    section_ids: Dict[str, List[str]] = {}
    for frag in active:
        section_ids.setdefault(frag.section, []).append(frag.id)
    summary = {
        "active": [f.id for f in active],
        "skipped": [fid for fid in FRAMING if fid not in {f.id for f in active}],
        "sections": section_ids,
        "proofs": sorted(rendered_proofs),
    }
    return "\n".join(lines) + "\n", summary


def export_framing(experiment: Optional[str] = None) -> Dict:
    """Resolve the run's active fragments and write research/framing.tex.
    ``experiment`` defaults to the newest run's. Returns a summary dict."""
    if experiment is None:
        experiment = newest_experiment()
    if not experiment:
        raise ValueError("could not determine an experiment (none found)")
    body, summary = render(resolve_config(experiment))
    OUT_PATH.write_text(body)
    summary["experiment"] = experiment
    return summary
