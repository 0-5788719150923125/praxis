#!/usr/bin/env python3
"""Auto-docs generator for ghost.

The Praxis move (see praxis/docs.py) applied to a Godot project: introspect
the source of record and generate the documentation from it, so the docs
cannot drift from the code. Here the source of record is the GDScript
itself - every script carries a leading `##` doc comment, the scene roster
is the literal `Director.SCENES` array, the component registries are
literal `const REGISTRY := {...}` dictionaries, and the Masking effect
table is `MASK_EFFECTS` / `EFFECT_CONTROLS`. This script parses all of
that statically (no Godot boot required) and writes:

  docs/index.md       - the map: design, directory layout, every script.
  docs/scenes.md      - the scene catalogue, from each scene's own doc.
  docs/layers.md      - the Layer registry (visual components).
  docs/forces.md      - the Primitives registry (physics forces).
  docs/stage.md       - the storyboard stage: Cast actors + Actions verbs.
  docs/masking.md     - the Masking: effects, controls, CLI.
  docs/cli.md         - every ghost command-line flag.

It also patches README.md between `<!-- AUTODOC:NAME:BEGIN/END -->` marker
pairs (LAYOUT, SCENES), the same mechanism as the Praxis README.

Writes are idempotent: a file is only touched when its content changes.
Drift that a human must resolve (a script missing from the group map, a
CLI flag missing from the table, a scene on disk but not registered) is
printed as a warning and surfaced in the generated docs rather than
silently dropped.

Run from anywhere: ``python axis/ghost/docs.py`` (stdlib only).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
SCENES_DIR = SCRIPTS / "scenes"
DOCS = ROOT / "docs"

WARNINGS: List[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[ghost-docs] WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Curated maps. These carry the judgments a parser can't make: how scripts
# group into subsystems, and what each CLI flag means. Both are verified
# against the source at every run - a script or flag that exists in the
# code but not here is reported, so the map can rot loudly, never quietly.
# ---------------------------------------------------------------------------

# (group title, group description, [script filenames under scripts/]).
SCRIPT_GROUPS: List[Tuple[str, str, List[str]]] = [
    (
        "Audio & analysis",
        "From a song file to the typed per-frame `AudioFeatures` every scene "
        "consumes - live analyzer, offline bake, and the content descriptors "
        "that make sessions deterministic per song.",
        [
            "spectrum.gd",
            "audio_features.gd",
            "bake.gd",
            "bake_runner.gd",
            "harmonic_signature.gd",
            "echo.gd",
        ],
    ),
    (
        "Direction & session",
        "The lifecycle around the scenes: boot, splash, the Director's "
        "scheduling/transitions, the manual Workspace, and the Dial "
        "performance controls.",
        [
            "main.gd",
            "boot.gd",
            "chrome.gd",
            "splash.gd",
            "director.gd",
            "workspace.gd",
            "dial.gd",
            "dial_widget.gd",
            "volume_knob.gd",
        ],
    ),
    (
        "Scene substrate",
        "What every scene is built on: the `GhostScene` base, the camera, "
        "framing pools, seeded motion, sparse response gating, lighting, and "
        "the organic primitives (curves, flow fields, growth).",
        [
            "ghost_scene.gd",
            "scene_view.gd",
            "shots.gd",
            "mod_bank.gd",
            "activation.gd",
            "lighting.gd",
            "nonlinear.gd",
            "flow.gd",
            "filament.gd",
            "swarm.gd",
        ],
    ),
    (
        "Composition registries",
        "The two shared registries scenes compose by key - appearance "
        "(`Layer`) and physics (`Primitives`) - plus the particle substrate "
        "the forces act on.",
        [
            "layer.gd",
            "primitives.gd",
            "particle.gd",
            "particle_system.gd",
        ],
    ),
    (
        "3D path",
        "The unified software-3D renderer: a positionable perspective camera, "
        "mesh/plane primitives, procedural fields, palettes, terrain "
        "heightfields, and shadowing.",
        [
            "lens3d.gd",
            "plane3d.gd",
            "scene3d.gd",
            "mesh3d.gd",
            "geometry.gd",
            "field.gd",
            "palette.gd",
            "terrain.gd",
            "shadow_field.gd",
        ],
    ),
    (
        "Bodies",
        "Reusable composed characters - sampled stacks of primitives, not "
        "bespoke meshes.",
        ["eye_body.gd", "prism_body.gd"],
    ),
    (
        "Synthesis (voice)",
        "Text to narrated audio, no generative AI and no recordings: the "
        "phoneme front end, the source-filter synthesizer, and the synthesis "
        "editor with karaoke subtitles. Design and rungs: next/voice.md at "
        "the repo root.",
        [
            "phonemes.gd",
            "voice.gd",
            "voice_stream.gd",
            "synth_editor.gd",
            "subtitles.gd",
        ],
    ),
    (
        "Storyboards & stage",
        "Manual mode as data: the YAML-subset parser, the storyboard loader, "
        "and the Cast/Actions/Track stack that renders a described scene. "
        "See [storyboards/README.md](../storyboards/README.md) for the data "
        "spec and [stage.md](stage.md) for the actor/verb registries.",
        ["yaml.gd", "storyboard.gd", "cast.gd", "actions.gd", "track.gd"],
    ),
    (
        "Masking",
        "The video chroma-key masking editor - a second app surface inside "
        "ghost. See [masking.md](masking.md).",
        [
            "mask_session.gd",
            "mask_editor.gd",
            "mask_timeline.gd",
            "timeline_view.gd",
            "track_lane.gd",
            "mask_marker_tool.gd",
        ],
    ),
    (
        "Export",
        "Rendering a session to video (bake + Movie Maker, background " "processes).",
        ["exporter.gd"],
    ),
    (
        "Feedback & assistant",
        "The in-app authoring loop: capture reproducible critiques, browse "
        "them, and dispatch automated fixes.",
        ["feedback.gd", "assistant.gd"],
    ),
]

# (flag, argument placeholder, description, internal). Internal flags are
# passed between ghost's own processes (exporter -> render, editor ->
# render), not meant for hand use. Verified against the source scan below.
CLI_FLAGS: List[Tuple[str, str, str, bool]] = [
    (
        "--audio",
        "<path>",
        "Load this song (`.wav` / `.mp3` / `.ogg` / `.flac`; FLAC is "
        "transcoded via ffmpeg) and skip the splash.",
        False,
    ),
    (
        "--scene",
        "<name|N>",
        "Pin one scene for authoring (by script-name substring or registry " "index).",
        False,
    ),
    (
        "--storyboard",
        "<name>",
        "Manual mode: play `storyboards/<name>.yaml` (or `.json`).",
        False,
    ),
    ("--no-splash", "", "Boot straight to auto mode, bundled/no audio.", False),
    (
        "--seed",
        "<N>",
        "Override the session seed (default derives from the audio's own "
        "content fingerprint, so the same song replays the same show).",
        False,
    ),
    ("--dial-demo", "", "Auto-turn the first Dial hands-free (demos, renders).", False),
    (
        "--synth",
        "[text-file]",
        "Open the voice-synthesis editor: write or paste a script, sample a "
        "voice by seed, Speak renders a WAV take and plays it as a normal "
        "session (scenes react to the narration; karaoke subtitles track it).",
        False,
    ),
    (
        "--say",
        "",
        "With `--synth`: speak the loaded text immediately on boot "
        "(automation, demos, headless checks).",
        False,
    ),
    (
        "--mask-edit",
        "<session.json>",
        "Open the Masking editor on a session (also creates one from a " "video path).",
        False,
    ),
    (
        "--mask-render",
        "<session.json>",
        "Render a Masking session to video (used with `--write-movie`).",
        True,
    ),
    (
        "--export",
        "",
        "Marks a Movie Maker render process (set by the exporter; `Boot` "
        "shrinks the window early).",
        True,
    ),
    (
        "--synth-autopilot",
        "",
        "With `--export`: open the Synthesis panel over the take and let the "
        "fishing game play itself (random Throw/Pull/reel/hold-or-fold), so the "
        "UI is recorded into the video. Generates no audio and persists nothing "
        "(set by the exporter's 'Automate the Synthesis game' toggle).",
        True,
    ),
    (
        "--use-bake",
        "",
        "Drive `Spectrum` from the song's cached bake instead of the live " "analyzer.",
        True,
    ),
    (
        "--bake-file",
        "<path>",
        "Explicit spectrum-bake cache for a render (implies `--use-bake`).",
        True,
    ),
    ("--bake-song", "<path>", "`bake_runner`: the song to analyze.", True),
    ("--bake-out", "<path>", "`bake_runner`: where to write the bake cache.", True),
]

# Flags that belong to the Godot engine itself (or are argument separators),
# excluded from the drift check.
ENGINE_FLAGS = {
    "--headless",
    "--path",
    "--editor",
    "--quit",
    "--script",
    "--write-movie",
    "--fixed-fps",
    "--",
}

# Top-level entries for the README LAYOUT block and docs/index.md.
# (name, description). Directories that may not exist locally (runtime dirs)
# are annotated rather than skipped.
TOP_LEVEL: List[Tuple[str, str]] = [
    (
        "project.godot",
        "Godot 4.6 project; autoloads `Boot`, `Spectrum`, `Director`; "
        "`scenes/main.tscn` is the entry scene.",
    ),
    ("scenes/", "The Godot entry scene (`main.tscn`). Everything else is code-built."),
    (
        "scripts/",
        "All GDScript. Per-script map in [docs/index.md](docs/index.md); the "
        "subsystem groups are described there too.",
    ),
    (
        "scripts/scenes/",
        "The visualizer scene catalogue - one class per scene. See "
        "[docs/scenes.md](docs/scenes.md).",
    ),
    (
        "shaders/",
        "The two GPU surfaces: `flame.gdshader` (fire layer), `mask_split.gdshader` (all Masking effects).",
    ),
    (
        "storyboards/",
        "Manual-mode scene scores (YAML; JSON accepted). "
        "[storyboards/README.md](storyboards/README.md) is the data spec.",
    ),
    (
        "masks/",
        "Saved Masking sessions, one directory per source video (runtime, "
        "git-ignored).",
    ),
    ("tests/", "Headless check scripts (`godot --headless --script tests/<x>.gd`)."),
    ("reference/", "Reference imagery scenes were prototyped from."),
    (
        "docs/",
        "Generated documentation. Regenerate with `python docs.py`; do not edit by hand.",
    ),
    (
        "feedback/",
        "Feedback console output: `NNNN.json` + `NNNN.png` per report (runtime, git-ignored).",
    ),
    (
        "audio/",
        "Drop a `song.wav` here to bundle one (runtime, git-ignored); or use `--audio`.",
    ),
]

AUTOGEN_HEADER = "<!-- AUTOGENERATED by docs.py - do not edit by hand -->"


# ---------------------------------------------------------------------------
# GDScript parsing
# ---------------------------------------------------------------------------


class Script:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.stem
        self.text = path.read_text()
        self.class_name, self.extends, self.doc = _parse_header(self.text)

    @property
    def rel(self) -> str:
        return self.path.relative_to(ROOT).as_posix()


def _parse_header(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Extract `class_name`, `extends`, and the leading `##` doc block.

    The doc block is the first contiguous run of `##` lines before the first
    declaration (func/var/const/class/signal/enum). Returned with the `##`
    prefix stripped; blank doc lines preserved as paragraph breaks."""
    class_name = None
    extends = None
    doc: List[str] = []
    in_doc = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("##"):
            doc.append(s[2:].lstrip())
            in_doc = True
            continue
        if in_doc:
            break
        if s.startswith("extends ") and extends is None:
            extends = s.split()[1]
        elif s.startswith("class_name "):
            class_name = s.split()[1]
        elif s.startswith(("func ", "var ", "const ", "class ", "signal ", "enum ")):
            break
    return class_name, extends, doc


def _md(text: str) -> str:
    """GDScript doc markup -> markdown: `[param x]` -> `x`, `[Class]` -> `Class`."""
    text = re.sub(r"\[param (\w+)\]", r"`\1`", text)
    return re.sub(r"\[([A-Z]\w*(?:\.\w+)*)\]", r"`\1`", text)


def _paragraphs(doc: List[str]) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for line in doc:
        if line == "":
            if cur:
                out.append(" ".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        out.append(" ".join(cur))
    return out


def _one_liner(doc: List[str], strip_name: str = "") -> str:
    """First doc paragraph as a single line, optional `Name -` prefix stripped."""
    paras = _paragraphs(doc)
    if not paras:
        return ""
    line = paras[0]
    if strip_name:
        line = re.sub(
            rf"^{re.escape(strip_name)}\s+-\s+", "", line, flags=re.IGNORECASE
        )
    if len(line) > 300:
        line = line[:300].rsplit(" ", 1)[0] + " ..."
    return _md(line)


def _full_doc(doc: List[str]) -> str:
    return "\n\n".join(_md(p) for p in _paragraphs(doc))


def _line_of(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text, re.M)
    if not m:
        return None
    return text.count("\n", 0, m.start()) + 1


def _parse_registry_dict(
    text: str, const_name: str = "REGISTRY"
) -> List[Tuple[str, str]]:
    """Parse `const NAME := { "key": Value, ... }` into (key, value) pairs."""
    m = re.search(rf"const {const_name} :?= \{{(.*?)\n\}}", text, re.S)
    if not m:
        return []
    return re.findall(r'"(\w+)":\s*"?(\w+)"?', m.group(1))


def _inner_class_docs(text: str) -> Dict[str, Tuple[str, int]]:
    """Map inner class name -> (doc text, 1-based line). The doc is the
    contiguous comment block (`#` or `##`) directly above the declaration,
    with decorative ruler lines dropped."""
    out: Dict[str, Tuple[str, int]] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"class (\w+)", line)
        if not m:
            continue
        doc: List[str] = []
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("#"):
            stripped = lines[j].strip().lstrip("#").strip()
            if not re.fullmatch(r"-{3,}", stripped):
                doc.insert(0, stripped)
            j -= 1
        paras = _paragraphs(doc)
        out[m.group(1)] = ("\n\n".join(_md(p) for p in paras), i + 1)
    return out


def _source_link(rel: str, line: Optional[int] = None) -> str:
    """Markdown source link from docs/ to a repo file (with #L anchor)."""
    anchor = f"#L{line}" if line else ""
    label = f"{rel}:{line}" if line else rel
    return f"[{label}](../{rel}{anchor})"


# ---------------------------------------------------------------------------
# Scene catalogue
# ---------------------------------------------------------------------------


class SceneInfo:
    def __init__(self, script: Script) -> None:
        self.script = script
        t = script.text
        self.behaviors: List[str] = []
        self.group = ""
        kind = re.search(r'render_kind\s*=\s*"(\w+)"', t)
        self.render_kind = (
            kind.group(1)
            if kind
            else ("scene3d" if script.extends == "Scene3D" else "canvas")
        )
        self.oneshot = 'lifecycle = "oneshot"' in t
        self.morph_in = _first(t, r'morph_in\s*=\s*"(\w+)"')
        self.morph_out = _first(t, r'morph_out\s*=\s*"(\w+)"')
        self.layers = sorted(set(re.findall(r'add_layer\(\s*"(\w+)"', t)))


def _first(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _parse_scene_roster(director_text: str) -> List[Tuple[str, str, str]]:
    """Parse Director.SCENES into (scene name, behavior, group comment)."""
    m = re.search(r"const SCENES :?= \[(.*?)\n\]", director_text, re.S)
    if not m:
        warn("could not parse Director.SCENES")
        return []
    entries: List[Tuple[str, str, str]] = []
    group = ""
    for line in m.group(1).splitlines():
        s = line.strip()
        if s.startswith("#"):
            group = s.lstrip("# ").strip()
            continue
        pm = re.search(r'scenes/(\w+)\.gd"\).*?"behavior":\s*"(\w+)"', s)
        if pm:
            entries.append((pm.group(1), pm.group(2), group))
    return entries


def _collect_scenes() -> Tuple[Dict[str, SceneInfo], List[str]]:
    """All scenes on disk, annotated with their Director registration.
    Returns (name -> SceneInfo, ordered group labels)."""
    scenes = {p.stem: SceneInfo(Script(p)) for p in sorted(SCENES_DIR.glob("*.gd"))}
    roster = _parse_scene_roster((SCRIPTS / "director.gd").read_text())
    groups: List[str] = []
    for name, behavior, group in roster:
        if name not in scenes:
            warn(f"Director.SCENES registers scenes/{name}.gd which does not exist")
            continue
        scenes[name].behaviors.append(behavior)
        scenes[name].group = group
        if group not in groups:
            groups.append(group)
    return scenes, groups


def _split_group(group: str) -> Tuple[str, str]:
    """A roster group comment -> (short section header, optional intro line).
    The comment's lead (before ' - ' or a parenthetical) is the header; the
    full comment becomes the intro when it says more than the header."""
    if not group:
        return "Core catalogue", ""
    head = re.split(r" - |\(", group, maxsplit=1)[0]
    head = head.replace('"', "").strip().rstrip(".")
    head = head[0].upper() + head[1:]
    intro = _md(group) if group.rstrip(".") != head else ""
    return head, intro


def _scene_flags(info: SceneInfo) -> str:
    bits = [info.render_kind]
    if info.behaviors:
        bits.extend(sorted(set(info.behaviors)))
    if info.oneshot:
        bits.append("oneshot seeds")
    return ", ".join(bits)


def _render_scenes_doc(scenes: Dict[str, SceneInfo], groups: List[str]) -> str:
    lines = [
        AUTOGEN_HEADER,
        "# Scene catalogue",
        "",
        f"{len(scenes)} scenes under `scripts/scenes/`, one class per file, "
        "each documented by its own leading doc comment (reproduced here). "
        "A scene is a seeded **definition** (`build_params(rng)`) modulated "
        "by audio (`update`) and drawn through a view; registration in "
        "`Director.SCENES` pairs it with one or more **behaviors** "
        "(`static` / `drift` / `fluid`).",
        "",
        "Legend per entry: render kind, registered behaviors, lifecycle. "
        "`morph in/out` are the typed geometries a scene can continuously "
        "hand over across a cut; `layers` are the [Layer](layers.md) "
        "components it composes.",
        "",
    ]
    ordered = [g for g in groups]
    by_group: Dict[str, List[str]] = {g: [] for g in ordered}
    unregistered: List[str] = []
    for name in sorted(scenes):
        info = scenes[name]
        if not info.behaviors:
            unregistered.append(name)
        else:
            by_group.setdefault(info.group, []).append(name)

    for group in ordered:
        names = by_group.get(group, [])
        if not names:
            continue
        header, intro = _split_group(group)
        lines.extend([f"## {header}", ""])
        if intro:
            lines.extend([intro, ""])
        for name in names:
            lines.extend(_render_scene_entry(name, scenes[name]))
    if unregistered:
        lines.extend(
            [
                "## On disk but not in the auto rotation",
                "",
                "Present under `scripts/scenes/` but not registered in "
                "`Director.SCENES` - reachable only by storyboard (`stage`), "
                "`--scene` pin, or not reachable at all. If one of these "
                "should be in the rotation, register it; if it is retired, "
                "delete it.",
                "",
            ]
        )
        for name in unregistered:
            lines.extend(_render_scene_entry(name, scenes[name]))
    return "\n".join(lines).rstrip() + "\n"


def _render_scene_entry(name: str, info: SceneInfo) -> List[str]:
    s = info.script
    out = [f"### `{name}` ({_scene_flags(info)})", ""]
    body = _full_doc(s.doc)
    if body:
        out.extend([body, ""])
    else:
        warn(f"scenes/{name}.gd has no doc comment")
    facts = []
    if info.morph_in:
        facts.append(f"morph in: `{info.morph_in}`")
    if info.morph_out:
        facts.append(f"morph out: `{info.morph_out}`")
    if info.layers:
        facts.append("layers: " + ", ".join(f"`{k}`" for k in info.layers))
    if facts:
        out.extend([" · ".join(facts), ""])
    out.extend([f"Source: {_source_link(s.rel)} (extends `{s.extends}`)", ""])
    return out


def _render_readme_scenes_block(scenes: Dict[str, SceneInfo], groups: List[str]) -> str:
    lines = [
        f"{sum(1 for s in scenes.values() if s.behaviors)} scenes in the auto "
        f"rotation ({len(scenes)} on disk). One line each - the full "
        "catalogue, with every scene's own documentation, is "
        "[docs/scenes.md](docs/scenes.md).",
        "",
    ]
    for group in groups:
        names = [
            n
            for n in sorted(scenes)
            if scenes[n].behaviors and scenes[n].group == group
        ]
        if not names:
            continue
        lines.extend([f"_{_split_group(group)[0]}_", ""])
        for name in names:
            info = scenes[name]
            desc = _one_liner(info.script.doc, strip_name=name.replace("_", " "))
            desc = desc or "(undocumented)"
            lines.append(f"- **`{name}`** ({_scene_flags(info)}) - {desc}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Registry pages (layers, forces, stage)
# ---------------------------------------------------------------------------


def _render_registry_page(
    title: str,
    slug_intro: str,
    script: Script,
    pairs: List[Tuple[str, str]],
) -> str:
    inner = _inner_class_docs(script.text)
    lines = [
        AUTOGEN_HEADER,
        f"# {title}",
        "",
        slug_intro,
        "",
        _full_doc(script.doc),
        "",
        f"Registry: `{script.class_name}.REGISTRY` in {_source_link(script.rel)} "
        f"({len(pairs)} entries)",
        "",
    ]
    for key, cls in pairs:
        doc, line = inner.get(cls, ("", None))
        lines.append(f"## `{key}` - {cls}")
        lines.append("")
        if doc:
            lines.extend([doc, ""])
        else:
            warn(f"{script.rel}: class {cls} (registry key '{key}') has no doc comment")
        lines.append(f"Source: {_source_link(script.rel, line)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_stage_doc(cast: Script, actions: Script, track: Script) -> str:
    action_pairs = _parse_registry_dict(actions.text)
    cast_pairs = _parse_registry_dict(cast.text)
    cast_inner = _inner_class_docs(cast.text)
    action_inner = _inner_class_docs(actions.text)
    lines = [
        AUTOGEN_HEADER,
        "# Stage: actors and verbs",
        "",
        "The data-driven scene stack behind storyboard `stage` entries: a "
        "**Cast** of actors, **Actions** verbs applied to them on a "
        "**Track** timeline. The storyboard file format itself is specified "
        "in [storyboards/README.md](../storyboards/README.md).",
        "",
        "## Cast (actor registry)",
        "",
        _full_doc(cast.doc),
        "",
        f"Registry: `Cast.REGISTRY` in {_source_link(cast.rel)} "
        f"({len(cast_pairs)} kinds)",
        "",
    ]
    for kind, _ in cast_pairs:
        cls = f"{kind.capitalize()}Actor"
        doc, line = cast_inner.get(cls, ("", None))
        lines.append(f"### `{kind}` - {cls}")
        lines.append("")
        if doc:
            lines.extend([doc, ""])
        lines.extend([f"Source: {_source_link(cast.rel, line)}", ""])
    lines.extend(
        [
            "## Actions (verb registry)",
            "",
            _full_doc(actions.doc),
            "",
            f"Registry: `Actions.REGISTRY` in {_source_link(actions.rel)} "
            f"({len(action_pairs)} verbs)",
            "",
        ]
    )
    for key, cls in action_pairs:
        doc, line = action_inner.get(cls, ("", None))
        lines.append(f"### `{key}` - {cls}")
        lines.append("")
        if doc:
            lines.extend([doc, ""])
        lines.extend([f"Source: {_source_link(actions.rel, line)}", ""])
    lines.extend(
        [
            "## Track (the timeline runner)",
            "",
            _full_doc(track.doc),
            "",
            f"Source: {_source_link(track.rel)}",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Masking page
# ---------------------------------------------------------------------------


def _render_masklab_doc(session: Script, editor: Script) -> str:
    t = session.text
    effects_m = re.search(r"const MASK_EFFECTS :?= \[(.*?)\]", t, re.S)
    effects = re.findall(r'"(\w+)"', effects_m.group(1)) if effects_m else []
    controls: Dict[int, Tuple[List[str], str]] = {}
    cm = re.search(r"const EFFECT_CONTROLS :?= \{(.*?)\n\}", t, re.S)
    if cm:
        for line in cm.group(1).splitlines():
            lm = re.match(r"\s*(\d+):\s*\[([^\]]*)\],?\s*(?:#\s*(.*))?", line)
            if not lm:
                continue
            groups = re.findall(r'"(\w+)"', lm.group(2))
            note = (lm.group(3) or "").strip()
            controls[int(lm.group(1))] = (groups, note)
    lines = [
        AUTOGEN_HEADER,
        "# Masking",
        "",
        "The video chroma-key masking editor - a second app surface inside "
        "ghost, separate from the audio visualizer. Open it on a session (or "
        "straight on a video file) with:",
        "",
        "```",
        "godot --path axis/ghost -- --mask-edit masks/<video>/session.json",
        "```",
        "",
        "## The data model",
        "",
        _full_doc(session.doc),
        "",
        f"Source: {_source_link(session.rel)}",
        "",
        "## The editor",
        "",
        _full_doc(editor.doc),
        "",
        f"Source: {_source_link(editor.rel)}",
        "",
        "## Effects",
        "",
        f"{len(effects)} effects (`MaskSession.MASK_EFFECTS`). The actual "
        "implementations live in "
        "[shaders/mask_split.gdshader](../shaders/mask_split.gdshader) - "
        "each marker becomes a shader layer, and `apply_layer()` dispatches "
        "per effect. Control groups: `keying` (threshold / feather / "
        "colorfulness steer the volumetric mask), `reach` (how wide around "
        "the key colour a restore acts), `pattern` (field placement / "
        "coverage / contrast / resonance), plus per-effect groups (`echo`, "
        "`snow`, `fur`). An effect with no groups exposes only the universal "
        "colour + intensity controls.",
        "",
        "| # | Effect | Control groups | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for idx, name in enumerate(effects):
        groups, note = controls.get(idx, ([], ""))
        note = re.sub(rf"^{re.escape(name)}\s*", "", note)
        note = note.lstrip("(").rstrip(")")
        gtxt = ", ".join(f"`{g}`" for g in groups) if groups else "-"
        lines.append(f"| {idx} | `{name}` | {gtxt} | {note} |")
    lines.extend(
        [
            "",
            "## Headless marker insertion",
            "",
            "`scripts/mask_marker_tool.gd` inserts one marker into a saved "
            "session from the command line (no editor boot): ",
            "",
            "```",
            "godot --headless --path axis/ghost --script scripts/mask_marker_tool.gd \\",
            "    -- masks/<video>/session.json <time> [field=value ...]",
            "```",
            "",
            "The marker seeds from whatever was governing at `<time>`; each "
            "`field=value` then overrides one session field (e.g. "
            "`effect_a=14 fx_contrast=0.6 duration=0.5`). Note: a live "
            "editor autosaving the same session will clobber markers "
            "inserted this way - reload the session first.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI page + flag drift check
# ---------------------------------------------------------------------------


def _scan_flags() -> Dict[str, List[str]]:
    """Every `--flag` string literal in the GDScript, mapped to the scripts
    that mention it."""
    found: Dict[str, List[str]] = {}
    for path in sorted(SCRIPTS.rglob("*.gd")):
        for flag in set(re.findall(r'"(--[a-z][a-z-]*)"', path.read_text())):
            found.setdefault(flag, []).append(path.relative_to(ROOT).as_posix())
    return found


def _render_cli_doc(found: Dict[str, List[str]]) -> str:
    documented = {f for f, _, _, _ in CLI_FLAGS}
    undocumented = sorted(set(found) - documented - ENGINE_FLAGS)
    for flag in undocumented:
        warn(
            f"CLI flag {flag} ({', '.join(found[flag])}) missing from CLI_FLAGS in docs.py"
        )
    for flag in sorted(documented - set(found)):
        warn(f"CLI_FLAGS documents {flag} but it no longer appears in the source")
    lines = [
        AUTOGEN_HEADER,
        "# CLI flags",
        "",
        "Ghost's own flags follow the Godot separator: "
        "`godot --path axis/ghost -- <ghost flags>`. Any of "
        "`--audio` / `--scene` / `--storyboard` / `--no-splash` boots "
        "straight past the splash. Flags marked _internal_ are passed "
        "between ghost's own processes (exporter, bake runner, mask render); "
        "you rarely type them.",
        "",
        "| Flag | Argument | Description | |",
        "| --- | --- | --- | --- |",
    ]
    for flag, arg, desc, internal in CLI_FLAGS:
        if flag not in found:
            continue
        tag = "_internal_" if internal else ""
        cells = [f"`{flag}`", f"`{arg}`" if arg else "", desc, tag]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    if undocumented:
        lines.extend(
            [
                "",
                "**Undocumented flags found in the source** (add them to "
                "`CLI_FLAGS` in `docs.py`): "
                + ", ".join(f"`{f}`" for f in undocumented),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Index + README layout
# ---------------------------------------------------------------------------

DESIGN_NOTES = """\
## Design, in five commitments

Ghost is not an entity-component system and does not pretend to be one; it
borrows the part of that idea it actually needs - **composition over
inheritance, by key, from registries** - and pairs it with a functional
core. The commitments:

1. **Deterministic functions of (seed, audio, time).** A scene's definition
   is a pure roll of a seeded RNG (`build_params`); its per-frame state is a
   function of the typed `AudioFeatures` stream. The session seed derives
   from the audio's own content fingerprint, so the same song always plays
   the same show. Determinism is what makes the feedback console's records
   reproducible and the offline export byte-stable.
2. **Declarative where a human authors.** Storyboards describe scenes as
   data (cast + verbs on a timeline, every number a sampleable range);
   Masking effects are table-driven (`MASK_EFFECTS` / `EFFECT_CONTROLS`);
   the scene roster, forces, and visual layers are literal registries.
   Adding to a registry is the extension mechanism - not new control flow.
3. **Sampled, not baked ("cattle, not pets").** Every tunable constant is a
   candidate for sampling from a per-instance range, so two things of a
   kind always differ and the catalogue gains expression for free.
4. **Convergence over lockstep.** Presentation state never snaps to its
   target: cameras, activations, glows, and the Echo cursor all move by
   exponential smoothing toward targets that may jump discontinuously.
   Work that misses a frame is absorbed, not queued - the picture is
   allowed to be briefly stale and converges when the signal (or the frame
   budget) allows.
5. **Registries make integration free.** The same `snow` layer that is a
   scene on its own falls over the cityscape; the same `scatter` force
   bursts glass, rocks, and embers. A new scene is mostly a parts list.
"""


def _render_index(
    scripts: Dict[str, Script],
    scenes: Dict[str, SceneInfo],
    pages: List[Tuple[str, str, str]],
) -> str:
    lines = [
        AUTOGEN_HEADER,
        "# ghost docs index",
        "",
        "Generated from the source of record (doc comments, registries, the "
        "scene roster) by [docs.py](../docs.py) - see the "
        "[README](../README.md) for the project's philosophy and workflow. "
        "Regenerate with `python docs.py`.",
        "",
        DESIGN_NOTES,
        "## Pages",
        "",
    ]
    for slug, title, desc in pages:
        lines.append(f"- [{title}]({slug}.md) - {desc}")
    lines.extend(["", "## Directory layout", ""])
    for name, desc in TOP_LEVEL:
        lines.append(f"- `{name}` - {_reprefix(desc)}")
    lines.extend(["", "## Script map", ""])
    grouped = {f for _, _, files in SCRIPT_GROUPS for f in files}
    for title, desc, files in SCRIPT_GROUPS:
        lines.extend([f"### {title}", "", desc, ""])
        for fname in files:
            script = scripts.get(Path(fname).stem)
            if script is None:
                warn(f"docs.py SCRIPT_GROUPS lists {fname} which does not exist")
                continue
            one = _one_liner(script.doc, strip_name=script.class_name or "")
            label = script.class_name or script.name
            lines.append(f"- [`{fname}`](../scripts/{fname}) **{label}** - {one}")
        lines.append("")
    stray = sorted(p.name for p in SCRIPTS.glob("*.gd") if p.name not in grouped)
    if stray:
        warn("scripts not in docs.py SCRIPT_GROUPS: " + ", ".join(stray))
        lines.extend(
            [
                "### Unsorted (add to `SCRIPT_GROUPS` in docs.py)",
                "",
            ]
        )
        for fname in stray:
            script = scripts.get(Path(fname).stem)
            one = _one_liner(script.doc) if script else ""
            lines.append(f"- [`{fname}`](../scripts/{fname}) - {one}")
        lines.append("")
    lines.append(
        f"Scene scripts ({len(scenes)}) are catalogued separately in "
        "[scenes.md](scenes.md)."
    )
    return "\n".join(lines).rstrip() + "\n"


def _reprefix(desc: str) -> str:
    """Rebase repo-root-relative markdown links for use inside docs/."""
    return re.sub(
        r"\]\((?!https?:|/|#|\.\.?/)([^)]+)\)", lambda m: f"](../{m.group(1)})", desc
    )


def _render_readme_layout_block() -> str:
    lines = [
        "Top-level layout; the per-script map (every class, one line each) "
        "is [docs/index.md](docs/index.md).",
        "",
    ]
    for name, desc in TOP_LEVEL:
        lines.append(f"- `{name}` - {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[ghost-docs] wrote {path.relative_to(ROOT)}")


def _patch_readme_block(name: str, body: str) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    current = readme.read_text()
    begin = f"<!-- AUTODOC:{name}:BEGIN -->"
    end = f"<!-- AUTODOC:{name}:END -->"
    if begin not in current or end not in current:
        warn(f"README.md missing AUTODOC:{name} markers - block not patched")
        return
    block = f"{begin}\n\n{body.rstrip()}\n\n{end}"
    before, _, rest = current.partition(begin)
    _, _, after = rest.partition(end)
    _write_if_changed(readme, before + block + after)


def main() -> int:
    scripts = {p.stem: Script(p) for p in sorted(SCRIPTS.glob("*.gd"))}
    scenes, groups = _collect_scenes()

    layer = scripts["layer"]
    prims = scripts["primitives"]
    pages = [
        (
            "scenes",
            "Scene catalogue",
            "every visualizer scene, from its own doc comment.",
        ),
        (
            "layers",
            "Layers",
            "the visual-component registry (weather, skies, atmosphere).",
        ),
        ("forces", "Forces", "the physics-primitive registry particles compose."),
        (
            "stage",
            "Stage",
            "storyboard actors (Cast) and verbs (Actions) + the Track runner.",
        ),
        (
            "masklab",
            "Masking",
            "the video chroma-key editor: model, effects, headless tools.",
        ),
        ("cli", "CLI flags", "every ghost command-line flag."),
    ]

    _write_if_changed(DOCS / "scenes.md", _render_scenes_doc(scenes, groups))
    _write_if_changed(
        DOCS / "layers.md",
        _render_registry_page(
            "Layers: the visual-component registry",
            "Reusable appearance components any scene composes by key via "
            "`add_layer` / `update_layers` / `draw_layers(z)`.",
            layer,
            _parse_registry_dict(layer.text),
        ),
    )
    _write_if_changed(
        DOCS / "forces.md",
        _render_registry_page(
            "Forces: the physics-primitive registry",
            "Reusable force modules a scene composes into a `ParticleSystem` "
            "by key.",
            prims,
            _parse_registry_dict(prims.text),
        ),
    )
    _write_if_changed(
        DOCS / "stage.md",
        _render_stage_doc(scripts["cast"], scripts["actions"], scripts["track"]),
    )
    _write_if_changed(
        DOCS / "masking.md",
        _render_masklab_doc(scripts["mask_session"], scripts["mask_editor"]),
    )
    _write_if_changed(DOCS / "cli.md", _render_cli_doc(_scan_flags()))
    _write_if_changed(DOCS / "index.md", _render_index(scripts, scenes, pages))

    _patch_readme_block("LAYOUT", _render_readme_layout_block())
    _patch_readme_block("SCENES", _render_readme_scenes_block(scenes, groups))

    print(
        f"[ghost-docs] done: {len(scenes)} scenes, "
        f"{len(scripts)} scripts, {len(WARNINGS)} warning(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
