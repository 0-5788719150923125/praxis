"""The documented config format: commented YAML, written in two places.

The web app's Download button renders a parsed experiment (``extends`` already
merged) back into YAML with the documentation each key carries: the flag's
``help``, its long form (its own ``doc``, or the doc of the registry namespace it
selects from), and what the chosen value is. The same renderer rewrites the
experiment files this repo commits, so the public examples stay documented in one
voice; the gitignored experiments are a user's own and are never touched.

Every entry reads the same way - what the flag is, what it accepts and defaults
to, what the chosen value means, any note written by hand, a blank line, then the
key - so there is one comment shape to read rather than four. Labelled
paragraphs (``Options:``, ``Default:``, a value's name, ``Note:``) are the only
indented thing in the file, and they all hang at the same depth.
"""

import argparse
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from praxis.cli.core.parser import resolve_namespace
from praxis.registry import Namespace, namespaces

WIDTH = 80
HANG = 4  # continuation indent under a `Label: ` paragraph
GAP = 2  # blank lines between entries; one blank sits inside each entry

# A formatted file says so in its header. Re-formatting one keeps only the
# `Note:` paragraphs, because everything else in it was generated from the
# parser and would otherwise accrete a stale copy of itself.
MARKER = "Praxis formats this file"
NOTE = "Note"

# Registries big enough that listing every key buries the entry. Their docs page
# lists them with descriptions, which is the better place to look anyway.
OPTIONS_LIMIT = 12

# Groups the experiment and environment loaders fill with one toggle per YAML
# file. Those flags activate a config; they are never keys inside one.
_TOGGLE_GROUPS = ("experiments", "environments")

EXTENDS_KEY = "extends"
EXTENDS_DOC = (
    "Experiments this one inherits from, merged left to right; every key below "
    "overrides them. Each name is a file stem in experiments/."
)


def reference_parser() -> argparse.ArgumentParser:
    """The parser this process launched with, integration flags included; a
    freshly built static parser when the CLI never ran (tests, tools)."""
    import praxis.cli as cli
    from praxis.cli.groups import build_static_parser

    return cli.parser if cli.parser is not None else build_static_parser()


def documentation_parser() -> argparse.ArgumentParser:
    """The parser the experiment files are documented against: every static
    group, plus the flags each discovered integration registers.

    Not the parser a run launched with. Integrations are loaded but never
    bootstrapped, so this installs nothing, and it does not depend on which
    integration conditions this particular argv happened to satisfy - which is
    what keeps the committed files from churning between launches."""
    from praxis.cli.groups import build_static_parser
    from praxis.cli.loaders import IntegrationBridge

    parser = build_static_parser()
    bridge = IntegrationBridge()
    bridge.integrations = bridge.loader.discover_integrations()
    for manifest in bridge.integrations:
        bridge.loader.load_integration(manifest, verbose=False)
    bridge.add_cli_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# the two renderers
# ---------------------------------------------------------------------------


def render_annotated_config(
    config: Dict[str, Any], parser: argparse.ArgumentParser, name: str
) -> str:
    """The experiment ``config`` as commented YAML, with every remaining flag
    listed at its default underneath. ``name`` is the experiment stem, used for
    the launch instructions in the header.

    ``yaml.safe_load`` of the result equals ``config``: every annotation is a
    comment, and the reference section is commented out."""
    return _render(config, parser, name, reference=True)


def render_experiment_file(
    path, parser: argparse.ArgumentParser
) -> Optional[Tuple[str, str]]:
    """``(original, formatted)`` text for the experiment at ``path``.

    The file's own keys only, so inheritance survives the rewrite: ``extends`` is
    kept as a key rather than resolved into the bases it names. Every other flag
    follows at the bottom commented out at its default, exactly as in the
    download - these files are the examples, and an example that lists what else
    is available is worth the length.

    None when :func:`undocumentable_keys` finds anything: a key nothing in this
    checkout can describe is either dead or comes from an integration that is not
    loaded, and in both cases the rewrite would depend on how the process was
    started rather than on the file. ``tools/format_experiments.py --check`` names
    the keys."""
    path = Path(path)
    original = path.read_text()
    config = yaml.safe_load(original) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")

    from praxis.migrations import rename_legacy_config

    rename_legacy_config(config)
    if undocumentable_keys(config, parser):
        return None
    file_note, notes = _source_notes(original)

    text = _render(
        config,
        parser,
        path.stem,
        reference=True,
        on_disk=True,
        notes=notes,
        file_note=file_note,
    )
    return original, text


def undocumentable_keys(
    config: Dict[str, Any], parser: argparse.ArgumentParser
) -> List[str]:
    """Keys nothing can say anything about: no flag of that name, and no registry
    namespace of that name either.

    A key with no flag is legitimate - ``embeddings`` is read straight from the
    config by the code that uses it - as long as its namespace documents it. One
    with neither is a key nothing reads, or a flag of an integration this process
    did not load."""
    known = {a.dest for _, actions in _flag_groups(parser) for a in actions}
    return [
        key
        for key in config
        if key != EXTENDS_KEY
        and key.replace("-", "_") not in known
        and _registry_named(key) is None
    ]


def format_tracked_experiments(
    parser: argparse.ArgumentParser, experiments_dir="experiments"
) -> List[Path]:
    """Rewrite every git-tracked experiment in the documented format; return the
    ones that changed.

    Tracked means committed to this repository - the reference experiments the
    project publishes. Everything else in the directory is gitignored, belongs to
    whoever wrote it, and is never read or touched here."""
    changed: List[Path] = []
    for path in tracked_experiments(experiments_dir):
        try:
            rendered = render_experiment_file(path, parser)
        except Exception as error:  # a malformed example must not stop a launch
            print(f"Warning: could not format {path}: {error}")
            continue
        if rendered is None:
            continue
        original, text = rendered
        if text != original:
            path.write_text(text)
            changed.append(path)
    return changed


def tracked_experiments(experiments_dir="experiments") -> List[Path]:
    """The experiment files git tracks. Empty when git is unavailable or the
    directory is not in a repository, so nothing is ever rewritten on a guess."""
    directory = Path(experiments_dir)
    if not directory.is_dir():
        return []
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.yml"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    names = [name for name in result.stdout.split("\0") if name]
    return sorted(directory / name for name in names if (directory / name).is_file())


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def _render(
    config: Dict[str, Any],
    parser: argparse.ArgumentParser,
    name: str,
    *,
    reference: bool,
    on_disk: bool = False,
    notes: Optional[Dict[str, str]] = None,
    file_note: Optional[str] = None,
) -> str:
    notes = notes or {}
    groups = _flag_groups(parser)
    by_dest = {a.dest: (title, a) for title, actions in groups for a in actions}

    used: Dict[str, List[Tuple[str, argparse.Action]]] = {}
    unmatched: List[str] = []
    for key in config:
        if key == EXTENDS_KEY:
            continue
        hit = by_dest.get(key.replace("-", "_"))
        if hit is None:
            unmatched.append(key)
        else:
            used.setdefault(hit[0], []).append((key, hit[1]))
    used_dests = {a.dest for pairs in used.values() for _, a in pairs}

    out = _header(name, reference=reference, on_disk=on_disk, note=file_note)
    if reference:
        out += _banner("Set by this experiment")
    if EXTENDS_KEY in config:
        out += _entry(
            EXTENDS_KEY,
            config[EXTENDS_KEY],
            description=EXTENDS_DOC,
            note=notes.get(EXTENDS_KEY),
        )
    for title, actions in groups:
        order = {id(a): i for i, a in enumerate(actions)}
        pairs = sorted(used.get(title, []), key=lambda p: order[id(p[1])])
        if not pairs:
            continue
        out += _section(title)
        for key, action in pairs:
            out += _entry(
                key,
                config[key],
                description=_description(action),
                facts=_facts(action, show_default=True),
                values=_value_notes(config[key], getattr(action, "registry", None)),
                note=notes.get(key),
            )
    if unmatched:
        out += _section("keys without a flag")
        out += _comment(
            "Read from the config by the code that uses them, or flags of an "
            "integration this checkout does not load."
        )
        out += [""] * GAP
        for key in unmatched:
            registry = _registry_named(key)
            out += _entry(
                key,
                config[key],
                description=_sentence(registry.title) if registry else None,
                values=_value_notes(config[key], registry),
                note=notes.get(key),
            )
    if reference:
        out += _banner("Everything else, at its default")
        out += _comment(
            "Uncomment a line to set it; the value shown is the default. Flags "
            "kept out of the run hash (logging, serving) are not listed."
        )
        out += [""] * GAP
        for title, actions in groups:
            rest = [
                a
                for a in actions
                if a.dest not in used_dests
                and not getattr(a, "exclude_hash", False)
                and a.default is not argparse.SUPPRESS
            ]
            if not rest:
                continue
            out += _section(title)
            for action in rest:
                volatile = getattr(action, "volatile_default", False)
                out += _entry(
                    action.dest,
                    _placeholder(action) if volatile else action.default,
                    description=_description(action),
                    facts=_facts(action, show_default=volatile),
                    commented=True,
                    literal=volatile,
                )
    return "\n".join(out).rstrip() + "\n"


def _header(
    name: str, *, reference: bool, on_disk: bool, note: Optional[str] = None
) -> List[str]:
    lines = [f"# {name} - a Praxis experiment", "#"]
    if not on_disk:
        lines += _comment(f"Save as experiments/{name}.yml, then run:")
        lines += ["#"]
    lines += [f"#     ./launch --{name}", "#"]
    lines += _comment(
        "Every key is a CLI flag with dashes written as underscores (block_size "
        "is --block-size), and a flag given on the command line wins over this "
        "file."
    )
    lines.append("#")
    if reference:
        lines += _comment(
            "The keys this experiment sets come first, with any `extends` bases "
            "merged in. Every other flag follows at the bottom, commented out."
        )
    if on_disk:
        lines.append("#")
        lines += _comment(
            f"{MARKER}: the prose under each key is generated from the flag and "
            "the registry, so an edit to it is overwritten. Anything written by "
            "hand is kept, folded into the `Note:` paragraph above its key."
        )
    if note:
        lines.append("#")
        lines += _labeled(NOTE, note)
    return lines + [""] * GAP


def _banner(text: str) -> List[str]:
    rule = "# " + "=" * (WIDTH - 2)
    return [rule, f"# {text.upper()}", rule] + [""] * GAP


def _section(title: str) -> List[str]:
    return ["# " + f"--- {title} ".ljust(WIDTH - 2, "-")] + [""] * GAP


def _entry(
    key: str,
    value: Any,
    *,
    description: Optional[str] = None,
    facts: Optional[List[str]] = None,
    values: Optional[List[str]] = None,
    note: Optional[str] = None,
    commented: bool = False,
    literal: bool = False,
) -> List[str]:
    """One entry: paragraphs of comment, a blank line, then ``key: value``.

    The blank line is the whole point - it lets the eye find the keys without
    reading the prose, which is what you do when you already know the file."""
    paragraphs = [
        _comment(description) if description else [],
        facts or [],
        values or [],
        _labeled(NOTE, note) if note else [],
    ]
    lines: List[str] = []
    for paragraph in (p for p in paragraphs if p):
        if lines:
            lines.append("#")
        lines += paragraph
    body = [f"{key}: {value}"] if literal else _key_lines(key, value)
    if commented:
        body = ["# " + line for line in body]
    return lines + [""] + body + [""] * GAP


# ---------------------------------------------------------------------------
# what an entry says
# ---------------------------------------------------------------------------

# Fragments that exist only to describe the command line. A config file spells a
# list as a list, and the Options/Default paragraph says the rest, so repeating
# it in prose is noise - and it was the loudest kind of noise in this format.
_MECHANICS = re.compile(
    r"\bspace-\s*or\s*comma-separated\b|\bspace-separated\b|\bcomma-separated\b"
    r"|\brepeatable\b|\bpass\w* (?:it |the flag )?with no values\b"
    r"|\bchoices\s*:|\bavailable\s*:|\be\.g\.\s*'?--",
    re.I,
)
_MECHANICS_SENTENCE = re.compile(
    r"^\s*(?:choices|available|options)\s*:|^\s*e\.g\.\s|^\s*(?:space|comma)[-\s]",
    re.I,
)
# Abbreviations that end in a period without ending a sentence.
_ABBREV = re.compile(
    r"(?:\b(?:e\.g|i\.e|vs|etc|cf|al|resp|approx|fig|eq|ref|Dr|Mr|Ms|St)\.|\b[A-Za-z]\.)$"
)


def _description(action: argparse.Action) -> str:
    """A flag's prose: its ``help`` as the lead sentence, then its long form with
    any sentence the help already said removed. Several flags open their long doc
    by restating their help verbatim, which read as a stutter."""
    lead = _scrub(action.help)
    doc = _long_doc(action) or ""
    if lead and doc:
        doc = _without(doc, {_fold(s) for s in _sentences(lead)})
    paragraphs = []
    if lead:
        paragraphs.append(_sentence(lead))
    if doc.strip():
        paragraphs.append(doc.strip())
    return "\n\n".join(paragraphs) or "(undocumented)"


def _placeholder(action: argparse.Action) -> str:
    """What to show where a value would go, for a flag whose default is drawn
    fresh each process. Printing the number this process happened to draw would
    rewrite every example file on every launch."""
    kind = getattr(action.type, "__name__", None) or "value"
    return f"<{kind}>"


def _facts(action: argparse.Action, *, show_default: bool) -> List[str]:
    """The ``Options:``/``Default:`` paragraph. A commented-out reference entry
    skips the default, because the value printed on its key line is it."""
    lines: List[str] = []
    options = _options(action)
    if options:
        lines += _labeled("Options", options)
    if show_default:
        default = (
            "drawn fresh each run"
            if getattr(action, "volatile_default", False)
            else _inline(action.default)
        )
        lines += _labeled("Default", default)
    return lines


def _options(action: argparse.Action) -> Optional[str]:
    """What the flag accepts, in one of three descending qualities: its
    ``choices``, the keys of the registry it selects from (a free-form list flag
    has no ``choices`` but still has a namespace), or the list its ``help``
    spells out by hand - which is where ``_scrub`` took it from."""
    keys = None
    if action.choices is not None:
        keys = [str(c) for c in action.choices]
    else:
        ns = _namespace_of(action)
        if isinstance(ns, Namespace) and len(ns):
            keys = [str(k) for k in ns]
    if keys is None:
        return _help_options(action.help)
    if len(keys) <= OPTIONS_LIMIT:
        return ", ".join(keys)
    page = _docs_page(action)
    if page:
        return f"{len(keys)} of them, described in {page}"
    return ", ".join(keys)


def _help_options(help_text: Optional[str]) -> Optional[str]:
    """The ``Choices: a, b, c`` or ``Available: a, b, c`` list a hand-written
    help string carries, for the flags with no registry behind them."""
    for sentence in _sentences(" ".join((help_text or "").split())):
        match = re.match(r"\s*(?:choices|available|options)\s*:\s*(.+)", sentence, re.I)
        if match:
            return match.group(1).strip().rstrip(".")
    return None


def _namespace_of(action: argparse.Action) -> Optional[Namespace]:
    """The registry namespace a flag selects from (the first, when it names
    several)."""
    ref = getattr(action, "registry", None)
    if isinstance(ref, (tuple, list)):
        ref = ref[0] if ref else None
    return resolve_namespace(ref) if ref is not None else None


def _docs_page(action: argparse.Action) -> Optional[str]:
    ns = _namespace_of(action)
    if isinstance(ns, Namespace) and ns.title:
        return f"docs/{ns.name.replace('_', '-')}.md"
    return None


def _long_doc(action: argparse.Action) -> Optional[str]:
    """A flag's long form: its own ``doc``, or the doc of the registry
    namespace it selects from (the first, when it names several)."""
    doc = getattr(action, "doc", None)
    if doc:
        return doc
    ns = _namespace_of(action)
    return ns.doc if ns is not None else None


def _value_notes(value: Any, registry) -> List[str]:
    """One labelled paragraph per registry entry ``value`` names, searched
    through nested lists and mappings. Entries with nothing to say are skipped."""
    if registry is None:
        return []
    refs = registry if isinstance(registry, (tuple, list)) else (registry,)
    candidates = [resolve_namespace(ref) for ref in refs]
    lines: List[str] = []
    seen = set()
    for leaf in _string_leaves(value):
        if leaf in seen:
            continue
        seen.add(leaf)
        for reg in candidates:
            text = reg.describe(leaf) if isinstance(reg, Namespace) else None
            if text:
                lines += _labeled(leaf, text)
                break
    return lines


def _registry_named(key: str) -> Optional[Namespace]:
    for ns in namespaces():
        if ns.title and ns.name == key:
            return ns
    return None


def _string_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_leaves(item)


# ---------------------------------------------------------------------------
# comment mechanics
# ---------------------------------------------------------------------------


def _comment(text: str, indent: int = 0) -> List[str]:
    """``text`` as wrapped ``#`` lines; blank lines separate paragraphs."""
    lines: List[str] = []
    prefix = "# " + " " * indent
    for i, para in enumerate(p for p in text.strip().split("\n\n") if p.strip()):
        if i:
            lines.append("#")
        lines += textwrap.wrap(
            " ".join(para.split()),
            width=WIDTH,
            initial_indent=prefix,
            subsequent_indent=prefix,
            break_on_hyphens=False,
        )
    return lines


def _labeled(label: str, text: str) -> List[str]:
    """``# Label: text``, wrapped, with continuations hanging at ``HANG``. The
    only indented construct in the format, and the only one with a colon.

    A bullet paragraph hangs two columns deeper than prose does, which is what
    lets :func:`_paragraphs` tell a real bullet from a wrapped line that happens
    to begin with a dash - the difference between a note that survives a second
    formatting pass unchanged and one that grows a paragraph break every time."""
    paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    hang = "# " + " " * HANG
    bullets = "# " + " " * (HANG + 2)
    lines: List[str] = []
    if _is_bullet(paragraphs[0]):
        lines.append(f"# {label}:")
    else:
        lines += _wrap(f"{label}: {paragraphs[0]}", "# ", hang)
        paragraphs = paragraphs[1:]
    for para in paragraphs:
        if lines and not lines[-1].endswith(":"):
            lines.append("#")
        if _is_bullet(para):
            lines += _wrap(para, bullets, "# " + " " * (HANG + 4))
        else:
            lines += _wrap(para, hang, hang)
    return lines


def _wrap(text: str, initial: str, subsequent: str) -> List[str]:
    return textwrap.wrap(
        " ".join(text.split()),
        width=WIDTH,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_on_hyphens=False,
    )


def _is_bullet(text: str) -> bool:
    return text.lstrip().startswith(("* ", "- "))


def _sentences(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    out: List[str] = []
    for piece in pieces:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [p for p in out if p]


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _scrub(text: Optional[str]) -> str:
    """``help`` with its command-line mechanics removed."""
    text = " ".join((text or "").split())
    text = re.sub(
        r"\s*\(([^()]*)\)",
        lambda m: "" if _MECHANICS.search(m.group(1)) else m.group(0),
        text,
    )
    kept = [s for s in _sentences(text) if not _MECHANICS_SENTENCE.search(s)]
    return " ".join(kept).strip()


def _without(doc: str, seen: set) -> str:
    """``doc`` with every sentence in ``seen`` dropped, paragraphs preserved."""
    out = []
    for para in doc.split("\n\n"):
        kept = [s for s in _sentences(" ".join(para.split())) if _fold(s) not in seen]
        if kept:
            out.append(" ".join(kept))
    return "\n\n".join(out)


def _sentence(text: Optional[str]) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "(undocumented)"
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "?", "!", ")")) else text + "."


def _key_lines(key: str, value: Any) -> List[str]:
    """``key: value`` - one flow-style line when it fits, block style when it
    does not."""
    value = _plain(value)
    flow = f"{key}: {_inline(value, unset=False)}"
    if not isinstance(value, (dict, list)) or len(flow) <= WIDTH - 20:
        return [flow]
    try:
        return yaml.safe_dump(
            {key: value},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=WIDTH,
        ).splitlines()
    except yaml.YAMLError:
        return [f"{key}:  # ({value!r} has no YAML form)"]


def _plain(value: Any) -> Any:
    """Tuples and sets as lists, so the safe dumper accepts them."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


def _inline(value: Any, unset: bool = True) -> str:
    """``value`` as one line of flow-style YAML (``unset`` for None when
    ``unset`` is true, since that reads better in prose than ``null``)."""
    if value is None and unset:
        return "unset"
    try:
        text = yaml.safe_dump(
            _plain(value),
            default_flow_style=True,
            width=float("inf"),
            allow_unicode=True,
        )
    except yaml.YAMLError:
        return repr(value)
    return text.removesuffix("\n...\n").strip()


def _same(a: Any, b: Any) -> bool:
    return _plain(a) == _plain(b)


# ---------------------------------------------------------------------------
# reading a file back
# ---------------------------------------------------------------------------

_TOP_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:")


def _source_notes(text: str) -> Tuple[Optional[str], Dict[str, str]]:
    """``(file note, {key: note})`` - the hand-written prose in ``text``.

    A comment run is a block of ``#`` lines with no blank line in it, and it
    belongs to the next key below it - which is how anyone reads a comment - so
    two runs above one key both land on that key. A run with no key below it at
    all belongs to the file, and joins the run that opens the file as its note.

    An unformatted file has no generated comments, so a run is someone's note
    whole. A formatted file is read back through its ``Note:`` paragraphs only,
    and that is what makes the rewrite idempotent: everything else in it came
    from the parser and is about to be regenerated."""
    formatted = any(
        line.startswith("#") and MARKER in line for line in text.splitlines()
    )
    notes: Dict[str, str] = {}
    file_note: Optional[str] = None
    orphans: List[str] = []
    run: List[str] = []
    pending: Optional[str] = None
    first = True

    def flush_run() -> None:
        nonlocal run, pending, first, file_note
        if not run:
            return
        note = _note_in(run) if formatted else _uncomment(run)
        run = []
        if first:
            first = False
            file_note = note
        elif note:
            pending = f"{pending}\n\n{note}" if pending else note

    for line in text.splitlines():
        if line.startswith("#"):
            run.append(line)
            continue
        flush_run()
        key = _TOP_KEY.match(line)
        if key is not None:
            first = False
            if pending:
                notes[key.group(1)] = pending
                pending = None
        elif line.strip() and pending:
            # A continuation line of some value; nothing attaches across it.
            orphans.append(pending)
            pending = None
    flush_run()
    if pending:
        orphans.append(pending)
    if orphans:
        file_note = "\n\n".join(p for p in [file_note, *orphans] if p) or None
    return file_note, notes


def _note_in(block: List[str]) -> Optional[str]:
    """The ``Note:`` paragraph of a generated comment block, dedented. It is
    always last, so it runs to the end of the block."""
    for i, line in enumerate(block):
        body = line[1:]
        if body.strip().startswith(f"{NOTE}:"):
            head = body.strip()[len(NOTE) + 1 :].lstrip()
            tail = [_unhang(later[1:]) for later in block[i + 1 :]]
            return _paragraphs(([head] if head else []) + tail, bullet_indent=2)
    return None


def _unhang(body: str) -> str:
    """A generated continuation line with its ``Label: `` indent removed, so a
    bullet lands at column 2 and a wrapped line of prose at column 0."""
    indent = len(body) - len(body.lstrip(" "))
    return body[min(indent, HANG + 1) :]


def _uncomment(block: List[str]) -> Optional[str]:
    """A hand-written comment run as paragraph text, its common indent removed so
    bullets sit at column 0 and their continuations deeper."""
    bodies = [line[1:] for line in block]
    written = [b for b in bodies if b.strip()]
    common = min((len(b) - len(b.lstrip(" ")) for b in written), default=0)
    return _paragraphs([b[common:] if b.strip() else "" for b in bodies])


def _paragraphs(lines: List[str], bullet_indent: int = 0) -> Optional[str]:
    """Comment bodies (``#`` already stripped, dedented) as paragraph text.

    A line indented by exactly ``bullet_indent`` and opening with a bullet starts
    a paragraph of its own, so a list read back out of a note stays a list. Any
    other indent is a wrapped continuation, which is how a sentence containing
    " - " survives being re-read."""
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped and indent == bullet_indent and _is_bullet(stripped) and current:
            paragraphs.append(" ".join(current))
            current = []
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    text = "\n\n".join(p for p in paragraphs if p)
    return text or None


def _flag_groups(parser) -> List[Tuple[str, List[argparse.Action]]]:
    """``(title, actions)`` in ``--help`` order. Integrations re-open a group
    by calling ``add_argument_group`` with its title again, which argparse
    answers with a second group of the same name, so groups merge by title."""
    merged: Dict[str, List[argparse.Action]] = {}
    for group in parser._action_groups:
        if group.title in _TOGGLE_GROUPS:
            continue
        actions = [
            a
            for a in group._group_actions
            if a.option_strings
            and not isinstance(a, argparse._HelpAction)
            and a.help is not argparse.SUPPRESS
        ]
        if actions:
            merged.setdefault(group.title or "options", []).extend(actions)
    return list(merged.items())
