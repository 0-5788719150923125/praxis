"""Annotated experiment config: the YAML behind the web app's Download button.

An experiment file reaches the server parsed, so its comments are gone and its
``extends`` bases are merged in. This writes it back out with the documentation
each key carries: the flag's ``help``, its long form (its own ``doc``, or the doc
of the registry namespace it selects from), and the description of the value
chosen. The keys the experiment sets come first, grouped
as in ``--help``; every other flag follows at the bottom, commented out at its
default.
"""

import argparse
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from praxis.cli.core.parser import resolve_namespace
from praxis.registry import Namespace, namespaces

WIDTH = 80

# Groups the experiment and environment loaders fill with one toggle per YAML
# file. Those flags activate a config; they are never keys inside one.
_TOGGLE_GROUPS = ("experiments", "environments")


def reference_parser() -> argparse.ArgumentParser:
    """The parser this process launched with, integration flags included; a
    freshly built static parser when the CLI never ran (tests, tools)."""
    import praxis.cli as cli
    from praxis.cli.groups import build_static_parser

    return cli.parser if cli.parser is not None else build_static_parser()


def render_annotated_config(
    config: Dict[str, Any], parser: argparse.ArgumentParser, name: str
) -> str:
    """The experiment ``config`` as commented YAML. ``name`` is the experiment
    stem, used for the launch instructions in the header.

    ``yaml.safe_load`` of the result equals ``config``: every annotation is a
    comment, and the reference section is commented out."""
    groups = _flag_groups(parser)
    by_dest = {a.dest: (title, a) for title, actions in groups for a in actions}

    used: Dict[str, List[Tuple[str, argparse.Action]]] = {}
    unmatched: List[str] = []
    for key in config:
        hit = by_dest.get(key.replace("-", "_"))
        if hit is None:
            unmatched.append(key)
        else:
            used.setdefault(hit[0], []).append((key, hit[1]))
    used_dests = {a.dest for pairs in used.values() for _, a in pairs}

    out = _header(name)
    out += _banner("Set by this experiment")
    for title, actions in groups:
        order = {id(a): i for i, a in enumerate(actions)}
        pairs = sorted(used.get(title, []), key=lambda p: order[id(p[1])])
        if not pairs:
            continue
        out += _section(title)
        for key, action in pairs:
            out += _used_entry(key, config[key], action)
    if unmatched:
        out += _section("keys without a flag")
        out += _comment(
            "Read from the config by the code that uses them, or flags of an "
            "integration this checkout does not load."
        )
        out.append("")
        for key in unmatched:
            out += _unmatched_entry(key, config[key])

    out += _banner("Everything else, at its default")
    out += _comment(
        "Uncomment a line to set it. Flags kept out of the run hash (logging, "
        "serving) are not listed."
    )
    out.append("")
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
            out += _reference_entry(action)
    return "\n".join(out).rstrip() + "\n"


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


def _header(name: str) -> List[str]:
    lines = [f"# {name} - a Praxis experiment config", "#"]
    lines += _comment(f"Save as experiments/{name}.yml, then run:")
    lines += ["#", f"#     ./launch --{name}", "#"]
    lines += _comment(
        "Each key is a CLI flag with dashes as underscores (block_size is "
        "--block-size), and a flag given on the command line wins over this file."
    )
    lines.append("#")
    lines += _comment(
        "The keys this experiment sets come first, with any `extends` bases "
        "merged in. Every other flag follows at the bottom, commented out."
    )
    return lines + ["", ""]


def _banner(text: str) -> List[str]:
    rule = "# " + "=" * (WIDTH - 2)
    return [rule, f"#  {text.upper()}", rule, "", ""]


def _section(title: str) -> List[str]:
    return ["# " + f"--- {title} ".ljust(WIDTH - 2, "-"), ""]


def _used_entry(key: str, value: Any, action: argparse.Action) -> List[str]:
    lines = _comment(_sentence(action.help))
    doc = _long_doc(action)
    if doc:
        lines += ["#"] + _comment(doc)
    lines += _value_notes(value, getattr(action, "registry", None))
    changed = not _same(value, action.default)
    note = f"default: {_inline(action.default)}" if changed else None
    return lines + _key_lines(key, value, note) + [""]


def _unmatched_entry(key: str, value: Any) -> List[str]:
    """A key with no flag. When it is named after a registry, that registry
    still says what the key and its value are."""
    registry = _registry_named(key)
    lines: List[str] = []
    if registry is not None:
        lines += _comment(_sentence(registry.title))
        lines += _value_notes(value, registry)
    return lines + _key_lines(key, value) + [""]


def _reference_entry(action: argparse.Action) -> List[str]:
    lines = _comment(_sentence(action.help))
    doc = _long_doc(action)
    if doc:
        lines += ["#"] + _comment(doc)
    if action.choices is not None:
        lines += _comment(
            "one of: " + ", ".join(str(c) for c in action.choices), indent=2
        )
    try:
        body = _key_lines(action.dest, action.default)
    except yaml.YAMLError:
        body = [f"{action.dest}:  (default {action.default!r} has no YAML form)"]
    return lines + ["# " + line for line in body] + [""]


def _long_doc(action: argparse.Action) -> Optional[str]:
    """A flag's long form: its own ``doc``, or the doc of the registry
    namespace it selects from (the first, when it names several)."""
    doc = getattr(action, "doc", None)
    if doc:
        return doc
    ref = getattr(action, "registry", None)
    if isinstance(ref, (tuple, list)):
        ref = ref[0] if ref else None
    return resolve_namespace(ref).doc if ref is not None else None


def _value_notes(value: Any, registry) -> List[str]:
    """A definition list of every registry entry ``value`` names, searched
    through nested lists and mappings, each entry on its own line with its
    description indented beneath. Entries with nothing to say are skipped."""
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
                lines += [f"#   {leaf}"] + _comment(text, indent=4)
                break
    return ["#"] + lines if lines else []


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


def _sentence(text: Optional[str]) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "(undocumented)"
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "?", "!", ")")) else text + "."


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


def _key_lines(key: str, value: Any, note: Optional[str] = None) -> List[str]:
    """``key: value`` - one flow-style line when it fits, block style when it
    does not - with ``note`` as a trailing comment on the key's line."""
    value = _plain(value)
    flow = f"{key}: {_inline(value, unset=False)}"
    if not isinstance(value, (dict, list)) or len(flow) <= WIDTH - 20:
        lines = [flow]
    else:
        lines = yaml.safe_dump(
            {key: value},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=WIDTH,
        ).splitlines()
    if note:
        lines[0] = f"{lines[0]}  # {note}"
    return lines


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
    text = yaml.safe_dump(
        _plain(value), default_flow_style=True, width=float("inf"), allow_unicode=True
    )
    return text.removesuffix("\n...\n").strip()


def _same(a: Any, b: Any) -> bool:
    return _plain(a) == _plain(b)
