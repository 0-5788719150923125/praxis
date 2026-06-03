"""Proofs: small formal claims attached to framings, checked for consistency.

Where a framing makes an argument in prose, a proof states the load-bearing
piece formally - a lemma or proposition with a short body - and renders inline
beside the framing that invokes it. Each proof is one YAML file in ``proofs/``
(keyed by stem); a framing cites its support via ``proofs: [keys]`` and each
proof names what it backs via ``supports: [framing-ids]``.

This is not a logical verifier (no LEAN); it is a *consistency* checker - the
robust-but-approximate prover the work needs. :func:`check_consistency` verifies
the wiring hangs together: every cited key resolves, the framing/proof
references agree in both directions, ``given`` dependencies resolve without
cycles, and every framing claim is either backed or flagged unproven.

A proof file::

    kind: lemma                 # lemma | proposition | theorem
    supports: [halting-trinary] # framings it backs (must cite it back)
    given: [lemma-foo]          # other proofs it builds on (optional)
    statement: >                # the claim
      ...
    proof: >                    # the argument; amsthm adds the QED box
      ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

PROOFS_DIR = Path(__file__).parent / "proofs"
KINDS = ("lemma", "proposition", "theorem")


@dataclass(frozen=True)
class Proof:
    """One formal claim, loaded from a proofs/*.yml file."""

    id: str
    kind: str
    statement: str
    proof: str
    supports: List[str] = field(default_factory=list)
    given: List[str] = field(default_factory=list)


def _norm(text: str) -> str:
    """Folded-scalar prose -> TeX paragraphs (newline runs become breaks)."""
    return re.sub(r"\n+", "\n\n", (text or "").strip())


def _load_proof(path: Path) -> Proof:
    raw = yaml.safe_load(path.read_text()) or {}
    for key in ("statement", "proof", "supports"):
        if key not in raw:
            raise ValueError(f"{path.name}: proof must define '{key}'")
    kind = raw.get("kind", "lemma")
    if kind not in KINDS:
        raise ValueError(f"{path.name}: kind must be one of {KINDS}, got '{kind}'")
    return Proof(
        id=path.stem,
        kind=kind,
        statement=_norm(raw["statement"]),
        proof=_norm(raw["proof"]),
        supports=list(raw.get("supports") or []),
        given=list(raw.get("given") or []),
    )


def load_proofs() -> Dict[str, Proof]:
    """All proofs from ``proofs/``, keyed by id (filename stem)."""
    return {p.stem: _load_proof(p) for p in sorted(PROOFS_DIR.glob("*.yml"))}


PROOFS: Dict[str, Proof] = load_proofs()


def render_proof(proof: Proof) -> str:
    """A proof as an amsthm statement + proof block, labelled ``prf:<id>``."""
    return (
        f"\\begin{{{proof.kind}}}\\label{{prf:{proof.id}}}\n{proof.statement}\n"
        f"\\end{{{proof.kind}}}\n\\begin{{proof}}\n{proof.proof}\n\\end{{proof}}"
    )


def check_consistency(framings: Dict, proofs: Dict[str, Proof] = PROOFS) -> Dict[str, List[str]]:
    """Verify the framing<->proof wiring. Static over all framings/proofs (not
    config-gated). Returns {"errors": [...], "warnings": [...]}.

    Errors (broken wiring): a cited proof/framing/dependency that does not
    exist, or a one-sided citation (a cites b but b does not name a back), or a
    cycle in ``given``. Warnings: a framing that makes a claim with no proof.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # framing -> proof references, cross-checked both ways.
    for fid, frag in framings.items():
        cited = getattr(frag, "proofs", []) or []
        if not cited:
            warnings.append(f"framing '{fid}' has no proof (claim unbacked)")
        for pkey in cited:
            proof = proofs.get(pkey)
            if proof is None:
                errors.append(f"framing '{fid}' cites missing proof '{pkey}'")
            elif fid not in proof.supports:
                errors.append(
                    f"framing '{fid}' cites '{pkey}', but '{pkey}' does not support it back"
                )

    for pid, proof in proofs.items():
        for fid in proof.supports:
            frag = framings.get(fid)
            if frag is None:
                errors.append(f"proof '{pid}' supports unknown framing '{fid}'")
            elif pid not in (getattr(frag, "proofs", []) or []):
                errors.append(
                    f"proof '{pid}' supports '{fid}', but '{fid}' does not cite it back"
                )
        for dep in proof.given:
            if dep not in proofs:
                errors.append(f"proof '{pid}' depends on missing proof '{dep}'")

    errors.extend(_cycles(proofs))
    return {"errors": errors, "warnings": warnings}


def _cycles(proofs: Dict[str, Proof]) -> List[str]:
    """Report any cycle in the ``given`` dependency graph."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in proofs}
    out: List[str] = []

    def visit(pid, stack):
        color[pid] = GREY
        for dep in proofs[pid].given:
            if dep not in proofs:
                continue
            if color[dep] == GREY:
                cyc = " -> ".join(stack[stack.index(dep):] + [dep])
                out.append(f"cyclic proof dependency: {cyc}")
            elif color[dep] == WHITE:
                visit(dep, stack + [dep])
        color[pid] = BLACK

    for pid in proofs:
        if color[pid] == WHITE:
            visit(pid, [pid])
    return out
