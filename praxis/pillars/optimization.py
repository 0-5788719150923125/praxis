"""Optimization pillar: how the reported run was actually trained.

The paper argues that the geometry a model is allowed to represent decides what
it can learn. That argument has an optimizer-side half the document had been
leaving out: an update rule is *also* a choice of geometry - the norm under
which a step is steepest descent - and the framework treats it as swappable for
the same reason it treats attention and heads as swappable.

This module renders that choice for the run being written up, so a reader knows
what trained the model rather than having to assume Adam. Two kinds of content,
kept deliberately apart:

- **Facts** are read from :data:`praxis.optimization.OPTIMIZER_PROFILES` and the
  resolved experiment config - learning rate, weight decay, the secondary
  optimizer on vocab-facing parameters, wrappers, trainer. These cannot drift
  from the code, because they *are* the code.
- **Interpretation** - which norm an update descends under, and why that is the
  same argument the rest of the paper makes - is curated prose in
  :data:`GEOMETRY`, keyed by optimizer name. An optimizer with no entry still
  renders; it just gets the facts and no geometric reading.

Output: research/optimization.tex (``\\paperOptimizationSection``).

Entry point: :func:`export_optimization`, driven by :mod:`praxis.pillars.build`.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from praxis.pillars.geometries import RESEARCH_DIR

OUT_TEX = os.path.join(RESEARCH_DIR, "optimization.tex")

# Curated reading of each optimizer's update geometry. Keyed lowercase by the
# `optimizer:` value in an experiment. Absent key -> facts only, no reading.
#
# The three norms named below are the standard steepest-descent duals:
# sign(g) is the steepest descent direction under the elementwise-infinity
# norm, an orthogonalized update is steepest under the spectral norm, and a
# norm-rescaled gradient is steepest under Frobenius. That is the whole reason
# a "choice of optimizer" is a choice of geometry rather than a choice of
# hyperparameters.
GEOMETRY: Dict[str, str] = {
    "adamw": (
        "AdamW descends under a diagonal, per-coordinate rescaling of the "
        "gradient - each weight moves on its own axis, and the geometry the "
        "step respects is the one in which those axes are already the right "
        "ones. It is the field default, and this framework treats it as the "
        "control rather than the goal."
    ),
    "lion": (
        "Lion takes the \\emph{sign} of a momentum estimate, which is exactly "
        "steepest descent under the elementwise-infinity norm: every coordinate "
        "moves by the same magnitude and only the direction is learned. The "
        "geometry it respects is the hypercube, and its step size is bounded by "
        "construction - which is why it pairs cleanly with parameters whose "
        "scale is set by token frequency rather than by the loss surface."
    ),
    "muon": (
        "Muon orthogonalizes each matrix update through a Newton-Schulz "
        "iteration, which is steepest descent under the \\emph{spectral} norm: "
        "the step is bounded in its largest singular direction, so no single "
        "mode of a weight matrix can dominate an update. Two-dimensional "
        "parameters only; embeddings, the head, and the scalars route to a "
        "secondary optimizer, because orthogonalizing a token-frequency "
        "geometry is the classic way to destabilize it."
    ),
    "muongeo": (
        "Muon orthogonalizes each matrix update through a Newton-Schulz "
        "iteration, which is steepest descent under the \\emph{spectral} norm: "
        "the step is bounded in its largest singular direction. The Geo "
        "variant additionally eliminates weight decay, for the "
        "reason Section~\\ref{sec:manifold} gives - a uniform shrink is the "
        "one-knob instrument this paper argues against, and norm control here "
        "is structural (the orthogonalized step is bounded, the harmonic "
        "latents are RMS-normalized, the harmonic bases are fixed) rather than "
        "imposed."
    ),
    "liongeo": (
        "LionGeo is the clearest optimizer-side statement of this paper's own "
        "thesis, so it is worth unpacking. Three normalizations are applied to "
        "\\emph{one} shared Lion momentum, and each is steepest descent under a "
        "different norm: the sign under the elementwise-infinity norm, a "
        "Newton-Schulz orthogonalization under the spectral norm, and an "
        "RMS-rescaling under Frobenius. Rather than committing to one, the "
        "optimizer blends all three per matrix through a SMEAR-style softmax "
        "whose logits adapt online by hypergradient descent, floored so no "
        "geometry is ever extinguished. Every branch is RMS-matched, so a "
        "single learning rate bounds the step wherever the mixture settles. "
        "The model is therefore \\emph{learning which geometry to descend in}, "
        "per matrix, while it learns the weights - the same claim this paper "
        "makes about representations, applied one level down to the updates "
        "that build them. Where the mixture lands is reported, so the choice "
        "is readable rather than assumed. Weight decay is eliminated, as in "
        "MuonGeo and for the same "
        "reason."
    ),
    "mars": (
        "MARS applies a variance-reduced gradient correction on top of a "
        "preconditioned update, targeting the noise in the estimate rather "
        "than the geometry of the step."
    ),
}

# How the loss reaches the weights. Read from trainer_type / orchestration_type.
TRAINING = {
    "backprop": (
        "Credit is assigned by standard backpropagation: one global objective, "
        "one gradient, differentiated end to end through the whole forward pass."
    ),
    "mono_forward": (
        "Credit is assigned \\emph{locally}. Mono-Forward detaches gradients "
        "between layers and trains each against its own objective, so one "
        "global trajectory becomes many independent ones and no gradient "
        "traverses the full depth of the model."
    ),
    "swarm": (
        "Credit is assigned by back-propagation locally, with the remote-expert "
        "pool configured. Where peers are attached, their layer-wise updates are "
        "Mono-Forward by construction - no global backward pass to synchronize, "
        "so a slow peer lags rather than blocks and its update lands on a later "
        "step. With no peers attached the run is an ordinary single-process "
        "one, so this line reports what the configuration permits rather than "
        "how many peers happened to be online."
    ),
}

# Sources for wrappers that implement a published method. Keyed by wrapper name;
# a wrapper absent here simply renders uncited. Applied to described and bare
# wrappers alike, so the whole schedule-free family is sourced even though only
# the base one carries a blurb.
WRAPPER_CITES = {
    "schedule_free": "defazio2024schedulefree",
    "gated_schedule_free": "defazio2024schedulefree",
    "wave_schedule_free": "defazio2024schedulefree",
}

WRAPPERS = {
    "schedule_free": (
        "no learning-rate schedule; the wrapper Polyak-averages the iterate "
        "instead, so there is no decay curve to tune or to end"
    ),
    "lookahead": (
        "a slow set of weights follows the fast one at intervals, damping the "
        "trajectory without changing the update rule"
    ),
    "ema": ("a running average of the weights is kept alongside them"),
    "orthograd": (
        "the component of the gradient parallel to the weight is projected out, "
        "so the step changes direction rather than magnitude"
    ),
}


# Prose names for registry keys, so the paper reads a method name rather than
# a configuration identifier. A key absent here is title-cased from its slug.
WRAPPER_NAMES = {
    "schedule_free": "Schedule-Free",
    "gated_schedule_free": "gated Schedule-Free",
    "wave_schedule_free": "wave Schedule-Free",
    "lookahead": "Lookahead",
    "ema": "exponential moving averaging",
    "orthograd": "OrthoGrad",
}


def _display(name: str) -> str:
    """A registry key as prose. The paper names the method, never the key."""
    key = str(name)
    return WRAPPER_NAMES.get(key, key.replace("_", " ").title())


def _fmt(value) -> str:
    """A config value as readable TeX, or None if it should be skipped."""
    if value is None or value == "":
        return None
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def resolve(experiment: Optional[str] = None) -> Dict:
    """Facts about how ``experiment`` was optimized. Best-effort: any piece the
    config or the profile table does not supply is simply omitted."""
    from praxis.optimization import OPTIMIZER_PROFILES
    from praxis.pillars.framing import newest_experiment, resolve_config

    name = experiment or newest_experiment()
    cfg = resolve_config(name) if name else {}

    optimizer = str(cfg.get("optimizer") or "AdamW")
    profile = {k.lower(): v for k, v in OPTIMIZER_PROFILES.items()}.get(
        optimizer.lower(), {}
    )

    trainer = str(cfg.get("trainer_type") or "")
    if cfg.get("orchestration_type") == "swarm":
        method = "swarm"
    elif trainer.startswith("mono_forward"):
        method = "mono_forward"
    else:
        method = "backprop"

    return {
        "experiment": name,
        "optimizer": optimizer,
        "profile": profile,
        "method": method,
        "wrappers": list(cfg.get("optimizer_wrappers") or []),
    }


def section_tex(facts: Dict) -> str:
    """The ``\\paperOptimizationSection`` macro: one subsection of prose."""
    opt = facts["optimizer"]
    profile = facts["profile"]
    parts = []

    parts.append(
        "\\subsection{How the model is trained}\n\n"
        "A paper that argues architecture is not neutral ground owes the reader "
        "the same disclosure about its optimizer, because an update rule is a "
        "choice of geometry too. Steepest descent is only defined relative to a "
        "norm, and which norm a rule descends under decides which directions in "
        "weight space are cheap. The framework therefore registers optimizers "
        "the way it registers heads and attention - as swappable geometry - and "
        "this section reports the ones the present run selected."
    )

    reading = GEOMETRY.get(opt.lower())
    lead = f"This run optimizes with \\textbf{{{opt}}}."
    parts.append(f"{lead} {reading}" if reading else lead)

    # Facts from the profile, so the prose cannot drift from the code.
    bits = []
    for key, label in (
        ("lr", "learning rate"),
        ("betas", "betas"),
        ("weight_decay", "weight decay"),
    ):
        val = _fmt(profile.get(key))
        if val is not None:
            bits.append(f"{label} {val}")
    secondary = profile.get("secondary_optimizer")
    if secondary:
        bits.append(
            f"with {secondary} on the vocab-facing parameters (embeddings, "
            "head, norms and biases), which do not share the interior's geometry"
        )
    if bits:
        parts.append(
            "The configured settings are "
            + "; ".join(bits)
            + "."
            + (
                " A weight decay of zero is a deliberate departure, not an "
                "oversight; its falsifier is stated in "
                "Section~\\ref{sec:manifold}."
                if profile.get("weight_decay") == 0
                else ""
            )
        )

    named = [(w, WRAPPERS.get(w)) for w in facts["wrappers"]]

    def _cited(w: str) -> str:
        """The wrapper's name, followed by its source when it has one."""
        key = WRAPPER_CITES.get(w)
        return _display(w) + (f"~\\cite{{{key}}}" if key else "")

    described = [f"{_cited(w)} ({d})" for w, d in named if d]
    bare = [_cited(w) for w, d in named if not d]
    if described or bare:
        parts.append("Wrapped by " + ", ".join(described + bare) + ".")

    parts.append(
        TRAINING[facts["method"]]
        + " This is one point in a space the framework leaves open: "
        "back-propagation is the default rather than the commitment, and "
        "Mono-Forward, the remote-expert swarm, forward-forward and "
        "energy-based objectives are registered alternatives that change where "
        "credit comes from without changing the model they train. Which of "
        "them a run uses is a configuration line, and it is recorded here for "
        "the same reason the architecture is: so that a result can be "
        "attributed to something."
    )

    body = "\n\n".join(parts)
    return "% Generated by praxis/pillars/optimization.py - do not edit by hand.\n" + (
        "\\newcommand{\\paperOptimizationSection}{%\n" + body + "\n}\n"
    )


def export_optimization(experiment: Optional[str] = None) -> Dict:
    """Render research/optimization.tex for ``experiment``. Never raises: a run
    we cannot resolve leaves the paper's empty fallback in place."""
    try:
        facts = resolve(experiment)
    except (FileNotFoundError, ValueError, ImportError) as e:
        with open(OUT_TEX, "w") as fh:
            fh.write(
                "% Generated by praxis/pillars/optimization.py - do not edit by hand.\n"
                "\\newcommand{\\paperOptimizationSection}{}\n"
            )
        return {"experiment": None, "optimizer": None, "error": str(e)}

    with open(OUT_TEX, "w") as fh:
        fh.write(section_tex(facts))
    return facts
