"""Objective conflict: do this model's several losses pull the trunk apart?

A model here rarely trains on one objective. A typical prismatic run sums a
main cross-entropy with ``harmonic_kl``, HALO's geometric terms, the crystal
centers-RMS regularizer, ``contrastive_isotropy``, MTP, router and gate
penalties, and whatever the trunk itself emits - all backward through the same
shared representation. Summing them is the default and nothing checks whether
the terms actually agree about which way that representation should move.

Stack the per-objective gradients as rows and you have the object multi-task
learning is written about: a Jacobian with one row per loss and one column per
shared parameter. Every method in that literature - PCGrad's projection onto
the conflicting gradient's normal plane, GradNorm's norm equalization, CAGrad,
Nash-MTL - is a rule for combining those rows by something other than a plain
sum, and each is motivated by rows that point in opposing directions so one
objective silently cancels another.

THIS MODULE IS THE MEASUREMENT, NOT THE METHOD. Adopting any of those rules
costs compute on every step and, in most cases, a hyperparameter. The question
of whether it would buy anything is answered by one number per pair: the cosine
between each objective's gradient and the main loss's. Persistently negative
cosines are conflict and the case for gradient surgery; cosines hovering near 0
mean the terms are shaping orthogonal directions and the plain sum is already
fine; positive cosines mean the aux term is a reweighting of the main one.

WHY AT THE TRUNK OUTPUT, NOT THE PARAMETERS. The honest Jacobian is with
respect to shared PARAMETERS, which needs a full backward per objective - the
cost this module exists to avoid paying before knowing it is worth paying.
Differentiating with respect to the trunk's output ACTIVATION instead needs
only a backward through the head, and it answers the same question: by the
chain rule every shared parameter's gradient factors through this tensor, so
two objectives whose activation gradients oppose each other oppose each other
in the trunk as well. It cannot see conflict that arises only inside the trunk,
and it says nothing about terms that never touch the activation at all.

WHICH IS WHY A MISSING TERM IS A RESULT. ``centers_rms``, the gate repulsion
and the router repulsions are parameter-only: they have no path to the
activation, so they emit no series here. That is the correct reading, not a
gap. They do not compete with the main loss for the shared representation;
they only shape their own parameters.
"""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

# Steps between measurements. Each one costs one head-sized backward per live
# objective, so it is sampled rather than run every step - the same stance the
# compute profiler takes. Baked, model-agnostic: this is a diagnostic, not a
# knob an experiment is supposed to tune.
CONFLICT_INTERVAL: int = 100

# The objective every other one is compared against.
ANCHOR: str = "main"

# Below this the anchor or term gradient is numerically zero and the cosine is
# noise, so the series is skipped for that step rather than reported as 0.
MIN_NORM: float = 1e-12

_GROUP = "conflict"


class ObjectiveConflict:
    """Samples cosines between each loss term's trunk gradient and the main
    loss's, on one step in :data:`CONFLICT_INTERVAL`.

    Stateless apart from the step counter and the last measurement, so it can
    live on the model and be drained by the dynamics callback like the compute
    profiler's and the governor's stashes.
    """

    def __init__(self, interval: int = CONFLICT_INTERVAL) -> None:
        self.interval = max(1, int(interval))
        self._step = 0
        self.metrics: Dict[str, float] = {}

    def _due(self) -> bool:
        due = self._step % self.interval == 0
        self._step += 1
        return due

    @torch.no_grad()
    def _cosine(self, a: Tensor, b: Tensor) -> Optional[float]:
        na, nb = a.norm(), b.norm()
        if float(na) < MIN_NORM or float(nb) < MIN_NORM:
            return None
        return float((a.flatten() @ b.flatten() / (na * nb)).clamp(-1.0, 1.0).item())

    def measure(self, loss_dict: Dict[str, Any], wrt: Tensor) -> Dict[str, float]:
        """Update and return ``{conflict_<name>: cosine}`` for the live terms.

        ``wrt`` is the trunk output the head classifies. Returns the standing
        measurement unchanged on steps that are not sampled, so the chart holds
        its value between samples rather than going sparse.
        """
        # Dynamo cannot trace autograd.grad, and a control-flow branch that
        # flips every `interval` steps would thrash the compiled graph. Under
        # compile this installs nothing and the cards never appear, which is
        # the compute profiler's convention for the same situation.
        if torch.compiler.is_compiling():
            return self.metrics
        if not self._due():
            return self.metrics
        if not isinstance(wrt, Tensor) or not wrt.requires_grad:
            return self.metrics

        anchor_loss = loss_dict.get(ANCHOR)
        if not isinstance(anchor_loss, Tensor) or not anchor_loss.requires_grad:
            return self.metrics

        try:
            anchor_grad = torch.autograd.grad(
                anchor_loss, wrt, retain_graph=True, allow_unused=True
            )[0]
        except RuntimeError:
            # A term outside the retained graph (a stale container entry, an
            # already-freed branch). Diagnostics never take the run down.
            return self.metrics
        if anchor_grad is None:
            return self.metrics
        anchor_grad = anchor_grad.detach().float()

        out: Dict[str, float] = {}
        for name, term in loss_dict.items():
            if name == ANCHOR:
                continue
            if not isinstance(term, Tensor) or not term.requires_grad:
                continue
            try:
                g = torch.autograd.grad(
                    term, wrt, retain_graph=True, allow_unused=True
                )[0]
            except RuntimeError:
                continue
            if g is None:
                # Parameter-only term: no path to the shared representation, so
                # it cannot conflict over it. Emitting nothing is the answer.
                continue
            cos = self._cosine(g.detach().float(), anchor_grad)
            if cos is not None:
                out[f"conflict_{name}"] = cos

        if out:
            out["conflict_min"] = min(out.values())
            self.metrics = out
        return self.metrics


def conflict_metric_descriptions(keys) -> Dict[str, dict]:
    """Dashboard cards for whichever ``conflict_*`` series are live.

    Built from the stash's own keys because the set of objectives is a
    property of the config, not of this module: a run with MTP off has no
    ``conflict_mtp`` series and should show no empty card for one.
    """
    out: Dict[str, dict] = {
        "conflict_min": {
            "description": (
                "The most opposed objective this step: the minimum over every "
                "conflict_* cosine. Sitting persistently below zero is the case "
                "for gradient surgery (PCGrad and relatives); hovering near zero "
                "says the objectives shape orthogonal directions and summing "
                "them is already the right combination."
            ),
            "chart": {
                "title": "Objective Conflict",
                "y_label": "cosine vs main-loss gradient",
                "y_scale": "linear",
                "group": _GROUP,
                "group_order": 470,
                "order": 10,
                "series_group": "conflict",
                "series_label": "worst",
            },
        }
    }
    for key in sorted(k for k in keys if k.startswith("conflict_")):
        if key == "conflict_min":
            continue
        name = key[len("conflict_") :]
        out[key] = {
            "description": (
                f"Cosine between the '{name}' gradient and the main loss's, both "
                "taken at the trunk output the head classifies. Negative means "
                f"'{name}' is pulling the shared representation against the main "
                "objective; near zero means the two are shaping independent "
                "directions; positive means it is largely a reweighting of the "
                "main loss. Sampled one step in "
                f"{CONFLICT_INTERVAL}. Parameter-only terms (centers_rms, the "
                "gate and router repulsions) never appear here - they have no "
                "path to the shared representation to conflict over."
            ),
            # No title/axis: rides conflict_min's chart via series_group.
            "chart": {
                "group": _GROUP,
                "order": 11,
                "series_group": "conflict",
                "series_label": name,
            },
        }
    return out
