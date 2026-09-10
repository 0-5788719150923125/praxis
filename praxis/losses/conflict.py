"""Objective conflict: do this model's several losses pull the trunk apart?

A typical prismatic run sums a main cross-entropy with ``harmonic_kl``, HALO's
geometric terms, the crystal centers-RMS regularizer,
``contrastive_isotropy``, MTP, router and gate penalties, and whatever the trunk
emits - all backward through the same shared representation. Nothing checks
whether the terms agree about which way that representation should move.

Stack the per-objective gradients as rows and you have the object multi-task
learning is written about: a Jacobian with one row per loss and one column per
shared parameter. PCGrad, GradNorm, CAGrad, Nash-MTL are all rules for combining
those rows by something other than a plain sum, each motivated by rows pointing
in opposing directions.

THIS MODULE IS THE MEASUREMENT, NOT THE METHOD. Every one of those rules costs
compute per step and usually a hyperparameter. Whether one would buy anything is
answered by the cosine between each objective's gradient and the ANCHOR's:
persistently negative is conflict and the case for gradient surgery, near 0
means orthogonal directions and the plain sum is fine, positive means the aux
term is a reweighting of the anchor.

THE ANCHOR IS NOT ALWAYS ``main``. Under a surgical head the mixture CE trains
only the gate and has no path to the trunk, so anchoring on it measured
nothing at all - see praxis.losses.trunk_grads, which resolves this and owns
the row extraction both this module and the blending strategies use.

EACH SERIES HAS A TWIN, because cosine is scale-invariant: a term contributing
nothing and a term contributing a lot in an independent direction read
identically. Every ``conflict_<name>`` ships with ``conflict_mag_<name>``, the
ratio ``||g_term|| / ||g_anchor||`` at the same tensor. Cosine near 0 at ratio
near 0 says inert; cosine near 0 at ratio near 1 says two comparable forces are
shaping independent directions, which is the only reading that licenses the sum.

AT THE TRUNK OUTPUT, NOT THE PARAMETERS. The honest Jacobian is w.r.t. shared
parameters, which needs a full backward per objective - the cost this module
exists to avoid paying before knowing it is worth paying. Differentiating w.r.t.
the trunk's output ACTIVATION needs only a backward through the head and answers
the same question: by the chain rule every shared parameter's gradient factors
through that tensor. It cannot see conflict arising inside the trunk.

A MISSING TERM IS A RESULT. ``centers_rms``, the gate repulsion and the router
repulsions are parameter-only - no path to the activation, so no series here.
They do not compete for the shared representation; they shape their own
parameters only.

These are LOSS TERMS. A head whose arms are trained by ONE cross-entropy through
a mixture has a multi-task problem that never appears here, because the arms are
not separate terms; that measurement lives on ``ParallelHead.arm_conflict``.
"""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from praxis.losses.trunk_grads import (
    MIN_NORM,
    resolve_anchor,
    trunk_gradients,
    usable,
)

# Steps between measurements. Each one costs one head-sized backward per live
# objective, so it is sampled rather than run every step - the same stance the
# compute profiler takes. Baked, model-agnostic: this is a diagnostic, not a
# knob an experiment is supposed to tune.
CONFLICT_INTERVAL: int = 100

_GROUP = "conflict"


class ObjectiveConflict:
    """Samples cosines between each loss term's trunk gradient and the
    anchor's, on one step in :data:`CONFLICT_INTERVAL`.

    Stateless apart from the step counter and the last measurement, so it can
    live on the model and be drained by the dynamics callback like the compute
    profiler's and the governor's stashes.
    """

    def __init__(self, interval: int = CONFLICT_INTERVAL) -> None:
        self.interval = max(1, int(interval))
        self._step = 0
        self.metrics: Dict[str, float] = {}
        # Resolved on the first measurement; which term the cosines are against.
        self.anchor: Optional[str] = None

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
        if not usable(wrt):
            return self.metrics
        if not self._due():
            return self.metrics

        rows = trunk_gradients(loss_dict, wrt)
        anchor = resolve_anchor(rows)
        if anchor is None:
            # Nothing that reaches the trunk is the task, so there is no
            # reference direction and a cosine would mean nothing. Happens when
            # every candidate anchor is detached from the trunk by design.
            return self.metrics
        self.anchor = anchor
        anchor_grad = rows[anchor]
        anchor_norm = float(anchor_grad.norm())
        if anchor_norm < MIN_NORM:
            return self.metrics

        out: Dict[str, float] = {}
        for name, g in rows.items():
            if name == anchor:
                continue
            cos = self._cosine(g, anchor_grad)
            if cos is None:
                continue
            out[f"conflict_{name}"] = cos
            # The twin the cosine cannot supply: is this term even pulling?
            out[f"conflict_mag_{name}"] = float(g.norm()) / anchor_norm

        cosines = [v for k, v in out.items() if not k.startswith("conflict_mag_")]
        if cosines:
            out["conflict_min"] = min(cosines)
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
                "Minimum over every conflict_* cosine - the most opposed "
                "objective this step. Persistently below zero is the case for "
                "gradient surgery; near zero says summing is already right."
            ),
            "chart": {
                "title": "Objective Conflict",
                "y_label": "cosine vs anchor gradient",
                "y_scale": "linear",
                "group": _GROUP,
                "group_order": 470,
                "order": 10,
                "series_group": "conflict",
                "series_label": "worst",
            },
        }
    }
    for key in sorted(k for k in keys if k.startswith("conflict_mag_")):
        name = key[len("conflict_mag_") :]
        out[key] = {
            "description": (
                f"||g_{name}|| / ||g_anchor|| at the trunk output - the half "
                "a cosine cannot give. Near 0 means inert whatever the cosine "
                "reads; near 1 means the cosine is worth believing."
            ),
            "chart": {
                "title": "Objective Magnitude vs Main Loss",
                "y_label": "||g_term|| / ||g_anchor||",
                "y_scale": "logarithmic",
                "group": _GROUP,
                "order": 20,
                "series_group": "conflict_mag",
                "series_label": name,
            },
        }
    for key in sorted(k for k in keys if k.startswith("conflict_")):
        if key == "conflict_min" or key.startswith("conflict_mag_"):
            continue
        name = key[len("conflict_") :]
        out[key] = {
            "description": (
                f"Cosine between the '{name}' gradient and the anchor "
                "objective's at the trunk output. Negative = pulling against "
                "it; ~0 = "
                "independent; positive = a reweighting."
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
