"""Cap every auxiliary's pull on the trunk at the task objective's.

THE PROBLEM THIS SOLVES, AND THE ONE IT DOES NOT. A plain sum lets whichever
term happens to carry the largest gradient decide where the shared
representation goes. That is a MAGNITUDE failure and it is invisible in the
loss curve: a regularizer sitting at 0.02 can still out-pull a cross-entropy at
3.5, because the loss VALUE and the gradient NORM are unrelated. It is not a
DIRECTION failure - two terms can be badly mismatched in size while agreeing
perfectly about which way to move - and nothing here fixes direction.
``conflict_min`` is what says whether direction also needs fixing.

THE RULE. Take each term's gradient at the trunk output, and scale any term
that pulls harder than the anchor down until it does not::

    w_i = min(1, ||g_anchor|| / ||g_i||)

Three properties, and each one is the reason it is this rule rather than a
weighting scheme:

  - IT ONLY EVER SHRINKS. A term weaker than the anchor is untouched, so
    nothing can be amplified into dominance. When no term exceeds the anchor
    the result is the plain sum, exactly - so a config whose terms are already
    well scaled is unchanged, and step 0 of a run is unchanged.
  - IT READS GRADIENTS, NOT LOSS VALUES. Every value-space balancer settles
    each weight at ``1 / L``, which means AN OBJECTIVE THAT CONVERGES IS
    REWARDED WITH AN EVER-LARGER SHARE OF THE UPDATE. The term with the least
    left to say ends up shouting, and there is no upper bound on how loudly.
    The degenerate case is already in the tree: ``arm_surgery`` is a surrogate
    whose VALUE is identically 0.0 by construction while its gradient is the
    only task signal the trunk gets under a surgical classifier. Kendall's
    ``exp(-s) L + s`` at ``L = 0`` has gradient ``+1`` in ``s`` forever, so
    ``s`` runs to minus infinity and its weight diverges. A gradient norm has
    no such pathology: a term that stops pulling stops being clipped, which is
    the whole of what happens to it.
  - THE WEIGHTS ARE CONSTANTS, NOT PARAMETERS. Nothing to learn means nothing
    to run away, and no route for the model to switch off an objective it
    finds inconvenient - the failure mode a learned weight has by
    construction, since shrinking a weight lowers the total loss.

WHAT IT CANNOT SEE. Only terms with a path to the trunk output get a row.
Parameter-only terms (``centers_rms``, router repulsions) and terms acting
UPSTREAM of it (an encoder's VQ commitment losses) have no gradient there and
pass through at weight 1. That is correct for the first kind - they do not
compete for the shared representation - and a real blind spot for the second.

COST. One classifier-sized backward per live term, on the sampling cadence rather
than every step, with the weights held between refreshes. A cap is a slowly
moving quantity; measuring it every step would pay a per-step price for
resolution nothing needs.
"""

from typing import Dict, List, Optional

from torch import Tensor, nn

from praxis.losses.trunk_grads import resolve_anchor, trunk_gradients, usable

# Steps between weight refreshes. Baked and model-agnostic, the same stance
# ObjectiveConflict and the compute profiler take toward their own cadences.
REFRESH_INTERVAL: int = 25

# Smoothing across refreshes, so a single unlucky batch cannot move a cap far.
DECAY: float = 0.9

_GROUP = "loss_blend"


class AnchorCapped(nn.Module):
    """Sum the losses, first scaling any term that out-pulls the anchor."""

    def __init__(self, interval: int = REFRESH_INTERVAL) -> None:
        super().__init__()
        self.interval = max(1, int(interval))
        self._step = 0
        self._weights: Dict[str, float] = {}
        self.anchor: Optional[str] = None

    def forward(
        self,
        losses: List[Tensor],
        names: Optional[List[str]] = None,
        trunk: Optional[Tensor] = None,
    ) -> Tensor:
        if names is None or len(names) != len(losses):
            # No identities to key on (the layer-wise trainer's fold, or a
            # caller on the old signature). A cap needs to know which term is
            # which, so the honest fallback is the plain sum.
            return sum(losses)
        if usable(trunk):
            if self._step % self.interval == 0:
                self._refresh(losses, names, trunk)
            self._step += 1
        total = None
        for name, loss in zip(names, losses):
            w = self._weights.get(name, 1.0)
            term = loss if w >= 1.0 else w * loss
            total = term if total is None else total + term
        return total if total is not None else sum(losses)

    def _refresh(self, losses, names, trunk) -> None:
        """Re-measure the trunk rows and move the caps toward what they imply."""
        rows = trunk_gradients(dict(zip(names, losses)), trunk)
        anchor = resolve_anchor(rows)
        if anchor is None:
            return
        self.anchor = anchor
        ref = float(rows[anchor].norm())
        for name, g in rows.items():
            target = 1.0 if name == anchor else min(1.0, ref / float(g.norm()))
            if name not in self._weights:
                # First sighting: take the reading. Approaching it from 1.0
                # would spend the first thousand steps of a run applying a cap
                # nothing measured.
                self._weights[name] = target
                continue
            self._weights[name] = DECAY * self._weights[name] + (1.0 - DECAY) * target
        # A term that stopped reaching the trunk keeps its last cap rather than
        # snapping back to 1.0: an absent row is usually a step where the term
        # was not computed, not evidence the term became harmless.

    def get_extra_state(self) -> dict:
        """Caps ride in the checkpoint. Without this a resume restarts every
        weight at 1.0 and spends its first thousand steps re-learning a cap it
        had already measured - and the term set is dynamic, so this is the
        mechanism for it rather than a fixed-size buffer."""
        return {"weights": dict(self._weights), "anchor": self.anchor}

    def set_extra_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self._weights = dict(state.get("weights") or {})
        self.anchor = state.get("anchor")

    def training_metrics(self) -> Dict[str, float]:
        """Per-term caps, so the dashboard shows which term was being clipped
        and by how much. All 1.0 means this strategy is the plain sum."""
        if not self._weights:
            return {}
        out = {f"blend_w_{name}": float(w) for name, w in self._weights.items()}
        out["blend_w_min"] = min(out.values())
        return out


def blend_metric_descriptions(keys) -> Dict[str, dict]:
    """Cards for whichever ``blend_w_*`` series this config carries.

    Built from the live keys because the term set is a property of the config,
    not of this module.
    """
    out: Dict[str, dict] = {
        "blend_w_min": {
            "description": (
                "Smallest weight the blend applied this step - the term being "
                "clipped hardest. 1.0 means nothing out-pulled the anchor and "
                "the fold is exactly the plain sum."
            ),
            "chart": {
                "title": "Loss Blend Weights",
                "y_label": "weight",
                "y_scale": "logarithmic",
                "group": _GROUP,
                "group_order": 472,
                "order": 10,
                "series_group": "blend_w",
                "series_label": "most clipped",
            },
        }
    }
    for key in sorted(k for k in keys if k.startswith("blend_w_")):
        if key == "blend_w_min":
            continue
        name = key[len("blend_w_") :]
        out[key] = {
            "description": (
                f"Weight applied to '{name}'. Below 1.0 means its gradient at "
                "the trunk was larger than the anchor's and it was scaled "
                "back to match."
            ),
            # No title/axis: rides blend_w_min's chart via series_group.
            "chart": {
                "group": _GROUP,
                "order": 11,
                "series_group": "blend_w",
                "series_label": name,
            },
        }
    return out
