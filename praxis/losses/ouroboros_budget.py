import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from praxis.activations.ouroboros import (
    MAX_STEPS,
    drain_step_counts,
    enable_accounting,
)
from praxis.losses.regularizer_base import BaseRegularizer

# The budget, as an EQUALITY constraint: the loop must spend this many
# activation-steps per feature on average, no more and no less.
#
# 2.0, revised from 1.0 after run 865ed0387. At 1.0 - the un-looped baseline's
# exact cost - the comparison against -k was compute-matched and the only way to
# win was reallocation. The model declined: over 4.3k steps the gates closed
# monotonically (extra_frac 0.032 -> 0.0027, exit_1 -> 0.996, steps -> 1.001)
# with steps_std pinned at ~3e-4, i.e. a UNIFORM collapse back to Serpent rather
# than any redistribution. The budget's gradient is exact and applies to every
# gate at once; the task's "this feature deserves another step" signal is
# diffuse and routed through a first-order surrogate. The exact term won.
#
# Forcing the mean ABOVE the init value inverts that pressure: the loop must
# spend two steps per feature, so the open question becomes WHERE, not WHETHER.
# This costs the compute-matching against -k, deliberately - `ouroboros_steps_std`
# is now the whole measurement. Uniform depth 2 means no specialization; a
# deep/shallow split means there was something to find.
TARGET_STEPS: float = 2.0

# Bound on the dual variable, applied SYMMETRICALLY - lambda is signed. A
# non-negative multiplier (softplus) encodes an inequality, "steps <= target",
# which is satisfied at init and decays to zero without ever opening a gate.
# The equality constraint needs a multiplier free to go negative, which is what
# pushes step count UP while the loop is under budget.
LAMBDA_MAX: float = 10.0


class _GradReverse(torch.autograd.Function):
    """Sign-flip on the backward pass. Lets the model's own optimizer run the
    dual ascent on lambda - same learning rate, same schedule, no separate step
    size to pick."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad.neg()


def _exit_descriptions() -> dict:
    """One chart, one line per exit depth. ``series_group`` merges them, so the
    family stays a distribution rather than MAX_STEPS+1 unrelated scalars."""
    out = {}
    for k in range(MAX_STEPS + 1):
        if k == 0:
            what = "never opened at all - the activation is bypassed entirely"
        elif k == MAX_STEPS:
            what = "ran the full loop without ever converging"
        else:
            what = f"converged and stopped after {k} step{'s' if k > 1 else ''}"
        out[f"ouroboros_exit_{k}"] = {
            "description": f"Fraction of features that {what}.",
            "chart": {
                "title": "Ouroboros Exit Depth",
                "y_label": "Fraction of features",
                "y_scale": "linear",
                "group": "ouroboros",
                "series_group": "ouroboros_exit",
                "series_label": f"{k} steps",
                "order": 40 + k,
            },
        }
    return out


class OuroborosBudget(BaseRegularizer):
    """Holds Ouroboros' mean expected step count at the baseline's budget.

        loss = lambda * (E[steps] - TARGET_STEPS)

    ``E[steps]`` is analytic: each gate's hard-concrete P(z > 0) has a closed
    form, so the expected number of open (feature, step) pairs is a
    differentiable function of the gate logits rather than a sampled count.

    ``lambda`` is a Lagrange multiplier, not a coefficient. It sees a reversed
    gradient, so the same optimizer that minimizes the loss w.r.t. the model
    maximizes it w.r.t. lambda: the multiplier climbs while the model overspends
    and decays once it is back under budget. There is no penalty weight to tune
    here, which is the only reason this belongs in the objective at all.

    At init the activation is saturated to one open step, so E[steps] ~= 1.0,
    the constraint is already satisfied and this term contributes ~0. It only
    starts pushing when the gates open.

    See praxis/activations/ouroboros.py and next/ouroboros.md.
    """

    name = "ouroboros_budget"

    metric_descriptions = {
        "ouroboros_steps": {
            "description": (
                "Mean expected activation-steps per feature. The budget is an "
                "equality constraint holding this at TARGET_STEPS, so the "
                "question the run asks is not how much the loop spends but "
                "WHERE it spends it."
            ),
            "chart": {
                "title": "Ouroboros Step Budget",
                "y_label": "Steps",
                "y_scale": "linear",
                "group": "ouroboros",
                "group_order": 94,
                "order": 10,
            },
        },
        "ouroboros_extra_frac": {
            "description": (
                "Share of the step budget spent beyond the first step. Zero "
                "means the recurrence is unused and the run is Serpent."
            ),
            "chart": {
                "title": "Ouroboros Recurrence Use",
                "y_label": "Fraction",
                "y_scale": "linear",
                "group": "ouroboros",
                "order": 20,
            },
        },
        "ouroboros_steps_std": {
            "description": (
                "Spread of expected step count ACROSS features. The mean cannot "
                "distinguish 'every feature half-commits' from 'features have "
                "split into deep and shallow groups' - this can. Zero means "
                "uniform depth, i.e. the loop is behaving as one bigger "
                "activation rather than specializing."
            ),
            "chart": {
                "title": "Ouroboros Depth Spread",
                "y_label": "Std. steps",
                "y_scale": "linear",
                "group": "ouroboros",
                "order": 25,
            },
        },
        "ouroboros_lambda": {
            "description": (
                "Signed Lagrange multiplier on the step budget. NEGATIVE while "
                "the loop is under budget (pushing depth up), positive while "
                "over. Sitting hard against its bound means the constraint has "
                "stopped being informative and the target should move."
            ),
            "chart": {
                "title": "Ouroboros Budget Pressure",
                "y_label": "Lambda",
                # Linear, not logarithmic: the multiplier is signed now.
                "y_scale": "linear",
                "group": "ouroboros",
                "order": 30,
            },
        },
        "ouroboros_gain": {
            "description": (
                "Mean output/input RMS ratio across features. The iterated map "
                "drifts rather than converging, so this rises with depth by "
                "construction - read it alongside the depth/gain correlation."
            ),
            "chart": {
                "title": "Ouroboros Output Gain",
                "y_label": "Out / in RMS",
                "y_scale": "linear",
                "group": "ouroboros",
                "order": 26,
            },
        },
        "ouroboros_depth_gain_corr": {
            "description": (
                "Correlation across features between depth and output gain. "
                "THE control on this run: because deep features exit louder, a "
                "rising steps_std could be pure volume rather than computation. "
                "Calibration - a HAND-FORCED split that does no useful "
                "computation at all (half the features opened to 3 steps) "
                "already scores ~0.59, so that is the artifact baseline, not "
                "zero. A learned split near 0.59 is indistinguishable from "
                "volume; meaningfully below it means depth is varying for some "
                "reason other than loudness. Exactly 0 at init, where every "
                "feature has identical depth and the correlation is undefined."
            ),
            "chart": {
                "title": "Ouroboros Depth/Gain Confound",
                "y_label": "Correlation",
                "y_scale": "linear",
                "group": "ouroboros",
                "order": 27,
            },
        },
        **_exit_descriptions(),
    }

    def __init__(self, pad_id: int = 0, target: float = TARGET_STEPS):
        super().__init__()
        self.pad_id = pad_id
        self.target = target
        # Zero-init: no budget pressure at step 0, and the dual moves off zero
        # in whichever direction the constraint is violated - negative while the
        # loop is under budget (pushing steps up), positive while over.
        # Shape [1] rather than 0-dim: the schedule_free wrapper swaps
        # parameters via `x.view(torch.uint8).bitwise_xor_(...)`, which cannot
        # view a 0-dim tensor as a different element size.
        self.lambda_raw = Parameter(torch.zeros(1))
        self._metrics: dict = {}
        enable_accounting()

    def reset(self) -> None:
        """Discard anything the activations pushed before this forward.

        Ouroboros pushes whenever it runs in training mode, but this
        regularizer is only called when labels are present. Without this,
        a labels-free training forward would strand live graphs on the stack
        for a later step to drain - after the optimizer has swapped the
        parameters they reference in place.
        """
        drain_step_counts()

    def forward(self, hidden_states: Tensor, input_ids: Tensor, **_) -> Tensor:
        entries = drain_step_counts()
        if not entries:
            # No Ouroboros in this model, or an eval pass. Nothing to constrain.
            return hidden_states.new_zeros(())

        # [calls, MAX_STEPS] -> mean survival per step index across every call
        # that actually executed this forward.
        survival = torch.stack([curve for curve, _ in entries]).mean(dim=0)
        expected_steps = survival.sum()

        # Signed and symmetrically bounded; squeezed to 0-dim for the loss math,
        # while the PARAMETER stays [1]. tanh saturates rather than clamps, so
        # the dual keeps a gradient at the bound instead of going dead.
        lam = LAMBDA_MAX * torch.tanh(_GradReverse.apply(self.lambda_raw)).squeeze(0)
        loss = lam * (expected_steps - self.target)

        with torch.no_grad():
            self._metrics = self._describe(survival.detach(), expected_steps.detach())
            self._metrics["ouroboros_lambda"] = float(lam.detach())
            self._metrics.update(self._spread(entries))
        return loss

    def _describe(self, survival: Tensor, expected_steps: Tensor) -> dict:
        """Turn the survival curve into an exit-depth distribution.

        ``survival[k]`` is the fraction of features still looping at step k, so
        the fraction that stopped after exactly k steps is the drop between
        consecutive entries. Survival is monotone by construction (each step
        multiplies by a sigmoid < 1), so every bin is non-negative and they sum
        to 1 - it is a real distribution, not a set of loose scalars."""
        exits = torch.cat(
            [1.0 - survival[:1], survival[:-1] - survival[1:], survival[-1:]]
        ).clamp_min(0.0)

        total = float(expected_steps)
        extra = float(survival[1:].sum()) if MAX_STEPS > 1 else 0.0
        metrics = {
            "ouroboros_steps": total,
            "ouroboros_extra_frac": extra / total if total > 0 else 0.0,
        }
        for step in range(MAX_STEPS + 1):
            metrics[f"ouroboros_exit_{step}"] = float(exits[step])
        return metrics

    def _spread(self, entries: list) -> dict:
        """Across-feature spread of depth, mean gain, and their correlation.

        Summed (not averaged) moments, so sites of different width combine by
        feature count rather than each getting an equal vote."""
        pooled = torch.stack([stats for _, stats in entries]).sum(dim=0)
        sum_s, sum_s2, sum_g, sum_g2, sum_sg, count = pooled.unbind()
        if float(count) <= 0:
            return {}

        mean_s = sum_s / count
        mean_g = sum_g / count
        var_s = (sum_s2 / count - mean_s * mean_s).clamp_min(0.0)
        var_g = (sum_g2 / count - mean_g * mean_g).clamp_min(0.0)
        cov = sum_sg / count - mean_s * mean_g

        # Undefined while either side is constant - which is exactly the state
        # at init, when every feature has identical depth. Report 0 rather than
        # a NaN the dashboard would have to special-case.
        denom = var_s.sqrt() * var_g.sqrt()
        corr = float(cov / denom) if float(denom) > 1e-12 else 0.0

        return {
            "ouroboros_steps_std": float(var_s.sqrt()),
            "ouroboros_gain": float(mean_g),
            "ouroboros_depth_gain_corr": max(-1.0, min(1.0, corr)),
        }

    def training_metrics(self) -> dict:
        return dict(self._metrics)
