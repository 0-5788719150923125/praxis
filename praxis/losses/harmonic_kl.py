"""Harmonic drift penalty: KL between the readout and a slow EMA of itself.

THE CLAIM THIS TESTS. The harmonic line in research/main.tex argues that the
model's readout basis is CONSTITUTIVE rather than learned-in-passing: a fixed
eigenbasis whose coefficients rotate while its form does not. Read operationally,
that says the map from representation to output distribution should mostly never
change - every output token should look, mostly, the same. This regularizer is
that sentence written as a loss, so the claim becomes falsifiable instead of
rhetorical. If the basis really is constitutive, the penalty should sit near zero
without being paid for, and switching it on should cost almost nothing. If it is
expensive, the claim is doing less work than the paper says.

WHAT IT DOES. It keeps a non-trainable EMA copy of the output classifier's
PARAMETERS, runs the SAME live hidden states through both the live classifier and
a functional call of it under the EMA parameters, and penalises the divergence
between the two distributions. Two readout evaluations, no second trunk pass, no
second model, and nothing new for the optimizer to own.

The EMA is generic over the classifier's parameters rather than assuming a
``weight``/``bias`` linear. That is not defensiveness, it is the actual case
here: the abstractinator line runs ``head_type: prismatic4``, whose readout is
``CrystalClassifier`` - a distance-based layer whose only parameter is
``centers``, the per-vocabulary prototypes (praxis/heads/crystal.py). Those
centers ARE the geometry the paper's crystal claim is about, so "how far has the
readout moved from its own recent past" is measured directly on them. An earlier
version of this file duck-typed ``.weight`` and silently no-opped on every
abstractinator config.

DIRECTION. The penalty is KL(ema || live), i.e. the EMA acts as a teacher and
the live model is charged for mass the teacher assigned and it dropped. That is
the mass-covering direction, chosen deliberately: the failure this codebase
actually suffers is collapse onto a few high-probability continuations, and
mode-seeking KL(live || ema) would happily reward exactly that. Only the live
term carries gradient; the teacher is detached.

WHAT IT DOES NOT DO. It bounds drift of the READOUT, not of the trunk. Two
different trunks that happen to feed the same classifier are indistinguishable to
it. A full trust region needs an EMA of the whole model and a second forward
pass; this is the cheap version that fits the existing regularizer contract, and
it is the version worth trying first because the claim it tests is specifically
about the readout basis.

STATUS. Opt-in. It is NOT in DEFAULT_REGULARIZERS - select it by name in a
config's ``regularizers`` list. See next/rl.md for why this is the direction the
RL work is taking instead of the forward-path reward policies.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.losses.regularizer_base import BaseRegularizer

try:
    # torch.compile is default-on in this repo, and this forward mutates module
    # state (it registers its EMA buffers on the first call and re-registers
    # them if the readout is swapped). Tracing that is at best a graph break and
    # at worst an error, so the two extra matmuls run eager. The fallback keeps
    # the module importable if the private API ever moves.
    from torch._dynamo import disable as _no_compile
except Exception:  # pragma: no cover

    def _no_compile(fn):
        return fn


# EMA horizon for the teacher, ~1/(1 - decay) = 1000 steps. Long on purpose:
# "mostly never changes" is a claim about the run, not about the last few steps,
# and a short horizon degenerates into penalising the current step's own update.
# Fixed and model-agnostic.
EMA_DECAY = 0.999

# Penalty weight. Fixed and model-agnostic; the term is a mean KL in nats, so it
# does not scale with width, vocabulary or sequence length.
KL_WEIGHT = 0.05

# Cap on positions scored per step. Both projections materialise [N, V] logits,
# so an uncapped pass costs 2 * B * T * V floats - fine at a 260-symbol byte
# vocabulary, several GB at 50k. A fixed random subsample bounds the cost at any
# vocabulary while leaving the mean-KL estimate unbiased.
MAX_POSITIONS = 4096


class HarmonicKLRegularizer(BaseRegularizer):
    """Penalise divergence between the live readout and an EMA of itself."""

    name = "harmonic_kl"

    metric_descriptions = {
        "harmonic_kl_loss": {
            "description": (
                "Weighted KL(ema_readout || live_readout). The penalty actually "
                "added to the objective."
            ),
            "chart": {
                "title": "Harmonic Drift Penalty",
                "y_label": "Loss",
                "y_scale": "linear",
                "group": "harmonic_kl",
                "group_order": 92,
                "order": 10,
            },
        },
        "harmonic_drift": {
            "description": (
                "Raw mean KL in nats between the EMA readout and the live one - "
                "how far the output distribution has moved from its own recent "
                "past. The direct test of the constitutive-basis claim: near "
                "zero means the readout is holding its form. Log-scaled because "
                "a converged run spans orders of magnitude near zero; steps "
                "where the teacher is (re)seeded emit no value at all rather "
                "than a misleading 0."
            ),
            "chart": {
                "title": "Readout Drift (nats)",
                "y_label": "KL",
                "y_scale": "logarithmic",
                "group": "harmonic_kl",
                "order": 20,
            },
        },
        "harmonic_live_entropy": {
            "description": (
                "Mean entropy of the live output distribution, in nats. Read it "
                "beside the drift: a falling entropy with low drift is the "
                "readout sharpening in place, which is collapse, not stability."
            ),
            "chart": {
                "title": "Readout Entropy",
                "y_label": "Nats",
                "y_scale": "linear",
                "group": "harmonic_kl",
                "order": 30,
            },
        },
    }

    def __init__(self, pad_id: int = 0, decay: float = EMA_DECAY):
        super().__init__()
        self.pad_id = pad_id
        self.decay = float(decay)
        # (param_name, buffer_name) pairs for the teacher's parameter copies.
        self._ema_keys: list = []
        # Set if the readout raises when called; the penalty then stays off for
        # the rest of the run rather than killing training. Reported once, and
        # loudly - a silent no-op is what made the first version of this file
        # useless on the very config it was written for.
        self._disabled = False
        # Buffers are allocated on the first forward, once the classifier's
        # shape is known, and are NON-PERSISTENT on purpose. Re-seeding the
        # teacher from the live classifier on resume makes the penalty exactly
        # zero at that instant and lets it re-converge - a no-op, not a
        # transient. (Contrast praxis/policies/engagement.py, where a
        # non-checkpointed baseline re-zeroed to a value the reward was nowhere
        # near, injecting a one-sided burst on every restart. Cold-starting to
        # the identity is safe; cold-starting to an arbitrary constant is not.)
        self._ema_ready = False
        self._metrics: dict = {}

    @staticmethod
    def _buffer_name(param_name: str) -> str:
        """Parameter names contain dots; buffer names cannot."""
        return "ema__" + param_name.replace(".", "__")

    @staticmethod
    def _signature(classifier) -> tuple:
        """(name, shape) per parameter - what a re-seed is keyed on."""
        return tuple(
            (n, tuple(p.shape)) for n, p in sorted(classifier.named_parameters())
        )

    @torch.no_grad()
    def _seed(self, classifier) -> None:
        for _, buf in self._ema_keys:  # drop a previous teacher's buffers
            self._buffers.pop(buf, None)
            self._non_persistent_buffers_set.discard(buf)
        self._ema_keys = []
        for name, param in sorted(classifier.named_parameters()):
            buf = self._buffer_name(name)
            self.register_buffer(buf, param.detach().clone(), persistent=False)
            self._ema_keys.append((name, buf))
        self._sig = self._signature(classifier)
        self._ema_ready = bool(self._ema_keys)

    def _ema_state(self) -> dict:
        """The teacher's parameter override for ``functional_call``."""
        return {name: getattr(self, buf) for name, buf in self._ema_keys}

    @torch.no_grad()
    def _update(self, classifier) -> None:
        # Name/shape compatibility is established by the re-seed guard in
        # forward(), which runs before the readout is evaluated.
        d = self.decay
        live = dict(classifier.named_parameters())
        for name, buf in self._ema_keys:
            getattr(self, buf).mul_(d).add_(live[name].detach(), alpha=1.0 - d)

    @_no_compile
    def forward(self, hidden_states: Tensor, input_ids: Tensor, **ctx) -> Tensor:
        zero = hidden_states.new_zeros(())
        classifier = ctx.get("classifier")
        if self._disabled or classifier is None or hidden_states.dim() != 3:
            self._metrics = {}
            return zero
        if not hasattr(classifier, "named_parameters"):
            self._metrics = {}
            return zero

        # Seed, or re-seed when the readout has been swapped or resized. This has
        # to happen BEFORE the readout is evaluated: a teacher whose parameters
        # no longer match would either raise or silently compare the wrong thing.
        sig = self._signature(classifier)
        if not self._ema_ready or sig != getattr(self, "_sig", None):
            self._seed(classifier)
            # A parameter-free readout has nothing to drift; stay off.
            if not self._ema_ready:
                self._disabled = True
                print(
                    "[harmonic_kl] readout has no parameters; drift penalty "
                    "disabled for this run."
                )
            # Teacher == student, so the penalty is identically zero. Returning
            # early also avoids charting a meaningless 0.
            self._metrics = {}
            return zero

        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        # Drop padded positions when the ids line up 1:1 with the reps (they do
        # not on the patch/encoder path, which is why this is conditional).
        if input_ids is not None and input_ids.numel() == flat.size(0):
            keep = input_ids.reshape(-1) != self.pad_id
            if bool(keep.any()):
                flat = flat[keep]
        if flat.size(0) == 0:
            self._metrics = {}
            return zero

        if flat.size(0) > MAX_POSITIONS:
            idx = torch.randperm(flat.size(0), device=flat.device)[:MAX_POSITIONS]
            flat = flat[idx]

        # Both sides go through the SAME readout so the comparison isolates its
        # drift. Deliberately not reusing the model's own logits: a head may
        # apply a norm or projection before its classifier, and that extra
        # structure would show up as drift it is not responsible for.
        #
        # tie_weights=False because a tied readout would otherwise have the
        # teacher's override propagated onto the live embedding table's entry in
        # the functional state, which is not what "score the same inputs under
        # the old readout" means.
        try:
            live_logits = classifier(flat)
            with torch.no_grad():
                ema_logits = torch.func.functional_call(
                    classifier, self._ema_state(), (flat,), tie_weights=False
                )
        except Exception as exc:  # readout shape/signature we cannot drive
            self._disabled = True
            self._metrics = {}
            print(
                f"[harmonic_kl] readout {type(classifier).__name__} could not be "
                f"evaluated ({exc}); drift penalty disabled for this run."
            )
            return zero

        with torch.no_grad():
            ema_logp = F.log_softmax(ema_logits.float(), dim=-1)
            ema_p = ema_logp.exp()

        live_logp = F.log_softmax(live_logits.float(), dim=-1)
        # KL(ema || live): teacher-weighted, mass-covering, gradient only in the
        # live term. Clamped at zero because a converged KL lands within float
        # error of it, and the chart is log-scaled.
        drift = (ema_p * (ema_logp - live_logp)).sum(dim=-1).mean().clamp(min=0.0)
        loss = KL_WEIGHT * drift

        with torch.no_grad():
            self._metrics = {
                "harmonic_kl_loss": float(loss.detach()),
                "harmonic_drift": float(drift.detach()),
                "harmonic_live_entropy": float(
                    -(live_logp.exp() * live_logp).sum(dim=-1).mean()
                ),
            }

        # Advance the teacher after scoring, so the penalty never measures a
        # step against a teacher that has already absorbed it.
        self._update(classifier)
        return loss

    def training_metrics(self) -> dict:
        return dict(self._metrics)
