"""A bank of activations, blended into one. The wrapper form of "use two
nonlinearities", so anything that writes ``ACT2FN[name]`` gets it and the bank
can be any length. The activation-level analogue of ``ParallelClassifier``: N branches
over the same input, combined by learned weights, rather than a fixed pipeline.

THE METHOD. Manessi & Rozza, "Learning Combinations of Activation Functions"
(arXiv:1801.09403). Given base activations ``F = {f_1 .. f_N}``, learn

    conv(F):  sum_i c_i f_i(x),  sum_i c_i = 1, c_i >= 0     (``convex``)
    aff(F):   sum_i c_i f_i(x),  sum_i c_i = 1               (``affine``)

The convex hull is the regularized version: it cannot flip a base function's
sign, so it preserves monotonicity of the basis. The affine hull can subtract
one branch from another and build non-monotonic shapes the basis does not
contain. The paper reports +3.01 top-1 for AlexNet on ILSVRC-2012 over fixed
activations, on a basis of identity / ReLU / tanh.

WHERE THE COEFFICIENTS COME FROM is the only thing that varies across types, so
an ablation between any two of them is one variable:

    mix         learned scalars through a softmax        conv(F)
    mix_affine  learned scalars, sum-to-one, signs free  aff(F)
    mix_gated   per element, read off the input VALUE    ours
    mix_split   per element, read off an EXTERNAL index  ours

``mix_split`` is the discrete one and carries no coefficient parameters: the
caller hands each element a fraction in [0, 1) saying where it sits in whatever
index space the caller owns, the bank is cut into N equal segments, and the
element takes the branch its key lands in. PEER passes
``expert / num_experts``, so the function class is a permanent property of a
bank row and an expert specializes into it. The fraction contract generalizes:
a head axis, a depth, a codebook slot or a position all normalize the same way.

``mix_gated`` is the continuous opposite number:

    c(x) = softmax(slope * x + bias)   [.., N],  elementwise
    y    = sum_i c_i(x) f_i(x)

one distribution per ELEMENT, so the model learns which function class to use in
which input REGIME - a non-periodic branch near zero and a periodic one in the
tails, say - re-decided per token, rather than one ratio for every activation it
will ever see.

COEFFICIENTS ARE SCALAR, NOT PER-CHANNEL. The paper parameterizes its
combination as a kernel-size-1 conv, i.e. one coefficient vector per channel.
That needs the last axis to be a feature axis, and here it often is not: PEER
applies its activation to ``[b, n, h, k]``, whose last axis is retrieval RANK.
Scalar coefficients keep this a pure elementwise ``R -> R`` function, which is
the contract - it can stand anywhere an activation stands, including inside a
lazily-shaped one. ``mix_gated`` recovers element-level resolution without a
feature axis, because it reads the input VALUE rather than its POSITION.

INITIALIZATION IS UNIFORM. A mixture has no baseline to start as, so biasing it
toward branch 0 would pre-load the answer to the question it exists to ask.
Uniform init also puts ``activation_mix_entropy`` at exactly 1.0 on step 0, so
any movement in that chart is signal rather than an offset.

COST. All N branches are evaluated on the whole tensor, including under
``mix_split``, where the one-hot selection is ragged (a token's retrieved
experts are an arbitrary mix of segments, so there is no contiguous slice to
hand each branch). N elementwise passes plus the blend, against 1 - negligible
next to the matmul that produced the tensor, but linear in ``N``.
"""

import math
import re
from collections.abc import Mapping

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Imported lazily inside __init__: praxis.activations imports this module to
# register the profiles below, so a top-level import would be circular.

# Public type name -> coefficient source. This is the whole surface of the
# thing: a config names a type and supplies `values`, and the type says nothing
# except where the blending weights come from.
MIXTURE_MODES: Dict[str, str] = {
    "mix": "convex",
    "mix_affine": "affine",
    "mix_gated": "gated",
    "mix_split": "keyed",
}
MODES: Tuple[str, ...] = tuple(dict.fromkeys(MIXTURE_MODES.values()))


class ActivationMixture(nn.Module):
    """Blend a bank of activations into a single activation module. Declared as
    a type plus its values, e.g. ``{type: mix_split, values: [servant, swish]}``,
    since the bank is never baked into the registry name.

    Args:
        activations: registry keys for the base functions, as a sequence or a
            comma-separated string. Any key in ``praxis.activations.ACT2CLS``
            works, including parametric ones (``snake``, ``serpent``, ``prelu``)
            and other mixtures.
        mode: where the coefficients come from. ``convex`` (learned scalars
            through a softmax, the paper's ``conv(F)``), ``affine``
            (learned scalars, sum-to-one with signs free, the paper's
            ``aff(F)``), ``gated`` (per-element, read off the input VALUE), or
            ``keyed`` (per-element, read off an external key the caller
            supplies - a hard partition rather than a blend).
    """

    # Per-branch share keys are named after the branch, so they are only known
    # once a bank is declared. The collectors read `metric_descriptions` off the
    # CLASS (praxis/metrics/specialization.py), so instances register their keys
    # here at construction. Every mixture in a run comes from one registry
    # profile, so this converges to exactly the keys that get logged.
    metric_descriptions: Dict[str, dict] = {
        "activation_mix_entropy": {
            "description": (
                "Normalized entropy of the mixture coefficients. 1.0 is the uniform "
                "blend every mixture starts at; 0.0 means one activation won and the "
                "bank collapsed to a single function."
            ),
            "chart": {
                "title": "Activation Mixture Entropy",
                "y_label": "H(c) / log N",
                "y_scale": "linear",
                "group": "activation_mix",
                "group_order": 94,
                "order": 10,
            },
        },
        "activation_mix_top_share": {
            "description": (
                "Largest single coefficient. 1/N is uniform; a climb toward 1.0 says "
                "the model is discarding the rest of the bank rather than mixing it."
            ),
            "chart": {
                "title": "Activation Mixture Top Share",
                "y_label": "max c",
                "y_scale": "linear",
                "group": "activation_mix",
                "order": 20,
            },
        },
        "activation_mix_routing": {
            "description": (
                "Gated mode only: spread of each coefficient across elements. "
                "Dispersion is the measure - a large but constant preference is a "
                "static blend, not routing. Zero means no routing."
            ),
            "chart": {
                "title": "Activation Mixture Routing",
                "y_label": "mean_i std_elem(c_i)",
                "y_scale": "linear",
                "group": "activation_mix",
                "order": 30,
            },
        },
    }

    def __init__(
        self,
        activations: Union[str, Sequence[str]] = (),
        mode: str = "convex",
        type_name: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from praxis.activations import build_activation

        if isinstance(activations, str):
            activations = [part.strip() for part in activations.split(",")]
        specs: Tuple[Any, ...] = tuple(spec for spec in activations if spec)
        # A bank entry is a name or a nested spec; the label is what a repr and
        # a metric key can use, which for a nested one is its type.
        names: Tuple[str, ...] = tuple(
            spec.get("type", "mix") if isinstance(spec, Mapping) else str(spec)
            for spec in specs
        )
        if len(names) < 2:
            raise ValueError(
                f"A mixture needs at least two base activations, got {names or '()'}. "
                f"Declare it as `{{type: {type_name or 'mix'}, values: [gelu, tanh]}}`."
            )
        if mode not in MODES:
            raise ValueError(f"Unknown mixture mode {mode!r}; expected one of {MODES}")

        self.names: Tuple[str, ...] = names
        self.mode: str = mode
        self.type_name: str = type_name or _MODE_NAMES.get(mode, mode)
        # Recursion is deliberate: a bank entry may itself be a typed spec, so a
        # mixture can hold a mixture. Nothing here special-cases that.
        self.branches: nn.ModuleList = nn.ModuleList(
            [build_activation(spec) for spec in specs]
        )

        count = len(names)
        if mode == "convex":
            # Softmax over free logits: non-negative and sums to one by
            # construction, so conv(F) holds at every point in training rather
            # than being projected back onto after each step.
            self.logits = nn.Parameter(torch.zeros(count))
        elif mode == "affine":
            # aff(F) drops the sign constraint but keeps sum-to-one, and the
            # cheapest exact way to hold that is to re-center the free
            # parameters every forward: `c = w + (1 - sum w) / N` sums to one
            # for any `w` and leaves the gradient untouched apart from the
            # (harmless) removal of its uniform component.
            self.coefficients = nn.Parameter(torch.full((count,), 1.0 / count))
        elif mode == "gated":
            # A per-element distribution read off the input value. Both are
            # zero-init, so a gated mixture starts as a uniform blend - the same
            # function a `convex` one starts as - and the routing anneals in.
            self.slope = nn.Parameter(torch.zeros(count))
            self.bias = nn.Parameter(torch.zeros(count))
        # `keyed` carries no coefficient parameters at all: the partition IS the
        # key, so there is nothing to learn about the blend. That is what makes
        # it a clean control for the modes that do learn one - the arms differ
        # by where the coefficients come from and by nothing else.

        self._share_keys: Tuple[str, ...] = tuple(
            f"activation_mix_share_{re.sub(r'[^0-9a-zA-Z]+', '_', name)}"
            for name in names
        )
        _declare_share_descriptions(names)
        # Realized gate statistics, stashed on-device during live forwards.
        self._warm: bool = False
        self._share: Optional[Tensor] = None
        self._routing: Optional[Tensor] = None

    # -- coefficients ------------------------------------------------------

    @property
    def wants_keys(self) -> bool:
        """Whether this mixture reads an external key. Callers that HAVE an
        index worth partitioning on check this rather than the mode string, so
        a config can swap `keyed` for `gated` without touching the call site."""
        return self.mode == "keyed"

    def _static_coefficients(self) -> Optional[Tensor]:
        """Scalar mixture weights, or None for the per-element types."""
        if self.mode == "convex":
            return torch.softmax(self.logits, dim=0)
        if self.mode == "affine":
            weights = self.coefficients
            return weights + (1.0 - weights.sum()) / weights.numel()
        return None

    def _gate(self, inputs: Tensor) -> Tensor:
        """Per-element coefficients read off the input VALUE, ``[..., N]``."""
        return torch.softmax(inputs.unsqueeze(-1) * self.slope + self.bias, dim=-1)

    def _partition(self, keys: Tensor) -> Tensor:
        """Per-element ONE-HOT coefficients read off an external key.

        ``keys`` is a fraction in ``[0, 1)`` saying where the element sits in
        whatever index space the caller owns, broadcastable against the input.
        The bank is cut into ``N`` equal segments and an element takes the
        branch its key lands in, so the function class is a property of the
        CALLER'S index rather than of the value or of a learned parameter.

        The contract is a fraction rather than a raw index on purpose. PEER
        would pass ``expert / num_experts``, but a head axis, a depth, a
        codebook slot or a position all normalize the same way, and none of
        them have to teach this module what their index space looks like.
        """
        count = len(self.branches)
        buckets = (keys * count).floor().long().clamp_(0, count - 1)
        return F.one_hot(buckets, count)

    def forward(
        self,
        inputs: Tensor,
        keys: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Blend the bank over ``inputs``.

        Args:
            inputs: any shape. The mixture is elementwise, so nothing here
                depends on which axis is the feature axis.
            keys: ``mix_split`` only - a fraction in ``[0, 1)`` per element,
                broadcastable against ``inputs``, saying where that element sits
                in the caller's index space. Ignored by every other type, and
                OPTIONAL: a caller with no index to partition on falls back to
                the FIRST value, which is what makes a model-wide `mix_split`
                mean "split it where there is something to split, and otherwise
                run the primary activation". Raising instead would make the
                declaration unusable anywhere but PEER; blending instead would
                quietly give every other module a function nobody asked for.
        """
        if self.mode == "keyed" and keys is None:
            # Nothing to partition on. See `keys` above.
            self._materialize_unused(inputs)
            return self.branches[0](inputs)

        weights = self._elementwise_weights(inputs, keys)
        if weights is None:
            coefficients = self._static_coefficients()
            outputs = None
            for index, branch in enumerate(self.branches):
                # Accumulated rather than stacked: a `[..., N]` stack of branch
                # outputs would be N times the activation memory for no reason,
                # and the blend is a sum either way.
                term = coefficients[index] * branch(inputs)
                outputs = term if outputs is None else outputs + term
            return outputs

        outputs = None
        for index, branch in enumerate(self.branches):
            # EVERY branch is evaluated on the whole tensor and then weighted,
            # including under `keyed`, where the weights are one-hot and the
            # selection is therefore ragged: a token's retrieved experts are an
            # arbitrary mix of segments, so there is no contiguous slice to hand
            # each branch. Gradients still reach only the selected elements.
            term = weights[..., index] * branch(inputs)
            outputs = term if outputs is None else outputs + term
        if self.training and torch.is_grad_enabled():
            self._stash(weights)
        return outputs

    def _elementwise_weights(
        self, inputs: Tensor, keys: Optional[Tensor]
    ) -> Optional[Tensor]:
        """``[..., N]`` coefficients, or None when the mode has scalar ones."""
        if self.mode == "gated":
            return self._gate(inputs)
        if self.mode == "keyed" and keys is not None:
            return self._partition(keys).to(inputs.dtype)
        return None

    def _materialize_unused(self, inputs: Tensor) -> None:
        """Give every branch one forward, once, even ones this call will not use.

        A lazily-shaped activation (Serpent and its variants) builds its
        parameters on first forward, and the keyless ``mix_split`` path only ever
        calls branch 0. A later branch would therefore still hold
        ``UninitializedParameter`` when the optimizer walked
        ``model.parameters()``, which raises - so a bank whose FIRST value is
        parameter-free and whose second is not would crash the run, at optimizer
        construction, for a config that looks entirely reasonable.

        One pass, no grad, output discarded. After that ``_warm`` is set and this
        costs a boolean.
        """
        if self._warm:
            return
        self._warm = True
        with torch.no_grad():
            for branch in self.branches[1:]:
                if getattr(branch, "has_uninitialized_params", bool)():
                    branch(inputs)

    def _stash(self, weights: Tensor) -> None:
        """Realized gate statistics, on-device (no host sync in the hot path).

        Plain detached tensors, so they hold no graph and cannot go stale across
        iterations the way accumulated ones would.
        """
        element_dims = tuple(range(weights.dim() - 1))
        detached = weights.detach()
        self._share = detached.mean(dim=element_dims)
        self._routing = (
            detached.std(dim=element_dims)
            if detached[..., 0].numel() > 1
            else torch.zeros_like(self._share)
        )

    def training_metrics(self) -> Dict[str, float]:
        with torch.no_grad():
            if self.mode in ("gated", "keyed"):
                # Realized occupancy, not a declared ratio. Under `keyed` the
                # segments are equal by construction but the KEYS are not
                # uniformly drawn - PEER retrieves experts by score - so what
                # each branch actually carries is a measurement.
                if self._share is None:
                    return {}
                shares = self._share
            else:
                shares = self._static_coefficients().detach()

            # Entropy is only defined on a distribution; the affine hull can go
            # negative, so it is reported on the absolute shares renormalized to
            # one. That is a coverage statistic ("how much of the bank is in
            # use"), which is what the chart is for, and it degenerates to the
            # true entropy whenever the coefficients happen to be non-negative.
            magnitude = shares.abs()
            total = magnitude.sum().clamp_min(1e-12)
            distribution = magnitude / total
            entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum()
            out: Dict[str, float] = {
                "activation_mix_entropy": float(entropy) / math.log(len(self.names)),
                "activation_mix_top_share": float(shares.max()),
            }
            for key, value in zip(self._share_keys, shares.tolist()):
                out[key] = float(value)
            if self._routing is not None:
                out["activation_mix_routing"] = float(self._routing.mean())
        return out

    def extra_repr(self) -> str:
        return f"type={self.type_name}, values=[{', '.join(self.names)}]"


def _declare_share_descriptions(names: Iterable[str]) -> None:
    """Register chart declarations for a bank's per-branch share keys.

    A metric with no declaration is written to the database and then dropped on
    the floor - the manifest is built from descriptions, not from columns - so
    these have to exist before the first collection. They are class-level
    because ``collect_activation_descriptions`` reads ``type(module)``.
    """
    for name in names:
        key = f"activation_mix_share_{re.sub(r'[^0-9a-zA-Z]+', '_', name)}"
        if key in ActivationMixture.metric_descriptions:
            continue
        # Declaration order across every bank, so a second bank's branches
        # append lines to the one card instead of tying with the first bank's.
        # The card's title/axis belongs to whichever branch is declared first.
        order = sum(
            k.startswith("activation_mix_share_")
            for k in ActivationMixture.metric_descriptions
        )
        leads = order == 0
        ActivationMixture.metric_descriptions[key] = {
            "description": (
                (
                    "Share of the bank carried by each branch"
                    if leads
                    else f"Share of the bank carried by `{name}`"
                )
                + ". Under `gated` it is a mean over elements, so a flat line "
                "with nonzero routing means it is used only in some input regimes."
            ),
            "chart": {
                # One card for the whole bank: the branches share a scale and
                # only mean anything against each other, and the bank grows.
                "series_group": "activation_mix_share",
                "series_label": name,
                "title": "Activation Share" if leads else None,
                "y_label": "c" if leads else None,
                "y_scale": "linear",
                "group": "activation_mix",
                "order": 100 + order,
            },
        }


# Reverse of MIXTURE_MODES, for a mixture built directly rather than through the
# registry (tests, mostly). Not a second source of truth - it is derived.
_MODE_NAMES: Dict[str, str] = {mode: name for name, mode in MIXTURE_MODES.items()}
