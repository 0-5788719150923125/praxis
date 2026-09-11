"""The activation registry, and the one function that builds from it.

ONE ARGUMENT, ONE SHAPE. ``--activation-type`` carries every activation choice a
model makes, so the dashboard's Arguments card shows what the run actually built
rather than one name that several modules then override. It is always a
combination TYPE over a list of VALUES::

    activation_type:
      type: mix_split
      values: [servant, swish]

and the ordinary single-activation case is that same shape with one value::

    activation_type: {type: single, values: [gelu]}
    activation_type: gelu                              # shorthand for the above

EVERY VALUE IS A GATE. ``values`` is the list of activations the gating position
draws from, and ``type`` says how they combine there - one of them (``single``),
a learned blend (``mix``, ``mix_affine``, ``mix_gated``) or a hard partition
(``mix_split``). Nothing else in the list means anything else. A gated
feedforward holds out half its up-projection to multiply against, but that half
is a STRUCTURAL choice made by the feedforward, not an activation choice, which
is why it does not appear here.

    ``type``    how the values combine. Default ``single``.
    ``values``  the activations available at the gate.
    ``linear``  optional. Activates the held-out linear half of a gated
                feedforward, which a plain GLU leaves untouched. The one thing
                here that is not a gate, and named for the half it fills.

A TYPE THAT NEEDS AN INDEX AND CANNOT GET ONE FALLS BACK TO ``values[0]``.
``mix_split`` partitions by an index its caller supplies - PEER hands it each
element's position in the expert bank - and most modules have no such index. So
a model-wide ``{type: mix_split, values: [servant, swish]}`` splits PEER's bank
and runs plain ``servant`` everywhere else, which is what makes the declaration
usable as one line instead of needing a scope.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

from torch.nn import Module, PReLU
from transformers.activations import ACT2CLS, ClassInstantier

from praxis import registry
from praxis.activations.jagged_sine import JaggedSine
from praxis.activations.mixture import MIXTURE_MODES, ActivationMixture
from praxis.activations.nmda import NMDA
from praxis.activations.ouroboros import Ouroboros
from praxis.activations.periodic_relu import PeriodicReLU
from praxis.activations.serf import SERF
from praxis.activations.serpent import Serpent
from praxis.activations.servant import Servant
from praxis.activations.sin import Sine
from praxis.activations.sin_cos import SineCosine
from praxis.activations.sinlu import SinLU
from praxis.activations.snake import Snake
from praxis.registry import Entry

ACTIVATION_MAP = dict(
    jagged_sin=JaggedSine,
    nmda=NMDA,
    ouroboros=Ouroboros,
    periodic_relu=PeriodicReLU,
    prelu=PReLU,
    serf=SERF,
    serpent=Serpent,
    servant=Servant,
    sin=Sine,
    sin_cos=SineCosine,
    sinlu=SinLU,
    snake=Snake,
)

for k, v in ACTIVATION_MAP.items():
    ACT2CLS.update({k: v})

ACT2FN = ClassInstantier(ACT2CLS)
registry.declare(
    "activations",
    dict(sorted(ACT2FN.items())),
    title="Activation functions",
    doc=(
        (
            "Pointwise nonlinearities used inside blocks and heads. Concrete "
            "nonlinearities only: how several combine at a gate is a separate axis "
            "(``activation_types``), so every entry here is something you can put in "
            "``values``. In a config file the value is a bare name or a mapping, ``{type, "
            "values, linear}``: ``type`` says how the ``values`` combine (``single``, or "
            "one of the learned mixtures), ``values`` lists activation names and may nest "
            "further specs, and ``linear`` fills the linear half of a GLU or PEER gate."
        )
    ),
)

SINGLE: str = "single"

# Each value is its own key, so a lookup returns the type's name.
registry.declare(
    "activation_types",
    title="Activation combination types",
    doc=(
        (
            "How the ``values`` of an activation combine at the gate. Declared as ``{type: "
            "<one of these>, values: [<activations>]}``. ``single`` is not a mixture at "
            "all - it is the ordinary one-activation case, written in the same shape so "
            "that every config has one shape."
        )
    ),
    entries={
        SINGLE: Entry(SINGLE, "Use the one activation in `values`. The default."),
        "mix": Entry(
            "mix",
            "Learned convex blend: softmax coefficients over the whole bank, one "
            "set for the model (conv(F), arXiv:1801.09403).",
        ),
        "mix_affine": Entry(
            "mix_affine",
            "Learned affine blend: sum-to-one with the sign constraint dropped, so "
            "it can subtract one value from another (aff(F), same paper).",
        ),
        "mix_gated": Entry(
            "mix_gated",
            "Per-element blend whose coefficients are read off the input VALUE, so "
            "the model routes between values by input regime.",
        ),
        "mix_split": Entry(
            "mix_split",
            "Hard partition by an index the caller supplies (PEER passes each "
            "element's position in the expert bank). Falls back to values[0] where "
            "there is no index.",
        ),
    },
)

ActivationSpec = Union[str, Mapping, Module, None]


def _as_spec(spec: ActivationSpec) -> Mapping:
    """Normalize any accepted spelling to ``{type, values, linear}``."""
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "values", "linear"}
        if unknown:
            raise ValueError(
                f"Unknown activation key(s) {sorted(unknown)}; an activation is "
                f"`{{type, values, linear}}`."
            )
        options = dict(spec)
        options.setdefault("type", SINGLE)
        if "values" not in options:
            raise ValueError(
                f"An activation needs `values`, e.g. "
                f"`{{type: {options['type']}, values: [gelu]}}`."
            )
        options["values"] = _as_values(options["values"])
        return options
    # A bare name is the single-activation case; keeping the shorthand is what
    # lets `activation_type: gelu` stay the thing anyone would write.
    return {"type": SINGLE, "values": (spec,)}


def _as_values(values: Union[str, Sequence[Any]]) -> Tuple[Any, ...]:
    """``values`` as a tuple, from a list or a comma-separated string.

    Entries are left as they came, because a value may itself be a spec - a
    mixture can hold a mixture, and nothing along that path needs a special case.
    """
    if isinstance(values, str):
        values = values.split(",")
    out = []
    for value in values:
        if isinstance(value, Mapping):
            out.append(value)
        elif str(value).strip():
            out.append(str(value).strip())
    if not out:
        raise ValueError("An activation needs at least one value.")
    return tuple(out)


def build_activation(spec: ActivationSpec, **kwargs: Any) -> Module:
    """Build the GATE activation named by ``spec``.

    Accepts a bare name, a ``{type, values}`` mapping, or a module that is
    already built (returned unchanged, so a caller can accept either).

    This is the ONE way to turn a config value into an activation. Going through
    ``ACT2FN[name]`` still works for a literal name, but it cannot resolve a
    typed spec and it mishandles the ``(class, kwargs)`` tuples transformers
    registers for a few of its own entries - so config-driven sites come here.
    """
    if spec is None:
        raise ValueError("No activation given.")
    if isinstance(spec, Module):
        return spec

    options = _as_spec(spec)
    name, values = options["type"], options["values"]
    if name not in registry.namespace("activation_types"):
        raise ValueError(
            f"Unknown activation type {name!r}. Known: "
            f"{', '.join(registry.namespace("activation_types"))}."
        )
    if name == SINGLE:
        if len(values) != 1:
            raise ValueError(
                f"`type: single` takes exactly one value, got {list(values)}. "
                f"Use a mixture type to combine several: "
                f"{', '.join(n for n in registry.namespace("activation_types") if n != SINGLE)}."
            )
        return _instantiate(values[0], **kwargs)
    return ActivationMixture(
        activations=values, mode=MIXTURE_MODES[name], type_name=name, **kwargs
    )


def linear_activation(spec: ActivationSpec) -> Optional[Module]:
    """The activation for a gated feedforward's held-out LINEAR half, or None.

    None is the ordinary GLU, whose linear half is exactly that. Returning None
    rather than an identity keeps "unfilled" distinguishable from "filled with
    something that does nothing", which is what lets the filled case be a clean
    one-variable arm.
    """
    if not isinstance(spec, Mapping):
        return None
    value = _as_spec(spec).get("linear")
    return build_activation(value) if value else None


def _instantiate(spec: Any, **kwargs: Any) -> Module:
    """Build one VALUE: a registry name, or a nested spec."""
    if isinstance(spec, (Mapping, Module)):
        return build_activation(spec, **kwargs)
    try:
        entry = ACT2CLS[spec]
    except KeyError:
        raise KeyError(
            f"Unknown activation {spec!r}. Known: {', '.join(sorted(ACT2CLS))}"
        ) from None
    # transformers registers a few entries as (class, kwargs) for its
    # ClassInstantier; ours are plain classes.
    if isinstance(entry, tuple):
        cls, defaults = entry
        return cls(**{**defaults, **kwargs})
    return entry(**kwargs)


def harmonic_spectrum(activation: Any) -> Optional[Tuple[Any, Any]]:
    """The Serpent-family ``(alpha, gamma)`` behind an activation slot, or None.

    Several diagnostics read a module's harmonic spectrum straight off the
    activation it was configured with - ``act.a`` and ``act.g``, Serpent's
    per-feature frequency and secondary amplitude. That reach-in assumes the slot
    holds a Serpent, which stopped being true when a slot could hold a MIXTURE:
    ``{type: mix, values: [servant, swish, relu]}`` puts an ``ActivationMixture``
    there and ``act.a`` raises ``AttributeError``.

    So the lookup goes through here instead. It finds the spectrum wherever it
    actually is - the activation itself, or the first value of a bank that has
    one - and returns None when there is none to find or it is still lazy.
    Returning None rather than raising is what the callers already expect: they
    are diagnostics, and "no spectrum" is a legitimate answer for a model
    configured without a periodic activation at all.
    """
    from torch.nn.parameter import UninitializedParameter

    candidates = [activation]
    branches = getattr(activation, "branches", None)
    if branches is not None:
        candidates.extend(branches)
    for candidate in candidates:
        if not (hasattr(candidate, "a") and hasattr(candidate, "g")):
            continue
        if any(isinstance(p, UninitializedParameter) for p in candidate.parameters()):
            return None
        return candidate.a.detach(), candidate.g.detach()
    return None


def activation_class(entry: Any) -> Any:
    """Resolve a registry entry to the class it constructs.

    Three conventions live in this map at once: a bare class (ours, mostly), a
    ``(class, kwargs)`` tuple (transformers' ``ClassInstantier``), and a
    ``functools.partial`` over a class (how typed entries bind their mode).

    Every consumer that walks a model looking for activations needs all three
    unwrapped. Missing the partial case is not a cosmetic bug: the walkers build
    an ``isinstance`` tuple out of these values, and ``isinstance`` raises
    outright when handed something that is not a type.
    """
    while isinstance(entry, partial):
        entry = entry.func
    if isinstance(entry, tuple):
        entry = entry[0]
    return entry


def activation_classes() -> Tuple[type, ...]:
    """Every class a model walk should treat as an activation.

    ``ActivationMixture`` is added explicitly because it is NOT a registry
    entry - the registry holds concrete nonlinearities and combination types are
    a separate axis. Leaving it out is not cosmetic: the dashboard's
    activation-curve probe and the metric collectors both select modules by
    ``isinstance`` against this tuple, so every mixture in the model would
    silently vanish from the charts.
    """
    return tuple({activation_class(v) for v in ACT2CLS.values()} | {ActivationMixture})


def activation_name(cls: type) -> Optional[str]:
    """First registry key that constructs ``cls``, for a readable repr."""
    for name, entry in ACT2CLS.items():
        if activation_class(entry) is cls:
            return name
    return None
