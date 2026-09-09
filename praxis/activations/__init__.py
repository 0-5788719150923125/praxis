"""The activation registry, and the one function that builds from it.

TWO SPELLINGS, ONE REGISTRY. An activation is declared either as a bare
registry name::

    activation: gelu

or, when the entry needs arguments, as a type plus its values::

    activation:
      type: mix_split
      values: [servant, swish]

``build_activation`` accepts both, so every consumer resolves an activation the
same way and a config says at a glance what it runs. That second form exists
because ``ActivationMixture`` (praxis/activations/mixture.py) holds a BANK of
activations, and baking one bank per registry key produced names like
``mix_harmonic`` that tell a reader nothing about what is inside them.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

from torch.nn import Module, PReLU
from transformers.activations import ACT2CLS, ClassInstantier

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

# --- mixtures of activations (praxis/activations/mixture.py) ----------------
# These are the entries that take `values`. All four are the SAME module and
# differ only in where the blending coefficients come from, which is what makes
# an ablation between any two of them one variable:
#
#   mix         learned scalars through a softmax        conv(F), arXiv:1801.09403
#   mix_affine  learned scalars, sum-to-one, signs free  aff(F), same paper
#   mix_gated   per element, read off the input VALUE    ours
#   mix_split   per element, one-hot by an EXTERNAL key  ours (was PEER's act_alt)
#
# The bank is never baked into the name: `{type: mix_split, values: [...]}` says
# what it runs, `mix_harmonic` did not.
ACTIVATION_MAP.update(
    {
        name: partial(ActivationMixture, mode=mode, type_name=name)
        for name, mode in MIXTURE_MODES.items()
    }
)

for k, v in ACTIVATION_MAP.items():
    ACT2CLS.update({k: v})

ACT2FN = ClassInstantier(ACT2CLS)
ACTIVATION_REGISTRY = dict(sorted(ACT2FN.items()))

# Names whose entry cannot stand alone: they need `values`.
TYPED_ACTIVATIONS: Tuple[str, ...] = tuple(sorted(MIXTURE_MODES))

ActivationSpec = Union[str, Mapping, Module, None]


def build_activation(spec: ActivationSpec, **kwargs: Any) -> Module:
    """Instantiate an activation from a name, a ``{type, values}`` spec, or a
    module that is already built.

    This is the ONE way to turn a config value into an activation. Going
    through ``ACT2FN[name]`` or ``ACT2CLS[name]()`` directly still works for a
    literal name, but it cannot resolve a typed spec and it mishandles the
    ``(class, kwargs)`` tuples transformers registers for a few of its own
    entries - so config-driven sites all come through here.

    Args:
        spec: a registry name (``"gelu"``), a mapping with ``type`` and
            ``values`` (``{"type": "mix_split", "values": ["servant",
            "swish"]}``), or an already-constructed module, which is returned
            unchanged so a caller can accept either.
        **kwargs: forwarded to the constructor, under anything the spec names.
    """
    if spec is None:
        raise ValueError("No activation given.")
    if isinstance(spec, Module):
        return spec

    if isinstance(spec, Mapping):
        options = dict(spec)
        name = options.pop("type", None)
        if name is None:
            raise ValueError(
                f"An activation spec needs a `type`; got keys {sorted(spec)}."
            )
        values = options.pop("values", None)
        if values is not None:
            options["activations"] = _as_specs(values)
        return _instantiate(name, **{**options, **kwargs})

    return _instantiate(spec, **kwargs)


def _as_specs(values: Union[str, Sequence[Any]]) -> Tuple[Any, ...]:
    """``values`` as a tuple of specs, from a list or a comma-separated string.

    Entries are left as they came, because a bank entry may itself be a typed
    spec - a mixture can hold a mixture, and nothing along that path needs a
    special case.
    """
    if isinstance(values, str):
        values = values.split(",")
    out = []
    for value in values:
        if isinstance(value, Mapping):
            out.append(value)
        elif str(value).strip():
            out.append(str(value).strip())
    return tuple(out)


def _instantiate(name: str, **kwargs: Any) -> Module:
    """Build registry entry ``name``, whatever shape that entry happens to be."""
    try:
        entry = ACT2CLS[name]
    except KeyError:
        raise KeyError(
            f"Unknown activation {name!r}. Known: {', '.join(sorted(ACT2CLS))}"
        ) from None
    if name in MIXTURE_MODES and "activations" not in kwargs:
        raise ValueError(
            f"Activation {name!r} is a mixture and needs a bank. Declare it as "
            f"`{{type: {name}, values: [gelu, tanh]}}` rather than as a bare name."
        )
    # transformers registers a few entries as (class, kwargs) for its
    # ClassInstantier; ours are classes or partials over one.
    if isinstance(entry, tuple):
        cls, defaults = entry
        return cls(**{**defaults, **kwargs})
    return entry(**kwargs)


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
    """Every class reachable from the registry, for model walks."""
    return tuple({activation_class(v) for v in ACT2CLS.values()})


def activation_name(cls: type) -> Optional[str]:
    """First registry key that constructs ``cls``, for a readable repr."""
    for name, entry in ACT2CLS.items():
        if activation_class(entry) is cls:
            return name
    return None
