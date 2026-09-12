"""Foreign models: training any ``transformers`` checkpoint under Praxis.

``--model-name org/repo`` loads a published model and wraps it so the Praxis
objective half (:mod:`praxis.objectives`) applies to it: the criterion and its
regularizers, the task weighter, the assistant mask, and the forward-path RL
policies. The trunk is the published one; everything trained *around* it is
Praxis's.

Two registries:

``model_tasks``
    Which ``Auto*`` class loads the checkpoint and which wrapper adapts it.
    ``causal_lm`` today; the other HuggingFace task families are entries here,
    not rewrites, because the objective layer is already split task-agnostic /
    causal.

``model_adapters``
    Per-family quirks, keyed on the HuggingFace ``model_type`` with a ``*``
    default. Discovery comes first everywhere - the wrapper reads the output
    projection, hidden size and device off the documented ``PreTrainedModel``
    surface - and this exists for the checkpoints where discovery is wrong. An
    empty adapter is the normal case.

Kwargs are passed through VERBATIM. There is no Praxis-to-foreign translation
layer and no aliasing, because the config is the model's identity: it is what
lands in ``config.json``, feeds the run hash, and tells someone later what
actually ran. An alias layer means ``depth`` and ``num_hidden_layers`` both
work, disagree in the artifacts, and eventually one of them is a lie. Praxis
architecture flags are therefore REJECTED under ``--model-name``
(:func:`reject_incompatible_flags`) rather than silently ignored.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from praxis import registry
from praxis.models.base import ForeignModel
from praxis.models.causal import ForeignCausalLM
from praxis.registry import Entry


@dataclass(frozen=True)
class ModelTask:
    """One HuggingFace task family: how to load it, and what wraps it."""

    auto_class: str  # name on the ``transformers`` namespace
    wrapper: type  # ForeignModel subclass carrying the matching objectives


registry.declare(
    "model_tasks",
    title="Foreign model tasks",
    doc=(
        "HuggingFace task families a published checkpoint can be loaded as, "
        "selected with ``--model-task``. Each entry pairs the ``Auto*`` class "
        "that loads the weights with the Praxis wrapper that supplies the "
        "objectives for that task."
    ),
    entries={
        "causal_lm": Entry(
            ModelTask("AutoModelForCausalLM", ForeignCausalLM),
            (
                "Next-token prediction. The published model supplies hidden "
                "states and logits; Praxis supplies the criterion, the "
                "regularizers, the task weighter and the forward-path RL "
                "policies."
            ),
        ),
    },
)


@dataclass(frozen=True)
class ModelAdapter:
    """Per-family corrections to what the wrapper would otherwise discover.

    Every field defaults to "discovery was right". Populate one only with
    evidence from an actual checkpoint.
    """

    # Extra kwargs merged under the user's own ``--model-kwarg`` values, for
    # families that need one to load at all (a trust flag, an attn impl).
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Module-name suffixes LoRA should target when the PEFT default is wrong.
    lora_targets: Optional[List[str]] = None
    # Jinja expression naming the assistant message content in this family's
    # chat template, when the generic rewrite cannot find it.
    assistant_content_expr: Optional[str] = None


registry.declare(
    "model_adapters",
    title="Foreign model adapters",
    doc=(
        "Per-model-family corrections, keyed on the HuggingFace ``model_type`` "
        "(``*`` is the default). Everything a foreign model needs is normally "
        "discovered from the loaded model and tokenizer; an entry here exists "
        "only where discovery is wrong for a specific family, so that a quirk "
        "is one registry entry instead of a branch in the forward pass."
    ),
    entries={
        "*": Entry(
            ModelAdapter(),
            "The default: discover everything, correct nothing.",
        ),
    },
)


def get_adapter(model_type: Optional[str]) -> ModelAdapter:
    """The adapter for a HuggingFace ``model_type``, or the discover-everything
    default."""
    namespace = registry.namespace("model_adapters")
    return namespace.get(model_type or "*", namespace["*"])


# CLI flags that live in the ``architecture`` argument group but describe the
# RUN rather than the model, so they survive ``--model-name``. Everything else
# in that group describes a trunk Praxis is not building.
#
# The test for membership is whether a FOREIGN model honors the flag. A
# regularizer is an additive loss term Praxis owns and applies to whatever
# hidden states it is handed, so it does; ``dropout`` and ``tie_weights`` name
# fields in a config the checkpoint brought and Praxis never reaches, so they
# do not - and rejecting them is the point, because ignoring them silently is
# the failure this whole path invites.
_RUN_SCOPED_ARCHITECTURE_FLAGS = frozenset(
    {
        "target_batch_size",  # optimizer-step accumulation
        "block_size",  # sequence length the data pipeline packs to
        "regularizers",  # additive loss terms, not part of any trunk
    }
)


def architecture_flag_dests(parser) -> List[str]:
    """Every ``architecture`` group destination that describes the model.

    Read off the parser rather than listed here, so an architecture flag added
    later is covered without anyone remembering to update this module.
    """
    from praxis.cli.groups import ArchitectureGroup

    for group in getattr(parser, "_action_groups", []):
        if getattr(group, "title", None) != ArchitectureGroup.name:
            continue
        return [
            action.dest
            for action in group._group_actions
            if action.dest not in _RUN_SCOPED_ARCHITECTURE_FLAGS
        ]
    return []


# Flags OUTSIDE the architecture group that a published checkpoint also
# overrides. Deliberately tiny and explicit - the architecture group is derived
# from the parser precisely so this list does not have to grow.
_ALSO_OWNED_BY_THE_CHECKPOINT = {
    "tokenizer_type": (
        "the checkpoint's own tokenizer is loaded instead; its weights mean "
        "nothing against any other vocabulary"
    ),
}


def reject_incompatible_flags(args, parser) -> None:
    """Fail the run if it asks a foreign model to be a Praxis model.

    The registry-is-the-architecture contract cannot hold for a trunk somebody
    else built, and a silently-ignored ``--attention-type`` is worse than an
    error: transformers accepts unknown kwargs and drops them, so nothing
    downstream would ever notice. Compares against the parser's defaults, which
    catches an experiment YAML setting the key just as well as a flag.
    """
    defaults = {
        action.dest: action.default
        for group in getattr(parser, "_action_groups", [])
        for action in group._group_actions
    }

    def was_set(dest):
        return hasattr(args, dest) and getattr(args, dest) != defaults.get(dest)

    offenders = {
        dest: None for dest in architecture_flag_dests(parser) if was_set(dest)
    }
    offenders.update(
        {
            dest: reason
            for dest, reason in _ALSO_OWNED_BY_THE_CHECKPOINT.items()
            if was_set(dest)
        }
    )
    if not offenders:
        return
    listed = "\n".join(
        f"  --{dest.replace('_', '-')}" + (f" ({reason})" if reason else "")
        for dest, reason in sorted(offenders.items())
    )
    raise ValueError(
        "--model-name loads a published model, so the flags that describe a "
        "Praxis-built one do not apply to it. Remove:\n"
        f"{listed}\n"
        "Run-level flags (optimizer, batch, precision, losses, RL, data, task "
        "weights) all still apply."
    )


def reject_unsupported(wrapper: type, praxis_config) -> None:
    """Fail the run if it asks for a Praxis mechanism the wrapper cannot host.

    Same reasoning as above: the failure mode a foreign-model path invites is a
    flag that quietly does nothing, so every gap is a named error.
    """
    offenders = []
    for key, reason in wrapper.UNSUPPORTED.items():
        value = getattr(praxis_config, key, None)
        if value:
            offenders.append(f"  {key}={value!r}: {reason}")
    if offenders:
        raise ValueError(f"{wrapper.__name__} cannot host:\n" + "\n".join(offenders))


def parse_model_kwargs(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """``["k=v", ...]`` as a kwargs dict, forwarded verbatim to
    ``from_pretrained``.

    Values are read as YAML so ``torch_dtype=bfloat16`` stays a string while
    ``trust_remote_code=true`` and ``num_labels=3`` arrive as the types
    transformers expects.
    """
    import yaml

    kwargs: Dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--model-kwarg expects key=value, got {pair!r}")
        kwargs[key.strip()] = yaml.safe_load(raw)
    return kwargs


def load_foreign_model(
    model_name: str,
    praxis_config,
    *,
    task: str = "causal_lm",
    revision: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> ForeignModel:
    """Load a published checkpoint and wrap it with the Praxis objectives."""
    import transformers

    spec = registry.lookup("model_tasks", task)
    reject_unsupported(spec.wrapper, praxis_config)

    auto_class = getattr(transformers, spec.auto_class)
    kwargs = dict(model_kwargs or {})
    if revision:
        kwargs.setdefault("revision", revision)

    # A first load with no adapter, so the adapter can be keyed on the config's
    # own model_type rather than on a guess parsed out of the repo name.
    config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
    adapter = get_adapter(getattr(config, "model_type", None))
    for key, value in adapter.load_kwargs.items():
        kwargs.setdefault(key, value)

    model = auto_class.from_pretrained(model_name, **kwargs)
    return spec.wrapper(model, praxis_config, model_id=model_name)


__all__ = [
    "ForeignCausalLM",
    "ForeignModel",
    "ModelAdapter",
    "ModelTask",
    "architecture_flag_dests",
    "get_adapter",
    "load_foreign_model",
    "parse_model_kwargs",
    "reject_incompatible_flags",
    "reject_unsupported",
]
