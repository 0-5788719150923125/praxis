"""Hosting a pretrained ``transformers`` model inside Praxis.

:class:`ForeignModel` is the task-agnostic half: it owns a model loaded from the
hub, keeps that model's own config as ``self.config`` (so nothing Praxis does
can write a Praxis knob into somebody else's ``config.json``), and exposes the
run's ``PraxisConfig`` separately as ``objective_config``.

Everything the wrapper needs about the hosted model is DISCOVERED - the output
projection, the hidden size, the device - through the documented
``PreTrainedModel`` surface. ``model_adapters`` (see :mod:`praxis.models`) is
the escape hatch for the families where discovery is wrong; there is no
``if model_type == ...`` anywhere in this file.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class ForeignModel(PreTrainedModel):
    """A ``PreTrainedModel`` that hosts another one.

    Subclasses add the task: they mix in the matching objective mixin from
    :mod:`praxis.objectives` and implement ``forward``.

    ``self.model`` is the hosted model, named to match every other HF wrapper
    (``base_model_prefix``) so ``save_pretrained``, parameter walks and PEFT's
    target discovery all land where they expect to.

    Subclasses MUST also inherit an objective mixin from
    :mod:`praxis.objectives`; ``__init__`` here calls its ``init_objectives``.
    """

    # The hosted model carries its own config class; ours accepts whatever it
    # brought. Praxis never constructs one of these from a bare config.
    config_class = PretrainedConfig
    base_model_prefix = "model"

    # The wrapper implements no attention of its own, so it supports whatever
    # the hosted model supports - and that model already validated its own
    # choice when it was built. Left at the PreTrainedModel defaults these
    # would veto a hosted model's sdpa/flash kernels on the wrapper's behalf.
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    # Praxis capabilities this wrapper cannot honor, as ``name -> reason``.
    # Checked once at build (see :func:`praxis.models.reject_unsupported`), so a
    # run that asks for one gets a hard error naming it rather than a flag that
    # silently does nothing - which is the whole failure mode a foreign-model
    # path invites, because transformers accepts unknown forward kwargs and
    # drops them.
    UNSUPPORTED: Dict[str, str] = {}

    # ``model_tasks`` key this wrapper implements. Set by the subclass.
    TASK: str = ""

    # Praxis config fields the HOSTED model is authoritative about, as
    # ``{praxis field: hosted config field}``.
    #
    # The foreign trunk never reads these - it built itself from its own config
    # - but Praxis DISPLAYS them, in the blueprint, the model-info panel, the
    # run spec, the annotated config download and `config.json`. Left at the
    # CLI defaults they are simply wrong: a 30-layer SmolLM2 reported
    # `depth: 2` because 2 is what `--num-layers` defaults to. A number nobody
    # reads is still a number somebody believes.
    #
    # Declared here rather than listed at the call site so a task wrapper can
    # extend it, and keyed on the names every HuggingFace config uses so
    # discovery works without per-family knowledge; `model_adapters` carries
    # the corrections for a family that names one of them differently. A field
    # the hosted config does not have is skipped, never guessed.
    CONFIG_OVERRIDES: Dict[str, str] = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        # Praxis splits the embedding width from the trunk width; a standard
        # decoder embeds straight into the residual stream, so they are one.
        "embed_size": "hidden_size",
        "max_position_embeddings": "max_position_embeddings",
        # Praxis's `depth` is recurrent passes and `num_layers` is distinct
        # blocks; a foreign stack runs each of its layers once, so both are the
        # layer count. `num_hidden_layers` is what HF's cache reads.
        "depth": "num_hidden_layers",
        "num_layers": "num_hidden_layers",
        "num_hidden_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads",
        "head_size": "head_dim",
        "activation": "hidden_act",
        "epsilon": "rms_norm_eps",
    }

    def __init__(
        self,
        model: PreTrainedModel,
        praxis_config: Any,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(model.config)
        self.model = model
        # Plain attributes, not submodules: a PretrainedConfig is not a Module,
        # and neither of these may reach state_dict or save_pretrained.
        self._objective_config = praxis_config
        self._model_id = model_id
        # Subclasses mix in an objective layer from praxis.objectives, which is
        # where init_objectives comes from. Built here so a task wrapper is
        # just a forward().
        self.init_objectives(praxis_config, encoder=None, classifier=None)

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    def reconcile_praxis_config(self, praxis_config) -> Dict[str, Any]:
        """Overwrite the Praxis fields this checkpoint is authoritative about.

        Returns ``{field: value}`` for everything it changed, so the caller can
        report it. See :attr:`CONFIG_OVERRIDES` for why this exists and what
        governs the mapping.
        """
        from praxis.models import get_adapter

        mapping = dict(self.CONFIG_OVERRIDES)
        mapping.update(
            get_adapter(getattr(self.config, "model_type", None)).config_fields
        )

        applied: Dict[str, Any] = {}
        for praxis_field, hosted_field in mapping.items():
            value = getattr(self.config, hosted_field, None)
            if value is None:
                continue  # this family does not describe itself that way
            if getattr(praxis_config, praxis_field, None) != value:
                applied[praxis_field] = value
            setattr(praxis_config, praxis_field, value)
        return applied

    @property
    def objective_config(self):
        """The run's ``PraxisConfig`` - where the objective mixins read
        ``loss_func``, ``strategy``, ``task_weights``, ``regularizers`` and the
        rest. ``self.config`` stays the hosted model's own config."""
        return self._objective_config

    @property
    def model_id(self) -> Optional[str]:
        """The hub id this model was loaded from, as recorded at load."""
        return getattr(self, "_model_id", None)

    def extra_repr(self) -> str:
        # The blueprint tab renders repr(model), and "LlamaForCausalLM" alone
        # does not say which checkpoint.
        return f"model_id={self.model_id!r}, task={self.TASK!r}"

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    @property
    def scorer(self) -> Optional[nn.Module]:
        """The hosted model's output projection, as the Praxis criterion means
        it: the module holding the vocabulary-facing ``weight``. Cut-CE reads
        ``scorer.weight`` and projects internally, so this has to be the real
        ``lm_head`` rather than a wrapper around it."""
        return self.model.get_output_embeddings()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self.model.get_output_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.model.set_input_embeddings(value)

    def set_output_embeddings(self, value) -> None:
        self.model.set_output_embeddings(value)

    def tie_weights(self) -> None:
        # The hosted model already tied (or did not) at load, per its own
        # config. Re-tying here from our config would be a second opinion about
        # somebody else's model.
        self.model.tie_weights()

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        """Delegate to the hosted model.

        Deliberately NOT reimplemented on the wrapper: every model ships its own
        ``prepare_inputs_for_generation``, cache class and generation config,
        and the objective half has nothing to contribute to a label-free decode.
        ``ModelBackend`` calls this and gets the model's native behaviour.
        """
        return self.model.generate(*args, **kwargs)

    @property
    def default_sampling_temperature(self) -> Optional[float]:
        """No preference; the transformers default applies."""
        return None

    # ------------------------------------------------------------------
    # praxis hooks the training loop probes for
    # ------------------------------------------------------------------

    # The no-encoder / no-trunk sentinels every Praxis consumer already guards
    # on. Set as class attributes so nothing has to special-case a foreign
    # model: `getattr(model, "encoder", None)` answers the same way it does for
    # a standalone Praxis model.
    encoder = False
    decoder = None
    classifier = None
    backward_classifier = None
    embeds = None

    def stage_warmup_anchor(self) -> int:
        """Single-stage; nothing to re-warm."""
        return -1

    def get_metrics(self) -> dict:
        """Whatever the objective half reports. The hosted model contributes
        none - it has no Praxis-instrumented modules to ask."""
        return self.objective_metrics()

    def training_metrics(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    # Set by praxis.models.peft.apply_peft; None means every weight trains.
    peft_type: Optional[str] = None

    @property
    def partial_checkpoint(self) -> bool:
        """Whether this model's checkpoints hold only part of its weights.

        True under an adapter: the frozen base is reproducible from
        ``--model-name`` and ``--model-revision``, both of which are in the run
        hash, so writing several hundred megabytes of unchanged weights beside
        every adapter save buys nothing. The trainer reads this to relax
        ``strict`` on load, since a partial checkpoint cannot satisfy it.
        """
        return self.peft_type is not None

    def state_dict(self, *args, **kwargs):
        """The full state dict, or just the adapter under PEFT.

        Filtered IN PLACE rather than by returning a new mapping: a parent
        module's ``state_dict`` hands its own ``destination`` down and then
        ignores what the child returns, so a copy would be silently discarded
        the moment this model is nested inside anything - which, under
        Lightning, it always is.
        """
        state = super().state_dict(*args, **kwargs)
        if not self.partial_checkpoint:
            return state

        prefix = kwargs.get("prefix", "")
        if not prefix and len(args) >= 2:
            prefix = args[1]  # positional (destination, prefix, keep_vars)
        trainable = {
            prefix + name for name, p in self.named_parameters() if p.requires_grad
        }
        for key in [k for k in state if k.startswith(prefix) and k not in trainable]:
            del state[key]
        return state
