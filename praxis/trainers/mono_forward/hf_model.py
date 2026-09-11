"""A ``PreTrainedModel`` face for Mono-Forward inference.

Under Mono-Forward the weights do not live in one module - they are sharded
across Ray actors, and a forward is a chain of hops between them. That is the
only thing about this path that differs from ordinary in-process decoding, and
wrapping it in a ``PreTrainedModel`` is what lets everything else be ordinary:
``model.generate`` drives it, so Mono-Forward gets the prepared logits
processors, the chat format's stop strings, the request deadline
(``max_time``), the speculative/CALM decoding methods and ``streamer=`` for
free, instead of a second sampling loop that has to re-implement each of them
and drift.

That second loop is what this replaces. ``MonoForwardBackend`` used to sit
beside ``ModelBackend`` in ``praxis/inference/decode_backend.py`` and carry its
own halt set, its own stop-string scan, its own deadline check and its own
sampler; the served Mono-Forward path is now the SAME ``ModelBackend`` every
other run uses, pointed at this.

The one real constraint is prefill-every-step: the actors hold no KV cache, so
every forward re-runs the whole prefix. ``prepare_inputs_for_generation``
therefore always hands back the full sequence rather than the last token, and
``use_cache`` is forced off - a cache that is never populated would otherwise
make transformers slice the input down to a single token.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from praxis import PraxisConfig

__all__ = ["MonoForwardLM"]


class MonoForwardLM(PreTrainedModel, GenerationMixin):
    """Runs one forward over the live Mono-Forward actor chain.

    Holds the trainer by reference rather than owning any weights: the actors
    are the source of truth, and Ray serializes their method calls, so an
    inference forward submitted during training queues behind any in-flight
    ``train_batch`` and reads a consistent snapshot. That is the affordance
    that makes concurrent train + infer safe, and it is unchanged here.
    """

    config_class = PraxisConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_sdpa = False
    _supports_flash_attn = False

    def __init__(self, trainer: Any, config: Optional[PraxisConfig] = None) -> None:
        config = config if config is not None else trainer._config
        super().__init__(config)
        # Deliberately NOT a submodule and NOT via a name torch would try to
        # register: the trainer owns actors and optimizers, and making it part
        # of this module's tree would put all of that into state_dict.
        object.__setattr__(self, "_mf_trainer", trainer)
        # The actors are CPU-only and this face owns no weights, so there is
        # nothing for the inherited device/dtype lookups to read - both walk
        # `parameters()` and raise StopIteration on an empty module. A
        # non-persistent buffer gives them something real without putting
        # anything into a checkpoint.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    @property
    def device(self) -> torch.device:
        """Overridden: ``PreTrainedModel.device`` reads ``parameters()`` only,
        and this module deliberately has none - the weights are on the actors."""
        return self._device_anchor.device

    @property
    def dtype(self) -> torch.dtype:
        """Overridden for the same reason as :attr:`device`."""
        return self._device_anchor.dtype

    @property
    def trainer(self) -> Any:
        return self._mf_trainer

    def can_generate(self) -> bool:
        return True

    def get_input_embeddings(self):
        # The embedding table lives on the trainer's driver copy. Returned so
        # generic transformers utilities that ask for it find something real.
        return getattr(self._mf_trainer, "_embeds", None)

    def set_input_embeddings(self, value):  # pragma: no cover - never resized
        raise NotImplementedError(
            "Mono-Forward embeddings are owned by the trainer, not this face."
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Always the FULL prefix, never a cache-shortened tail.

        The base implementation trims ``input_ids`` to the tokens a KV cache has
        not seen. There is no cache here - every forward re-runs the route from
        the first token - so trimming would silently feed the actors a
        one-token sequence and the model would generate from nothing.
        """
        return {"input_ids": input_ids, "use_cache": False}

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Hop the prefix through the actor chain and project to logits.

        ``attention_mask`` is accepted and ignored: the route builds its own
        ``block_ids`` from the prefix on every hop (EOS-aware masking has to
        reflect what is actually being run), which is the masking this stack
        uses. ``past_key_values`` is likewise accepted and ignored - see the
        module docstring on prefill-every-step.
        """
        logits = self._mf_trainer.infer_logits(input_ids)
        return CausalLMOutputWithPast(logits=logits.to(input_ids.device))
