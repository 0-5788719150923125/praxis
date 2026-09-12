"""Praxis models, exposed through HuggingFace's `PreTrainedModel` interface.

Two classes matter to most readers: `PraxisModel` (the backbone that turns input
ids into hidden states) and `PraxisForCausalLM` (adds the classifier + `.generate()`,
and is what `AutoModelForCausalLM.from_pretrained(...)` returns). Both assemble
themselves from the `PraxisConfig` by looking implementations up in the registries.
"""

import contextlib
import functools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from praxis import PraxisConfig, registry
from praxis.attention.cache import PraxisCache
from praxis.containers import LossContainer
from praxis.memory import MemoryBase
from praxis.migrations import rename_legacy_state_dict
from praxis.objectives import CausalObjectiveMixin, build_rl_policies
from praxis.utils import create_block_ids


@dataclass
class PraxisModelOutput(BaseModelOutputWithPast):
    """`forward`'s return type: HF's `BaseModelOutputWithPast` (hidden states +
    KV cache) extended with Praxis-specific extras - byte-latent encoder state,
    patch metadata, and any auxiliary `losses` the modules emit."""

    current_state: Optional[torch.LongTensor] = None
    h_encoder: Optional[torch.FloatTensor] = None
    patch_lengths: Optional[torch.LongTensor] = None
    patch_embeds: Optional[torch.FloatTensor] = None
    local_decoder_tokens: Optional[torch.LongTensor] = None
    token_block_ids: Optional[torch.LongTensor] = None
    losses: List[torch.LongTensor] = None


class PraxisModel(PreTrainedModel):
    """The backbone: input ids (or bytes) -> hidden states, no classifier.

    A standard HF `PreTrainedModel`, so it carries the usual `.from_pretrained` /
    `.save_pretrained` / device + dtype machinery. `__init__` reads the config and
    builds the optional encoder, embeddings, and decoder stack from the registries;
    `forward` returns a `PraxisModelOutput`. Use `PraxisForCausalLM` for generation.
    """

    config_class = PraxisConfig
    _supports_cache_class = True

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.encoder = False
        self.embeds = None
        if config.encoder_type is not None:
            self.encoder = registry.namespace("encoders").get(config.encoder_type)(
                config
            )
            # Settle the decoding path once, here, so nothing downstream has to
            # re-derive it. Encoders that offer only one mode pick it
            # themselves; a run that names a mode its encoder cannot drive
            # fails loudly at build rather than silently decoding the other way.
            self.encoder.generation_mode = self.encoder.resolve_generation_mode(
                getattr(config, "generation_mode", None)
            )
            # Encoders that name an embedding profile get their input
            # embeddings built from the registry and injected, mirroring how
            # classifiers classify encoder-declared output dims. Encoders that own
            # their embeddings (e.g. CALM) name no profile.
            profile = self.encoder.embedding_profile
            if profile:
                # A config-level `embeddings` key overrides the encoder
                # profile's default, mirroring how it overrides block_type on
                # the non-encoder path below. Encoders that own their
                # embeddings (profile None) are left alone.
                profile = getattr(config, "embeddings", None) or profile
                self.embeds = registry.lookup("embeddings", profile)(
                    config, encoder=self.encoder
                )
                self.encoder.set_embeddings(self.embeds)
        else:
            profile = getattr(config, "embeddings", None) or config.block_type
            self.embeds = registry.lookup("embeddings", profile)(config)
        self.decoder = registry.namespace("decoders").get(config.decoder_type)(config)

    @property
    def default_sampling_temperature(self):
        """Encoder-preferred sampling temperature when the caller omits one
        (None = use the generator's default). CALM's count-based sampler is
        near-random at T=1, so it returns its vote_temperature."""
        return getattr(self.encoder, "vote_temperature", None) if self.encoder else None

    def stage_warmup_anchor(self) -> int:
        """Optimizer step at which a new LR warmup should begin, or -1 if none.

        The scheduler/stage contract: a multi-stage model (e.g. CALM, whose
        trunk and classifier sit idle until the codec freezes) reports the boundary
        step here so the scheduler can re-warm the newly-activated params
        instead of slamming them with the full post-warmup LR cold. Default
        -1 = single-stage; nothing to re-warm."""
        enc = self.encoder
        if enc and hasattr(enc, "stage_warmup_anchor"):
            return int(enc.stage_warmup_anchor())
        return -1

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        current_state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        block_ids: Optional[torch.LongTensor] = None,
        row_continues: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        losses = LossContainer()
        h_encoder = None
        patch_lengths = None

        # Document boundaries for packed sequences, used to keep attention
        # inside one document (praxis/attention/core.py).
        #
        # The packer supplies these directly (MessageQueueManager.get_batch).
        # Falling back to scanning input_ids for the separator id covers
        # callers that build a batch by hand - generation, probes, tests - and
        # is skipped entirely when the active format writes no separator,
        # because then the scan would find nothing and silently return a
        # single block, which is what "no packing" means anyway.
        token_block_ids = block_ids
        if token_block_ids is None:
            sep_id = getattr(self.config, "eos_token_id", None)
            if sep_id is not None:
                token_block_ids = create_block_ids(input_ids, sep_id)

        if self.encoder:
            (
                inputs,
                h_encoder,
                patch_lengths,
                block_ids,
                encoder_loss,
                local_decoder_tokens,
            ) = self.encoder.encode(input_ids, block_ids=token_block_ids)
            losses.add_loss("encoder", encoder_loss)
        else:
            block_ids = token_block_ids
            if getattr(self.embeds, "accepts_offset", False) and isinstance(
                past_key_values, PraxisCache
            ):
                # Cached decode: input_ids is just the new suffix, so learned
                # absolute positions must continue from the cached length.
                inputs = self.embeds(input_ids, offset=past_key_values.past_length())
            else:
                inputs = self.embeds(input_ids)
            local_decoder_tokens = None

        # Suppress halting metric recording while the encoder is in its codec
        # "preflight" stage: the decoder runs, but its loop count isn't yet a
        # meaningful early-exit signal, so it shouldn't populate the Halting
        # Distribution. Restored to True once the encoder reaches pretrain.
        halting = getattr(self.decoder, "halting", None)
        if halting is not None:
            stage = (
                self.encoder.training_stage()
                if self.encoder and hasattr(self.encoder, "training_stage")
                else "pretrain"
            )
            halting.record_metrics = stage != "preflight"

        # Byte-timeline positions for the trunk. A patched sequence hands the
        # decoder one vector per PATCH, so an implicit arange makes every
        # position-indexed mechanism downstream (RoPE theta, the per-depth
        # positional zoom, ALiBi slopes) measure "patches elapsed" rather than
        # elapsed input. Under content-adaptive patching that clock's tick
        # length is data-dependent - a patch is one byte for a lone "a" and ten
        # for a long word - so equal position deltas mean unequal spans of
        # text. The exclusive prefix sum of patch_lengths is the byte offset
        # each patch starts at, which restores a uniform clock without giving
        # up content-aligned boundaries. None when there is no encoder, and the
        # encodings fall back to arange exactly as before.
        positions = None
        if patch_lengths is not None:
            positions = torch.cumsum(patch_lengths, dim=1) - patch_lengths

        # Publish the batch's row linkage to the memory modules for the duration
        # of this forward. Set unconditionally (None clears it) so a forward
        # without links can never inherit the previous one's grouping.
        MemoryBase.set_row_links(self.decoder, row_continues)

        last_hidden_state, new_key_values, new_state, losses = self.decoder(
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            block_ids,
            losses,
            labels,
            positions,
            row_continues=row_continues,
        )

        return PraxisModelOutput(
            last_hidden_state=last_hidden_state,
            past_key_values=new_key_values,
            hidden_states=None,
            attentions=None,
            current_state=new_state,
            h_encoder=h_encoder,
            patch_lengths=patch_lengths,
            patch_embeds=inputs if self.encoder else None,
            local_decoder_tokens=local_decoder_tokens,
            token_block_ids=token_block_ids,
            losses=losses,
        )

    def get_addr(self) -> None:
        """
        Log visible multiaddresses for hivemind node if available.
        """
        if self.decoder.manager:
            self.decoder.manager.get_visible_maddrs()

    def get_metrics(self) -> dict:
        """
        Get model metrics from the decoder.

        Returns:
            Dictionary of model metrics
        """
        return dict(**self.decoder.get_metrics())


class PraxisForCausalLM(PraxisModel, CausalObjectiveMixin, GenerationMixin):
    """`PraxisModel` plus a classifier and HF `GenerationMixin` (`.generate()`).

    This is the causal-LM entry point - what `AutoModelForCausalLM` loads. It wraps
    the backbone with a classifier (from the ``classifiers`` registry); `forward` returns
    next-token logits and, when `labels` are given, the training loss.
    """

    model_type = "praxis"

    def __init__(self, config: PraxisConfig):
        config.causal = True
        super().__init__(config)

        # Build the classifier, passing the encoder reference so classifiers
        # that participate in encoder-mode forward (harmonic, crystal) can
        # size their own submodules. Encoder-agnostic classifiers (forward,
        # tied) ignore the reference.
        encoder_ref = self.encoder if self.encoder else None
        classifier_type = resolve_classifier_type(
            config, has_encoder=encoder_ref is not None
        )
        classifier_cls = registry.namespace("classifiers").get(
            classifier_type, registry.lookup("classifiers", "forward")
        )
        self.classifier = classifier_cls(config, encoder=encoder_ref)

        # Loss-owning encoders that borrow the classifier as their token
        # classifier (e.g. CALM) take a reference to it; they apply it
        # internally.
        if self.encoder and hasattr(self.encoder, "set_classifier"):
            self.encoder.set_classifier(self.classifier)

        # Initialize separate backward classifier if requested
        if config.bidirectional and config.encoder_type is None:
            backward_cls = registry.namespace("classifiers").get(
                config.classifier_type, registry.lookup("classifiers", "forward")
            )
            self.backward_classifier = backward_cls(config, encoder=None)
        else:
            self.backward_classifier = None

        # The objective half (criterion, regularizers, MTP, RL policies, task
        # weighter, fold strategy) lives in praxis/objectives.py, shared with
        # the foreign-model wrappers. Built last: the criterion claims terms
        # from modules that compute their own, so they all have to exist.
        self.init_objectives(config, encoder=self.encoder, classifier=self.classifier)

        # Checkpoints written before the terms were collapsed into one
        # container carry the regularizers at the model's own top level.
        self._register_load_state_dict_pre_hook(self.criterion.migrate_regularizer_keys)
        # Checkpoints written before the heads -> classifiers rename.
        self._register_load_state_dict_pre_hook(_rename_legacy_keys)

        # Tie weights if requested
        if config.tie_word_embeddings and self.classifier is not None:
            self.tie_weights()

    def get_metrics(self) -> dict:
        metrics = super().get_metrics()
        metrics.update(self.objective_metrics())
        return metrics

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        layer_idx: Optional[int] = None,
        aux_losses: Optional[list] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sugar around :func:`praxis.losses.compute_layer_wise_loss`.

        Convenience wrapper for callers that have a live ``PraxisForCausalLM``
        reference and want to compute a single layer's local loss using
        the model's own criterion / strategy / classifier. The module-level
        :func:`compute_layer_wise_loss` stays canonical for
        framework-agnostic code paths (Ray actors, Hivemind peers,
        ``torch.distributed.rpc`` nodes); this method is purely sugar for
        in-process callers.

        Args:
            hidden_states: Post-layer activations for the layer being
                supervised.
            labels: Next-token labels, already shifted
                (``input_ids[..., 1:]``).
            layer_idx: Optional layer index - currently unused by the
                helper but accepted for future per-layer dispatching
                (different classifiers per depth, etc.).
            aux_losses: Optional list of router/controller aux losses
                to fold into the local objective via
                ``self.strategy`` (D5).
            input_ids: Optional unshifted input_ids to hand to
                cut-CE / dedup criteria.
        """
        from praxis.losses.layer_wise import compute_layer_wise_loss

        del layer_idx  # reserved for future use
        # Like MTP, this classifies with the SHARED classifier and needs its
        # ordinary gradient path; a surgical classifier's blend detaching would
        # otherwise leave the loss unable to reach anything. See
        # ParallelClassifier.undetached.
        undetach = getattr(self.classifier, "undetached", None)
        with undetach() if undetach else contextlib.nullcontext():
            return compute_layer_wise_loss(
                hidden_states=hidden_states,
                labels=labels,
                classifier=self.classifier,
                criterion=self.criterion.main,
                strategy=self.strategy,
                aux_losses=aux_losses,
                input_ids=input_ids,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        current_state: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict:
        # NB: this path is HF's generate() loop. Byte-latent models with MTP
        # never reach it - generate() resolves the speculative decoding method
        # first (praxis/inference/speculative.py) - so the encoder branch here
        # only covers encoder models decoding without MTP.
        #
        # Why the encoder cannot cache: NOT "the prefix isn't stable" (the old
        # reason, and false - the space patcher is prefix-monotone and the local
        # conv encoder is causal, so closed patches never move). The blocker is
        # units. `past_length()` counts TRUNK positions, which are patches, while
        # `input_ids` is bytes, so the suffix slice below would cut the wrong
        # amount. Caching an encoder stack means caching at patch granularity and
        # dropping the open patch each step, not slicing tokens here.
        if not use_cache or self.encoder:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        # Replace whatever HF pre-created (a bare DynamicCache) with ours.
        if not isinstance(past_key_values, PraxisCache):
            past_key_values = PraxisCache()

        # Only feed the new suffix once something is actually cached.
        # Cache-less attentions never write, so past_length() stays 0 and
        # they keep recomputing the full sequence - slower but correct.
        past_len = past_key_values.past_length()
        if past_len > 0:
            input_ids = input_ids[:, past_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "current_state": current_state,
        }

    def _has_uninitialized_params(self) -> bool:
        """True if any parameter is still a lazy UninitializedParameter.

        Used to defer the pretraining short-circuit until the lazy-init dummy
        forward has materialized every module.
        """
        from torch.nn.parameter import is_lazy

        return any(is_lazy(p) for p in self.parameters())

    def _set_pretraining_lock(self, active: bool) -> None:
        """Lock/unlock the model around an encoder's pretraining phase.

        When ``active``, only the encoder's ``pretraining_parameters()`` stay
        trainable (everything else is frozen so the optimizer leaves it alone).
        On the transition back, restore exactly what we disabled and fire the
        encoder's one-shot ``freeze_after_pretraining`` (e.g. freeze the codec).
        """
        if active:
            if getattr(self, "_pretrain_locked", False):
                return
            warm = {id(p) for p in self.encoder.pretraining_parameters()}
            disabled = []
            for p in self.parameters():
                if id(p) not in warm and p.requires_grad:
                    p.requires_grad_(False)
                    disabled.append(p)
            for p in self.encoder.pretraining_parameters():
                p.requires_grad_(True)
            self._pretrain_disabled = disabled
            self._pretrain_locked = True
        else:
            if not getattr(self, "_pretrain_locked", False):
                return
            for p in getattr(self, "_pretrain_disabled", []):
                p.requires_grad_(True)
            self._pretrain_disabled = []
            self.encoder.freeze_after_pretraining()
            self._pretrain_locked = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        current_state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rewards: Optional[torch.FloatTensor] = None,
        token_weights: Optional[torch.FloatTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        assistant_mask: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.LongTensor] = None,
        pair_ids: Optional[torch.LongTensor] = None,
        row_continues: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Unconditionally, before anything runs: regularizers that collect state
        # from other modules during the forward get to drop whatever the last
        # one left behind. `_add_auxiliary_losses` only calls them when training
        # AND labels are present, so a labels-free training forward would
        # otherwise strand live autograd graphs for a later step to consume.
        self.criterion.reset()

        # Encoder-owned self-supervised warmup (e.g. CALM's autoencoder). While
        # active, train ONLY the encoder's objective and skip the global
        # transformer + classifier entirely; the rest of the model stays locked. This
        # is what makes "train the codec first, then freeze it" real - without
        # it the transformer trains alongside the codec.
        #
        # Skipped while any parameter is still lazy: the lazy-init dummy forward
        # must run the FULL path so the transformer/classifier materialize. Locking
        # (requires_grad_) an UninitializedParameter raises; deferring also
        # leaves the trunk uninitialized. Once materialized this is a no-op.
        # torch.is_grad_enabled() guards the no-grad dummy forward in
        # initialize_lazy_modules: it runs in train() mode BEFORE the optimizer
        # is built, and on models with no lazy params it used to engage the
        # lock right there - get_optimizer then filtered out every non-codec
        # param, so stage 2 could never train (flat energy loss, zero trunk
        # grads, the energy generator's zero-init final layer frozen at 0 forever).
        if (
            self.training
            and torch.is_grad_enabled()
            and self.encoder
            and self.encoder.in_pretraining()
            and not self._has_uninitialized_params()
        ):
            self._set_pretraining_lock(True)
            return CausalLMOutputWithPast(loss=self.encoder.pretraining_loss(input_ids))
        if self.training and self.encoder:
            self._set_pretraining_lock(False)

        # Decode-length bucketing (praxis/inference/bucketing.py). Inert
        # unless a generation opened the context AND this is a label-free
        # inference forward, so training and validation never see a padded
        # row. Everything downstream of here - the encoder, the trunk, the
        # classifier, the losses - runs on the padded length and stays internally
        # consistent; only what leaves this method is trimmed back, because
        # `_sample` reads `logits[:, -1]` and that has to be the caller's last
        # real position.
        #
        # NEVER when a KV cache is in play. Padding is only inert because
        # nothing reads the pad positions - and a cache is exactly a thing that
        # reads them later, so a padded prefill would write pad K/V into the
        # cache and every subsequent step would attend to it. `past_key_values`
        # is None precisely on the full-recompute paths (the encoder branch of
        # `prepare_inputs_for_generation` returns no cache at all, and the
        # speculative loop calls `PraxisModel.forward` without one), which is
        # the same set of paths bucketing is for.
        true_len = None
        if labels is None and past_key_values is None and not self.training:
            # Imported here, not at module scope: `praxis.inference` pulls in
            # the Generator, which imports this module back. The lookup is a
            # sys.modules hit and it is behind the training guard, so a
            # training step never reaches it at all.
            from praxis.inference.bucketing import active_buckets, pad_for_decode

            if active_buckets():
                input_ids, attention_mask, true_len = pad_for_decode(
                    input_ids, attention_mask
                )

        outputs = super().forward(
            input_ids=input_ids,
            current_state=current_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            block_ids=block_ids,
            row_continues=row_continues,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

        # Under cut-CE training, full logits are never materialized; the loss
        # projects internally from embeddings + scorer.
        is_cut_ce = type(self.criterion.main).__name__ == "CutCrossEntropyLoss"
        skip_logits = is_cut_ce and self.training and labels is not None

        logits, scorer, hidden_states, backward_logits = self._compute_logits(
            outputs, input_ids, skip_logits, attention_mask
        )

        self._apply_recall_policies(
            outputs.losses,
            logits,
            labels,
            assistant_mask,
            task_type_ids,
            skip_logits,
            pair_ids,
        )
        hidden_states = self._apply_rl_policy(
            outputs.losses,
            hidden_states,
            logits,
            labels,
            rewards,
            attention_mask,
            token_weights,
        )

        loss = self._main_loss(
            outputs.losses,
            logits,
            labels,
            hidden_states,
            scorer,
            input_ids,
            backward_logits,
            task_type_ids,
            assistant_mask,
        )
        self._collect_aux_losses(
            outputs.losses,
            hidden_states,
            logits,
            labels,
            input_ids,
            attention_mask,
            skip_logits,
            assistant_mask,
            scorer,
            outputs.patch_embeds if self.encoder else None,
        )
        loss = self._finalize_loss(loss, outputs.losses, labels, hidden_states)

        if true_len is not None and torch.is_tensor(logits) and logits.dim() == 3:
            logits = logits[:, :true_len]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_logits(
        self,
        outputs: "PraxisModelOutput",
        input_ids: torch.Tensor,
        skip_logits: bool,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[nn.Module], torch.Tensor, Optional[torch.Tensor]]:
        """Turn trunk hidden states into logits.

        Returns ``(logits, scorer, hidden_states, backward_logits)``.
        Three paths: encoder-owned decode (e.g. CALM), classifier projection, or
        passthrough when the trunk already emits vocab-width states. With
        ``skip_logits`` (cut-CE training) the projection is left to the loss
        and ``logits`` stays at the embedding width.
        """
        hidden_states = outputs.last_hidden_state
        # Cached decode hands the classifier only the new suffix, so any module
        # anchored to absolute position or carrying context (the harmonic
        # field's phase, prefix mean and fast bank) needs the cache to continue
        # from - the classifier-side twin of `self.embeds(input_ids, offset=...)`.
        _bind_classifier_cache(self.classifier, outputs.past_key_values)
        try:
            return self._compute_logits_inner(
                outputs, input_ids, skip_logits, attention_mask, hidden_states
            )
        finally:
            _bind_classifier_cache(self.classifier, None)

    def _compute_logits_inner(
        self,
        outputs: "PraxisModelOutput",
        input_ids: torch.Tensor,
        skip_logits: bool,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ):
        logits = hidden_states
        scorer = None
        backward_logits = None

        if self.encoder:
            enc_logits, decoder_embeds = self.encoder.decode(
                hidden_states,
                outputs.h_encoder,
                input_ids,
                outputs.patch_lengths,
                outputs.local_decoder_tokens,
                block_ids=outputs.token_block_ids,
            )
            if enc_logits is not None:
                # Encoder owns its full output pipeline (e.g. CALM): use its
                # logits and scorer directly.
                logits = enc_logits
                scorer = self.encoder.scorer
            else:
                # Encoder produced features; the classifier classifies them -
                # the same path as standalone mode.
                if not skip_logits:
                    logits = self.classifier(
                        decoder_embeds,
                        **_classifier_mask_kwargs(self.classifier, attention_mask),
                    )
                scorer = self.classifier.scorer
            hidden_states = decoder_embeds

            # Encoders that manage their own losses (e.g. CALM) may emit
            # side-channel losses during decode(); fold them into the
            # shared loss container so the strategy can combine them.
            for key, value in self.encoder.consume_pending_losses().items():
                outputs.losses.add_loss(key, value)
        elif hidden_states.size(-1) != self.config.vocab_size:
            if not skip_logits:
                logits = self.classifier(
                    hidden_states,
                    **_classifier_mask_kwargs(self.classifier, attention_mask),
                )
            # Always keep the scorer reference (needed for cut-CE).
            scorer = self.classifier.scorer
            if self.backward_classifier is not None and not skip_logits:
                backward_logits = self.backward_classifier(hidden_states)

        return logits, scorer, hidden_states, backward_logits

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def _resolve_decoding_method(self, inputs, generation_config):
        """The decoding method this call should run, or None for ``_sample``.

        Praxis has two loops that are not next-token sampling, and both are
        registered the way transformers wants a non-standard strategy
        registered - as a *decoding method* handed to ``generate``, which then
        does all of its ordinary preparation (prompt handling, cache setup, the
        logits-processor list, the stopping-criteria list, the streamer's first
        publish) and calls ours in place of ``_sample``. They are mutually
        exclusive, so exactly one is chosen here.

        Encoder-owned decoding wins when the encoder declares one: CALM
        autoregresses over PATCHES rather than tokens, so no token-level loop
        can express it at all.
        """
        if self.encoder and not self.training:
            method = self.encoder.decoding_method(generation_config)
            if method is not None:
                return method

        # Speculative decode: token models directly; byte-latent encoders via
        # byte-level MTP (the encoder declined above, so we own the loop here).
        # Both are greedy-lossless.
        # `num_beams` is UNSET (None), not 1, on a transformers>=5 default
        # GenerationConfig - and `getattr` finds the attribute, so its own
        # default never applies. Comparing that None to 1 silently disabled
        # speculative decoding on every real call: the tests pass num_beams=1
        # explicitly, so they exercised a path production never took. `or 1`
        # reads unset/0 as single-beam, which is what both mean here.
        # Note there is no `generation_config is not None` test here any more.
        # There used to be, because the old loop read every knob off the config
        # and a None one crashed it - but transformers builds a config from
        # loose kwargs before it dispatches, so a caller passing
        # `max_new_tokens=256` instead of a config (the RL path in
        # trainers/backpropagation.py does) was silently routed through plain
        # sampling. Same shape of bug as the num_beams one below, so the guard
        # went with the reason for it.
        num_beams = getattr(generation_config, "num_beams", None) or 1
        # BATCH SIZE 1 ONLY. The speculative loop verifies ONE growing prefix:
        # `verify_prefixes_batched` builds its rows from `generated[0]` and
        # `candidates[0]`, so the batch axis there carries the n truncated
        # PREFIXES, not n sequences. Handed a real batch it reached a `.item()`
        # on a per-row tensor and died with "a Tensor with B elements cannot be
        # converted to Scalar" - which is what broke BrierLMCallback, whose
        # whole job is to generate two continuations for each of a batch of
        # prompts. Nothing about this is byte-latent or CALM specific; any MTP
        # model generating with B > 1 hit it. Defer to the standard loop
        # instead, which batches correctly.
        batch_ok = inputs is None or inputs.dim() < 2 or inputs.size(0) == 1
        spec_ok = (
            self.mtp is not None
            and not self.training
            and num_beams == 1
            and batch_ok
            and (not self.encoder or getattr(self.mtp, "byte_level", False))
        )
        if spec_ok:
            from praxis.inference.speculative import speculative_decoding

            return speculative_decoding
        return None

    def _extract_generation_mode_kwargs(
        self, custom_generate, kwargs, synced_gpus, assistant_model, streamer
    ):
        """Restore the two mode kwargs transformers drops for a CALLABLE method.

        Verified against transformers 5.2.0. The base implementation opens by
        POPPING ``tokenizer`` out of ``kwargs``, and then, for a callable
        ``custom_generate``, discards the dict it just built and rebuilds it
        from ``kwargs`` - where ``tokenizer`` no longer is. ``streamer`` is lost
        the same way for a different reason: it is compared against
        ``_sample``'s signature, which contains it, so it never counts as one of
        the method's "own" arguments.

        Losing ``streamer`` only costs the preview. Losing ``tokenizer`` is
        fatal and not to our own loop - ``_get_stopping_criteria`` needs it to
        build ``StopStringCriteria``, and RAISES without it, several steps
        before our method is ever called. So any model resolving a custom
        decoding method under a text-boundary chat format (``prose``, whose
        halting is stop-strings-only) failed every single request with "we
        could not locate a tokenizer".

        Restoring both here rather than smuggling them under private names
        keeps our decoding methods signature-compatible with ``_sample``, which
        is the whole point of registering them as decoding methods.
        """
        tokenizer = kwargs.get("tokenizer")
        mode_kwargs = super()._extract_generation_mode_kwargs(
            custom_generate, kwargs, synced_gpus, assistant_model, streamer
        )
        if callable(custom_generate):
            if tokenizer is not None:
                mode_kwargs.setdefault("tokenizer", tokenizer)
            if streamer is not None:
                mode_kwargs.setdefault("streamer", streamer)
        return mode_kwargs

    def generate(self, inputs=None, generation_config=None, streamer=None, **kwargs):
        """Generate tokens, dispatching to specialised paths when applicable.

        A thin resolver over ``GenerationMixin.generate``: it picks the decoding
        method (see :meth:`_resolve_decoding_method`) and hands it to
        transformers, which prepares everything and calls it. There is no
        second implementation of prompt handling, sampling, or halting here -
        that duplication is exactly what this replaces.
        """
        method = self._resolve_decoding_method(inputs, generation_config)
        return super().generate(
            inputs,
            generation_config=generation_config,
            streamer=streamer,
            custom_generate=method,
            **kwargs,
        )

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings module."""
        # Encoder mode keeps embeddings on the encoder side; callers needing
        # the byte table for tying use _tieable_input_weight instead.
        if self.encoder:
            return None
        if self.embeds is not None:
            # For projected embeddings, get the actual embedding layer
            if hasattr(self.embeds, "tokens"):
                return self.embeds.tokens
            return self.embeds
        return None

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embeddings module: a leaf classifier's own scorer,
        else the classifier itself."""
        if self.classifier is not None:
            if _owns_scorer(self.classifier):
                return self.classifier.scorer
            return self.classifier
        return None

    def _tieable_input_weight(self) -> Optional[torch.Tensor]:
        """Input-embedding weight to share with a tying-capable classifier.

        Standard mode exposes it via ``get_input_embeddings()``; encoder
        (byte-latent) mode keeps the byte table in the injected embedding
        module, whose ``tie_source()`` / ``weight`` surfaces it.
        """
        emb = self.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight
        embeds = getattr(self, "embeds", None)
        if embeds is not None:
            source = embeds.tie_source() if hasattr(embeds, "tie_source") else embeds
            if source is not None and hasattr(source, "weight"):
                return source.weight
        return None

    def tie_weights(self) -> None:
        """Tie the input and output embedding weights."""
        if not (self.config.tie_word_embeddings and self.classifier is not None):
            return
        weight = self._tieable_input_weight()
        if weight is None:
            return
        if hasattr(self.classifier, "embedding_weight"):
            # TiedClassifier: hold the reference; it projects internally.
            self.classifier.embedding_weight = weight
        elif _owns_scorer(self.classifier):
            scorer = self.classifier.scorer
            # Crystal stores centers (not weight); both are [vocab, dim].
            # Only share when shapes line up so a misconfig fails loud-free.
            attr = "centers" if hasattr(scorer, "centers") else "weight"
            target = getattr(scorer, attr, None)
            if target is not None and target.shape == weight.shape:
                setattr(scorer, attr, weight)

    def state_dict(self, *args, **kwargs):
        """Override to ensure only tensors are in the state dict for HuggingFace compatibility."""
        # Use destination argument to control what gets included
        destination = kwargs.get("destination", {})
        prefix = kwargs.get("prefix", "")
        keep_vars = kwargs.get("keep_vars", False)

        # Get state dict with standard PyTorch behavior
        state = super().state_dict(*args, **kwargs)

        # Filter to only include tensors and parameters
        filtered_state = {}
        for key, value in state.items():
            # Skip any _extra_state keys from get_extra_state()
            if "_extra_state" in key:
                continue
            # Skip any optimizer-related keys
            if "optimizer" in key.lower():
                continue
            # Only include actual tensors
            if isinstance(value, (torch.Tensor, torch.nn.Parameter)):
                filtered_state[key] = value

        return filtered_state


# ---------------------------------------------------------------------------
# Standalone helpers (state-light pieces of model assembly and generation)
# ---------------------------------------------------------------------------


def resolve_classifier_type(config, has_encoder: bool) -> str:
    """Pick the classifier registry key for a model.

    Standard-mode weight tying routes to the dedicated "tied" classifier -
    unless the configured one ties its own weights (crystal, and compositions
    ending in it), which keeps its type and ties itself in tie_weights().
    The flag is read off the registered class, unwrapping any
    functools.partial variant. Encoder mode always keeps the configured type.
    """
    classifier_cls = registry.namespace("classifiers").get(config.classifier_type)
    while isinstance(classifier_cls, functools.partial):
        classifier_cls = classifier_cls.func
    self_ties = bool(getattr(classifier_cls, "self_ties", False))
    if not has_encoder and config.tie_word_embeddings and not self_ties:
        return "tied"
    return config.classifier_type


@functools.lru_cache(maxsize=None)
def _classifier_accepts_mask(classifier_cls: type) -> bool:
    """Whether a classifier's ``forward`` takes an ``attention_mask`` (named or
    via ``**kwargs``). Composed classifiers (Parallel/Sequential) and the
    crystal router do; simple terminals (linear/harmonic) take only hidden
    states."""
    import inspect

    try:
        params = inspect.signature(classifier_cls.forward).parameters
    except (ValueError, TypeError):
        return True
    return any(
        p.kind is p.VAR_KEYWORD or name == "attention_mask"
        for name, p in params.items()
    )


def _bind_classifier_cache(classifier: nn.Module, cache) -> None:
    """Point every cache-aware module inside ``classifier`` at the live decode
    cache (or clear it with None). Modules opt in with ``accepts_decode_cache``
    and read ``decode_cache`` in their forward; nothing else is touched. Only a
    ``PraxisCache`` counts - a bare HF cache carries no classifier-side state
    and no trunk-consistent ``past_length``."""
    if cache is not None and not isinstance(cache, PraxisCache):
        cache = None
    for module in classifier.modules():
        if getattr(module, "accepts_decode_cache", False):
            module.decode_cache = cache


def _classifier_mask_kwargs(classifier: nn.Module, attention_mask) -> dict:
    """``{attention_mask: ...}`` only when the classifier accepts it - the mask
    reaches the crystal router for per-sequence pad-masked routing (lossless
    batched multi-token decode) without breaking classifiers that don't take
    one."""
    if attention_mask is None or not _classifier_accepts_mask(type(classifier)):
        return {}
    return {"attention_mask": attention_mask}


def _owns_scorer(classifier: nn.Module) -> bool:
    """True for a leaf classifier, whose ``scorer`` is its own attribute;
    False for a composition, whose ``scorer`` property resolves into a stage
    or arm (or, for TiedClassifier, wraps the embedding matrix)."""
    return not isinstance(getattr(type(classifier), "scorer", None), property)


def _rename_legacy_keys(state_dict, prefix, *args) -> None:
    """Load-state-dict pre-hook: translate pre-rename parameter paths."""
    rename_legacy_state_dict(state_dict, prefix)


def sample_token(
    logits: torch.Tensor, do_sample: bool, temperature: float
) -> torch.Tensor:
    """Sample or greedily select a single token from logits."""
    if do_sample and temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    return logits.argmax(dim=-1)
