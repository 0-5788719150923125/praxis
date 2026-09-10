"""Praxis models, exposed through HuggingFace's `PreTrainedModel` interface.

Two classes matter to most readers: `PraxisModel` (the backbone that turns input
ids into hidden states) and `PraxisForCausalLM` (adds the LM head + `.generate()`,
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

from praxis import DECODER_REGISTRY, EMBEDDING_REGISTRY, ENCODER_REGISTRY, PraxisConfig
from praxis.attention.cache import PraxisCache
from praxis.containers import LossContainer
from praxis.heads import HEAD_REGISTRY
from praxis.losses import build_objectives
from praxis.losses.conflict import ObjectiveConflict
from praxis.policies import RL_POLICIES_REGISTRY
from praxis.strategies import STRATEGIES_REGISTRY
from praxis.tasks import TASK_NAMES, resolve_task_weighter
from praxis.memory import MemoryBase
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
    """The backbone: input ids (or bytes) -> hidden states, no LM head.

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
            self.encoder = ENCODER_REGISTRY.get(config.encoder_type)(config)
            # Settle the decoding path once, here, so nothing downstream has to
            # re-derive it. Encoders that offer only one mode pick it
            # themselves; a run that names a mode its encoder cannot drive
            # fails loudly at build rather than silently decoding the other way.
            self.encoder.generation_mode = self.encoder.resolve_generation_mode(
                getattr(config, "generation_mode", None)
            )
            # Encoders that name an embedding profile get their input
            # embeddings built from the registry and injected, mirroring how
            # heads classify encoder-declared output dims. Encoders that own
            # their embeddings (e.g. CALM) name no profile.
            profile = self.encoder.embedding_profile
            if profile:
                # A config-level `embeddings` key overrides the encoder
                # profile's default, mirroring how it overrides block_type on
                # the non-encoder path below. Encoders that own their
                # embeddings (profile None) are left alone.
                profile = getattr(config, "embeddings", None) or profile
                self.embeds = EMBEDDING_REGISTRY[profile](config, encoder=self.encoder)
                self.encoder.set_embeddings(self.embeds)
        else:
            profile = getattr(config, "embeddings", None) or config.block_type
            self.embeds = EMBEDDING_REGISTRY[profile](config)
        self.decoder = DECODER_REGISTRY.get(config.decoder_type)(config)

    @property
    def default_sampling_temperature(self):
        """Encoder-preferred sampling temperature when the caller omits one
        (None = use the generator's default). CALM's count-based sampler is
        near-random at T=1, so it returns its vote_temperature."""
        return getattr(self.encoder, "vote_temperature", None) if self.encoder else None

    def stage_warmup_anchor(self) -> int:
        """Optimizer step at which a new LR warmup should begin, or -1 if none.

        The scheduler/stage contract: a multi-stage model (e.g. CALM, whose
        trunk and head sit idle until the codec freezes) reports the boundary
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


class PraxisForCausalLM(PraxisModel, GenerationMixin):
    """`PraxisModel` plus an LM head and HF `GenerationMixin` (`.generate()`).

    This is the causal-LM entry point - what `AutoModelForCausalLM` loads. It wraps
    the backbone with an output head (from `HEAD_REGISTRY`); `forward` returns
    next-token logits and, when `labels` are given, the training loss.
    """

    model_type = "praxis"

    def __init__(self, config: PraxisConfig):
        config.causal = True
        super().__init__(config)

        # Build the LM head, passing the encoder reference so heads that
        # participate in encoder-mode forward (harmonic, crystal) can
        # size their own submodules. Encoder-agnostic heads (forward,
        # tied) ignore the reference and skip allocating an lm_head.
        encoder_ref = self.encoder if self.encoder else None
        head_type = resolve_head_type(config, has_encoder=encoder_ref is not None)
        head_cls = HEAD_REGISTRY.get(head_type, HEAD_REGISTRY["forward"])
        self.head = head_cls(config, encoder=encoder_ref)

        # Loss-owning encoders that borrow the head as their token classifier
        # (e.g. CALM) take a reference to it; they apply it internally.
        if self.encoder and hasattr(self.encoder, "set_head"):
            self.encoder.set_head(self.head)

        # Initialize separate backward head if requested
        if config.bidirectional and config.encoder_type is None:
            backward_cls = HEAD_REGISTRY.get(config.head_type, HEAD_REGISTRY["forward"])
            self.backward_head = backward_cls(config, encoder=None)
        else:
            self.backward_head = None

        # Initialize MTP if requested
        # Two execution paths:
        #   Standard (token-level): embeds from nn.Embedding, CE loss vs token IDs
        #   Encoder (patch-level): patch embeds projected to embed_size, MSE loss
        #     vs target patch representations — the patcher acts as the "tokenizer"
        #     and patch positions in the global transformer are the prediction targets.
        self.mtp = None
        if getattr(config, "mtp_type", None) is not None:
            if config.bidirectional:
                raise ValueError("MTP cannot be combined with --bidirectional")
            from praxis.heads.mtp import MultiTokenPrediction

            self.mtp = MultiTokenPrediction(config)

        # Forward-path RL policies (see build_rl_policies). Weight controllers
        # are built by training callbacks, never here.
        self.policy, self.policy_type, _recall = build_rl_policies(config)
        self.policies = nn.ModuleDict(_recall)
        self._engagement_metrics = {}  # latest recall-policy scalars

        # Every loss this model can add to its objective, in one container:
        # the main criterion, the additive representation-shaping
        # regularizers, and the terms other modules compute (registered just
        # below). An encoder that owns the loss (CALM) bypasses the main-CE
        # path entirely, so `main` stays unregistered there.
        self.criterion = build_objectives(config, self.encoder)
        # A module that computes its own term declares it via ``objectives()``
        # and reads it back out of the container; collected once here, so the
        # blueprint is complete before a step runs.
        self.criterion.claim(self)

        # Checkpoints written before the terms were collapsed into one
        # container carry the regularizers at the model's own top level.
        self._register_load_state_dict_pre_hook(self.criterion.migrate_regularizer_keys)

        # Per-task loss weighting. Identity (no-op) unless --task-weights
        # is set; the assistant mask from the chat template is always
        # applied when present. Learnable variants expose an anchor_loss
        # that gets folded into the combined objective by the strategy below.
        self.tasker = resolve_task_weighter(config.task_weights)

        # Set by the trainer to the task indices a live dataset produces;
        # get_metrics() uses it to skip charting weights for absent tasks.
        self.active_task_ids = None

        # The strategy for combining multiple losses into a single scalar objective.
        self.strategy = STRATEGIES_REGISTRY.get(config.strategy, "naive")()
        # Do the model's several objectives agree about the shared trunk? One
        # sampled cosine per loss term; see praxis/losses/conflict.py for why
        # this is the measurement and not the gradient-surgery method itself.
        self._conflict = ObjectiveConflict()
        self._conflict_metrics: Dict[str, float] = {}
        # Per-arm Jacobian diagnostics, stashed by _collect_aux_losses.
        self._arm_metrics: Dict[str, float] = {}
        # A head that owns its arms' objectives takes the HALO geometric term
        # as that arm's Jacobian row, so the criterion must stop also adding it
        # - otherwise it is double-counted AND reaches the trunk uncorrected,
        # bypassing the arbitration it is supposed to be subject to.
        if getattr(self.head, "arm_surgery", False) and hasattr(
            self.criterion.main, "composite_geometry"
        ):
            self.criterion.main.composite_geometry = False

        # Tie weights if requested
        if config.tie_word_embeddings and self.head is not None:
            self.tie_weights()

    def get_metrics(self) -> dict:
        metrics = super().get_metrics()
        # Surface dynamic task weights (learnable or difficulty-EMA) so
        # runs can see them drift. Fixed weighters are skipped; tasks with
        # no live dataset are skipped when active_task_ids is set.
        if getattr(self.tasker, "is_dynamic", False):
            eff = self.tasker.effective_weights().cpu().tolist()
            for idx, (name, value) in enumerate(zip(TASK_NAMES, eff)):
                if self.active_task_ids is not None and idx not in self.active_task_ids:
                    continue
                metrics[f"task_weight_{name}"] = float(value)
        # Engagement policy scalars (energy, activation rate, recall, advantage).
        if self._engagement_metrics:
            metrics.update(self._engagement_metrics)
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
        the model's own criterion / strategy / head. The module-level
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
                (different heads per depth, etc.).
            aux_losses: Optional list of router/controller aux losses
                to fold into the local objective via
                ``self.strategy`` (D5).
            input_ids: Optional unshifted input_ids to hand to
                cut-CE / dedup criteria.
        """
        from praxis.losses.layer_wise import compute_layer_wise_loss

        del layer_idx  # reserved for future use
        # Like MTP, this classifies with the SHARED head and needs its ordinary
        # gradient path; a surgical head's blend detaching would otherwise leave
        # the loss unable to reach anything. See ParallelHead.undetached.
        undetach = getattr(self.head, "undetached", None)
        with undetach() if undetach else contextlib.nullcontext():
            return compute_layer_wise_loss(
                hidden_states=hidden_states,
                labels=labels,
                head=self.head,
                criterion=self.criterion.main,
                strategy=self.strategy,
                aux_losses=aux_losses,
                input_ids=input_ids,
            )

    def _build_loss_weights(
        self,
        labels: torch.Tensor,
        task_type_ids: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compose per-token weights from task IDs and the assistant mask.

        Returns None when neither signal is provided. The returned tensor
        is shifted to align with ``labels`` -- positions in
        ``input_ids[t+1]`` set the weight on the loss for the prediction
        at ``labels[t]``.
        """
        # Honor --no-mask-prompts: drop the assistant_mask before composing
        # so every token contributes to the loss. Task weights still apply.
        if getattr(self.config, "no_mask_prompts", False):
            assistant_mask = None

        if task_type_ids is None and assistant_mask is None:
            return None

        # Both inputs cover input_ids positions [0..T-1]. Labels are usually
        # the trailing T-1 tokens, so the weight aligned to label[t] comes
        # from position t+1. When labels is full-length (e.g. aligned encoder),
        # there's no shift to do.
        target_len = labels.size(-1)

        weights = None
        if task_type_ids is not None:
            shifted_task = task_type_ids[..., -target_len:]
            weights = self.tasker(shifted_task.long())
            # Preference contrast material is never imitated: rejected-tagged
            # tokens are excluded from the main CE regardless of weighter
            # profile - the preference policy's margin is their only training
            # signal (the hh-rlhf card's contract).
            from praxis.tasks import TaskType

            weights = weights * (shifted_task != int(TaskType.PREF_REJECTED)).to(
                weights.dtype
            )

        if assistant_mask is not None:
            shifted_mask = assistant_mask[..., -target_len:].to(
                weights.dtype if weights is not None else torch.float32
            )
            weights = shifted_mask if weights is None else weights * shifted_mask

        return weights

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        classifier: Optional[nn.Module],
        input_ids: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the main loss using the criterion."""
        # cut_cross_entropy needs FULL UNSHIFTED embeddings to avoid materializing shifted tensors
        # Check by class name to avoid hard dependency on integration
        is_cut_ce = type(self.criterion.main).__name__ == "CutCrossEntropyLoss"

        # Check if encoder outputs are already aligned
        if self.encoder and self.encoder.outputs_are_aligned:
            return self.criterion.main(
                logits=logits.contiguous(),
                embeddings=embeddings if is_cut_ce else embeddings,
                classifier=classifier,
                labels=labels,
                input_ids=input_ids,
                loss_weights=loss_weights,
            )
        else:
            return self.criterion.main(
                logits=logits[..., :-1, :].contiguous(),
                embeddings=(
                    embeddings if is_cut_ce else embeddings[..., :-1, :].contiguous()
                ),
                classifier=classifier,
                labels=labels,
                input_ids=input_ids,
                loss_weights=loss_weights,
            )

    def _compute_bidirectional_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        classifier: Optional[nn.Module],
        input_ids: torch.Tensor,
        backward_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute bidirectional loss (forward and backward prediction).

        Note: labels are already shifted right (input_ids[..., 1:]) when passed in.
        """
        # Forward loss: predict next token (standard causal LM)
        forward_loss = self._compute_loss(
            logits, labels, embeddings, classifier, input_ids
        )

        # Backward loss: predict previous token
        # For backward prediction, we want logits[1:] to predict input_ids[:-1]
        backward_labels = input_ids[..., :-1].contiguous()

        # Select appropriate logits and classifier for backward prediction
        if backward_logits is not None:
            # Use separate backward head
            back_logits = backward_logits[..., 1:, :].contiguous()
            back_classifier = (
                self.backward_head.classifier
                if hasattr(self.backward_head, "classifier")
                else None
            )
        else:
            # Reuse forward head
            back_logits = logits[..., 1:, :].contiguous()
            back_classifier = classifier

        # Compute backward loss
        backward_loss = self.criterion.main(
            logits=back_logits,
            embeddings=embeddings[..., 1:, :].contiguous(),
            classifier=back_classifier,
            labels=backward_labels,
            input_ids=input_ids,
        )

        # Weighted combination based on forward_weight
        forward_weight = self.config.forward_weight
        backward_weight = 1.0 - forward_weight

        return forward_weight * forward_loss + backward_weight * backward_loss

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
        # first (praxis/generation/speculative.py) - so the encoder branch here
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
        # transformer + head entirely; the rest of the model stays locked. This
        # is what makes "train the codec first, then freeze it" real - without
        # it the transformer trains alongside the codec.
        #
        # Skipped while any parameter is still lazy: the lazy-init dummy forward
        # must run the FULL path so the transformer/head materialize. Locking
        # (requires_grad_) an UninitializedParameter raises; deferring also
        # leaves the trunk uninitialized. Once materialized this is a no-op.
        # torch.is_grad_enabled() guards the no-grad dummy forward in
        # initialize_lazy_modules: it runs in train() mode BEFORE the optimizer
        # is built, and on models with no lazy params it used to engage the
        # lock right there - get_optimizer then filtered out every non-codec
        # param, so stage 2 could never train (flat energy loss, zero trunk
        # grads, the energy head's zero-init final layer frozen at 0 forever).
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

        # Decode-length bucketing (praxis/generation/bucketing.py). Inert
        # unless a generation opened the context AND this is a label-free
        # inference forward, so training and validation never see a padded
        # row. Everything downstream of here - the encoder, the trunk, the
        # head, the losses - runs on the padded length and stays internally
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
            # Imported here, not at module scope: `praxis.generation` pulls in
            # the Generator, which imports this module back. The lookup is a
            # sys.modules hit and it is behind the training guard, so a
            # training step never reaches it at all.
            from praxis.generation.bucketing import active_buckets, pad_for_decode

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
        # projects internally from embeddings + classifier.
        is_cut_ce = type(self.criterion.main).__name__ == "CutCrossEntropyLoss"
        skip_logits = is_cut_ce and self.training and labels is not None

        logits, classifier, hidden_states, backward_logits = self._compute_logits(
            outputs, input_ids, skip_logits, attention_mask
        )

        self._apply_recall_policies(
            outputs.losses, logits, labels, assistant_mask, task_type_ids, skip_logits
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
            classifier,
            input_ids,
            backward_logits,
            task_type_ids,
            assistant_mask,
        )
        self._collect_aux_losses(
            outputs,
            hidden_states,
            logits,
            labels,
            input_ids,
            attention_mask,
            skip_logits,
            assistant_mask,
            classifier,
        )
        loss = self._finalize_loss(loss, outputs.losses, labels)

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

        Returns ``(logits, classifier, hidden_states, backward_logits)``.
        Three paths: encoder-owned decode (e.g. CALM), head projection, or
        passthrough when the trunk already emits vocab-width states. With
        ``skip_logits`` (cut-CE training) the projection is left to the loss
        and ``logits`` stays at the embedding width.
        """
        hidden_states = outputs.last_hidden_state
        # Cached decode hands the head only the new suffix, so any head module
        # anchored to absolute position or carrying context (the harmonic
        # field's phase, prefix mean and fast bank) needs the cache to continue
        # from - the head-side twin of `self.embeds(input_ids, offset=...)`.
        _bind_head_cache(self.head, outputs.past_key_values)
        try:
            return self._compute_logits_inner(
                outputs, input_ids, skip_logits, attention_mask, hidden_states
            )
        finally:
            _bind_head_cache(self.head, None)

    def _compute_logits_inner(
        self,
        outputs: "PraxisModelOutput",
        input_ids: torch.Tensor,
        skip_logits: bool,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ):
        logits = hidden_states
        classifier = None
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
                # logits and classifier directly.
                logits = enc_logits
                classifier = self.encoder.classifier
            else:
                # Encoder produced features; the head classifies them - the
                # same path as standalone mode.
                if not skip_logits:
                    logits = self.head(
                        decoder_embeds,
                        **_head_mask_kwargs(self.head, attention_mask),
                    )
                classifier = self.head.classifier
            hidden_states = decoder_embeds

            # Encoders that manage their own losses (e.g. CALM) may emit
            # side-channel losses during decode(); fold them into the
            # shared loss container so the strategy can combine them.
            for key, value in self.encoder.consume_pending_losses().items():
                outputs.losses.add_loss(key, value)
        elif hidden_states.size(-1) != self.config.vocab_size:
            if not skip_logits:
                logits = self.head(
                    hidden_states, **_head_mask_kwargs(self.head, attention_mask)
                )
            # Always keep the classifier reference (needed for cut-CE).
            classifier = self.head.classifier
            if self.backward_head is not None and not skip_logits:
                backward_logits = self.backward_head(hidden_states)

        return logits, classifier, hidden_states, backward_logits

    def _apply_recall_policies(
        self,
        losses: LossContainer,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
        task_type_ids: Optional[torch.Tensor],
        skip_logits: bool,
    ) -> None:
        """Recall-style forward policies (engagement / joke): each computes its
        own reward from the answer labels over the assistant region. Any number
        may coexist; each emits its own namespaced loss and metrics."""
        if not self.policies or labels is None:
            return
        self._engagement_metrics = {}
        for name, pol in self.policies.items():
            pol_loss, pol_metrics = pol(
                logits=logits if not skip_logits else None,
                labels=labels,
                assistant_mask=assistant_mask,
                task_type_ids=task_type_ids,
            )
            if pol_loss is not None:
                losses.add_loss(f"{name}_policy", pol_loss)
            if pol_metrics:
                self._engagement_metrics.update(pol_metrics)

    def _apply_rl_policy(
        self,
        losses: LossContainer,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        rewards: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the single (non-recall) forward-path RL policy.

        Returns ``hidden_states``, which REINFORCE-style policies may modify.
        """
        if self.policy is None:
            return hidden_states
        rl_type = self.policy_type

        if rl_type == "grpo" and rewards is not None and labels is not None:
            # GRPO needs logits and labels for proper loss computation.
            _, rl_losses = self.policy(
                hidden_states,
                logits=logits,
                labels=labels,
                rewards=rewards,
                ref_logits=None,  # TODO: Add reference model support
                mask=attention_mask,
            )
            if rl_losses is not None:
                for key, value in rl_losses.items():
                    losses.add_loss(f"rl_{key}", value)
        elif rl_type == "cot" and labels is not None:
            # Basic CoT uses supervised learning with weighted loss; logits
            # are shifted to match labels.
            _, cot_losses = self.policy(
                hidden_states,
                logits=logits[..., :-1, :].contiguous(),
                labels=labels,
                attention_mask=attention_mask,
                token_weights=token_weights,
            )
            if cot_losses is not None:
                losses.add_loss_container(cot_losses)
        elif rewards is not None and labels is not None:
            # REINFORCE and other methods.
            hidden_states, rl_loss = self.policy(
                hidden_states, rewards=rewards, mask=attention_mask
            )
            if rl_loss is not None:
                losses.add_loss("rl_policy", rl_loss)

        return hidden_states

    def _main_loss(
        self,
        losses: LossContainer,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        classifier: Optional[nn.Module],
        input_ids: torch.Tensor,
        backward_logits: Optional[torch.Tensor],
        task_type_ids: Optional[torch.Tensor],
        assistant_mask: Optional[torch.Tensor],
    ):
        """Register the main objective and return it (0 when none applies:
        no labels, a layer-wise trainer without layer losses, or an encoder
        that owns its loss - those are combined later in _finalize_loss)."""
        if labels is None:
            return 0
        loss_weights = self._build_loss_weights(
            labels=labels,
            task_type_ids=task_type_ids,
            assistant_mask=assistant_mask,
        )
        # Layer-wise trainers (e.g. MonoForward) already trained each layer;
        # just combine their recorded losses.
        if "_layer_wise_complete" in losses.loss_dict:
            layer_losses = [
                v
                for k, v in losses.loss_dict.items()
                if k != "_layer_wise_complete" and k != "main"
            ]
            return self.strategy(layer_losses) if layer_losses else 0
        if self.encoder and self.encoder.handles_loss:
            # Encoder owns its loss bookkeeping (see CALMEncoder); its
            # registered losses are combined in _finalize_loss.
            return 0
        if self.config.bidirectional:
            main_loss = self._compute_bidirectional_loss(
                logits=logits,
                labels=labels,
                embeddings=hidden_states,
                classifier=classifier,
                input_ids=input_ids,
                backward_logits=backward_logits,
            )
            return losses.add_loss("main", main_loss)
        main_loss = self._compute_loss(
            logits=logits,
            labels=labels,
            embeddings=hidden_states,
            classifier=classifier,
            input_ids=input_ids,
            loss_weights=loss_weights,
        )
        return losses.add_loss("main", main_loss)

    def _collect_aux_losses(
        self,
        outputs: "PraxisModelOutput",
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        skip_logits: bool,
        assistant_mask: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
    ) -> None:
        """Accumulate training-only auxiliary losses into the container:
        task-weight anchor, head aux losses, MTP, and the regularizers."""
        if not self.training or labels is None:
            return

        # Task-weight anchor loss (learnable weighters only).
        anchor = self.tasker.anchor_loss()
        if anchor is not None:
            outputs.losses.add_loss("task_weight_anchor", anchor)

        # Heads can emit named aux losses (e.g., HarmonicHead's forward-shift
        # smoothness loss, CrystalHead's centers-RMS regularizer).
        if self.head is not None:
            for name, value in self.head.aux_losses().items():
                outputs.losses.add_loss(name, value)

        # Multi-arm heads get the labels here rather than in forward, because
        # this is the first place both the labels and the head's own input
        # exist. `hidden_states` is exactly what the head classified.
        #   arm_conflict()   - diagnostic, sampled, on every ParallelHead
        #   arm_objectives() - per-arm CE + the PCGrad trunk gradient, empty
        #                      unless the profile opts in (prismatic9)
        if self.head is not None and hasattr(self.head, "arm_objectives"):
            self._arm_metrics = self.head.arm_conflict(
                hidden_states, labels, self.criterion
            )
            for name, value in self.head.arm_objectives(
                hidden_states, labels, self.criterion
            ).items():
                outputs.losses.add_loss(name, value)

        # Router aux losses (e.g. VEAR's parameter-only repulsion), collected once
        # per step here rather than through the per-forward aux return: a
        # parameter-only loss escaping the gradient-checkpointed recurrent forward
        # causes a double-backward.
        if self.decoder is not None and hasattr(self.decoder, "router_aux_losses"):
            for name, value in self.decoder.router_aux_losses().items():
                outputs.losses.add_loss(name, value)

        if self.mtp is not None:
            # Byte-level MTP embeds byte IDs through the encoder's byte table
            # (get_input_embeddings() is None in encoder mode); the patch path
            # never touches embed_fn.
            mtp_embed_fn = (
                self.embeds
                if getattr(self.mtp, "byte_level", False)
                else self.get_input_embeddings()
            )
            # MTP consumes UNDETACHED hidden states and the shared head, so its
            # loss trains the trunk and the head like any other objective. Left
            # unweighted it does that at every position, including the ones
            # assistant_mask zeroes - prompt text would keep shaping the trunk
            # through the auxiliary path no matter what the mask said. Only the
            # prompt mask is passed on: task weights are the main objective's
            # difficulty curriculum and coupling the draft head to the tasker
            # would be a separate decision.
            mtp_weights = (
                None
                if getattr(self.config, "no_mask_prompts", False)
                else assistant_mask
            )
            # MTP classifies its draft states with the SHARED head, so it needs
            # that head's ordinary gradient path. A surgical head (prismatic9)
            # detaches every arm in the blend so the main CE trains only the
            # gate - and detaching an arm's output severs the route back to the
            # head's input, which is where MTP's draft states enter. Left alone,
            # MTP's loss reaches nothing: 0 of 9 MTP parameters received a
            # gradient under prismatic9 against 9 of 9 under prismatic8, and
            # every mtp_field_* series sat frozen at its initialization.
            undetach = getattr(self.head, "undetached", None)
            with undetach() if undetach else contextlib.nullcontext():
                mtp_inputs = self.mtp.prepare_inputs(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    embed_fn=mtp_embed_fn,
                    head=self.head,
                    patch_embeds=outputs.patch_embeds if self.encoder else None,
                    loss_weights=mtp_weights,
                )
                outputs.losses.add_loss_container(self.mtp(mtp_inputs))

        # Additive representation-shaping regularizers (REGULARIZER_REGISTRY).
        # `classifier` is passed as optional context, not stored: a regularizer
        # holding a reference to the readout would register it a second time and
        # duplicate its parameters in state_dict and the optimizer.
        for reg in self.criterion.regularizers():
            outputs.losses.add_loss(
                reg.name, reg(hidden_states, input_ids, classifier=classifier)
            )

        # Last, so the sampler sees every objective this step actually carries.
        # Stashed rather than returned: the dynamics callback drains it on its
        # own cadence, the same contract the governor and compute profiler use.
        self._conflict_metrics = self._conflict.measure(
            outputs.losses.loss_dict, hidden_states
        )

    def _finalize_loss(
        self, loss, losses: LossContainer, labels: Optional[torch.Tensor]
    ):
        """Combine all tagged losses via the strategy.

        Auxiliary losses are omitted during validation and inference - except
        for handles_loss encoders (CALM), where the encoder owns the main loss
        and there is nothing else to fall back to (their val_loss would
        otherwise stay at 0).
        """
        handles_loss_encoder = self.encoder is not False and getattr(
            self.encoder, "handles_loss", False
        )
        if labels is None or not (self.training or handles_loss_encoder):
            return loss
        loss_values = losses.get_loss_values()
        if len(loss_values) > 1:
            return self.strategy(loss_values)
        if loss == 0 and len(loss_values) > 0:
            # Only auxiliary losses (no main) - combine via strategy.
            return self.strategy(loss_values)
        return loss

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
            from praxis.generation.speculative import speculative_decoding

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
        """Get the output embeddings (lm_head) module."""
        if self.head is not None:
            if hasattr(self.head, "lm_head"):
                return self.head.lm_head
            return self.head
        return None

    def _tieable_input_weight(self) -> Optional[torch.Tensor]:
        """Input-embedding weight to share with a tying-capable output head.

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
        if not (self.config.tie_word_embeddings and self.head is not None):
            return
        weight = self._tieable_input_weight()
        if weight is None:
            return
        if hasattr(self.head, "embedding_weight"):
            # TiedWeights head: hold the reference; it projects internally.
            self.head.embedding_weight = weight
        elif hasattr(self.head, "lm_head"):
            lm = self.head.lm_head
            # Crystal stores centers (not weight); both are [vocab, dim].
            # Only share when shapes line up so a misconfig fails loud-free.
            attr = "centers" if hasattr(lm, "centers") else "weight"
            target = getattr(lm, attr, None)
            if target is not None and target.shape == weight.shape:
                setattr(lm, attr, weight)

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


def resolve_head_type(config, has_encoder: bool) -> str:
    """Pick the head registry key for a model.

    Standard-mode weight tying routes to the dedicated "tied" head - unless
    the configured head ties its own weights (crystal, and compositions
    ending in it), which keeps its type and ties itself in tie_weights().
    The flag is read off the registered class, unwrapping any
    functools.partial variant. Encoder mode always keeps the configured type.
    """
    head_cls = HEAD_REGISTRY.get(config.head_type)
    while isinstance(head_cls, functools.partial):
        head_cls = head_cls.func
    self_ties = bool(getattr(head_cls, "self_ties", False))
    if not has_encoder and config.tie_word_embeddings and not self_ties:
        return "tied"
    return config.head_type


def build_rl_policies(config):
    """Construct forward-path RL policies from ``config.rl_type``.

    rl_type is a list of policy/profile keys, so multiple discrete RL tasks
    coexist. Returns ``(policy, policy_type, recall_policies)``:

    - recall-style policies (engagement, joke) share a (logits, labels, mask)
      signature and compute their own reward; any number may coexist, one per
      RL interface with distinct metrics.
    - all others (reinforce/grpo/cot) are mutually exclusive (different
      signatures, may modify hidden states) - the single ``policy``.

    Weight controllers act from a training callback and are never built here.
    """
    from praxis.policies import get_rl_profile, normalize_rl_types

    policy = None
    policy_type = None
    recall = {}
    for rl_name in normalize_rl_types(getattr(config, "rl_type", None)):
        profile = get_rl_profile(rl_name)
        policy_key = profile["policy"] if profile else rl_name
        if not policy_key or policy_key not in RL_POLICIES_REGISTRY:
            continue
        policy_cls = RL_POLICIES_REGISTRY[policy_key]
        if getattr(policy_cls, "is_weight_controller", False):
            continue
        if getattr(policy_cls, "is_recall", False):
            recall[rl_name] = policy_cls(config)
        else:
            if policy is not None:
                raise ValueError(
                    f"Multiple non-recall forward-path RL policies requested "
                    f"({policy_type!r}, {rl_name!r}); only one is supported."
                )
            policy = policy_cls(config)
            policy_type = rl_name
    return policy, policy_type, recall


@functools.lru_cache(maxsize=None)
def _head_accepts_mask(head_type: type) -> bool:
    """Whether a head's ``forward`` takes an ``attention_mask`` (named or via
    ``**kwargs``). Composed heads (Parallel/Sequential) and the crystal router
    do; simple terminals (linear/harmonic/flow) take only hidden states."""
    import inspect

    try:
        params = inspect.signature(head_type.forward).parameters
    except (ValueError, TypeError):
        return True
    return any(
        p.kind is p.VAR_KEYWORD or name == "attention_mask"
        for name, p in params.items()
    )


def _bind_head_cache(head: nn.Module, cache) -> None:
    """Point every cache-aware module inside ``head`` at the live decode cache
    (or clear it with None). Modules opt in with ``accepts_decode_cache`` and
    read ``decode_cache`` in their forward; nothing else is touched. Only a
    ``PraxisCache`` counts - a bare HF cache carries no head-side state and no
    trunk-consistent ``past_length``."""
    if cache is not None and not isinstance(cache, PraxisCache):
        cache = None
    for module in head.modules():
        if getattr(module, "accepts_decode_cache", False):
            module.decode_cache = cache


def _head_mask_kwargs(head: nn.Module, attention_mask) -> dict:
    """``{attention_mask: ...}`` only when the head accepts it - the mask reaches
    the crystal router for per-sequence pad-masked routing (lossless batched
    multi-token decode) without breaking heads that don't take one."""
    if attention_mask is None:
        return {}
    return {"attention_mask": attention_mask} if _head_accepts_mask(type(head)) else {}


def sample_token(
    logits: torch.Tensor, do_sample: bool, temperature: float
) -> torch.Tensor:
    """Sample or greedily select a single token from logits."""
    if do_sample and temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    return logits.argmax(dim=-1)
