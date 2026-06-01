import functools
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from praxis import DECODER_REGISTRY, EMBEDDING_REGISTRY, ENCODER_REGISTRY, PraxisConfig
from praxis.containers import LossContainer
from praxis.heads import HEAD_REGISTRY
from praxis.losses import get_loss_function
from praxis.losses.contrastive_isotropy import ContrastiveIsotropyLoss
from praxis.policies import RL_POLICIES_REGISTRY
from praxis.strategies import STRATEGIES_REGISTRY
from praxis.tasks import TASK_NAMES, resolve_task_weighter
from praxis.utils import create_block_ids


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig
    _supports_cache_class = True

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.encoder = False
        self.embeds = None
        if config.encoder_type is not None:
            self.encoder = ENCODER_REGISTRY.get(config.encoder_type)(config)
            # Encoders that name an embedding profile get their input
            # embeddings built from the registry and injected, mirroring how
            # heads classify encoder-declared output dims. Encoders that own
            # their embeddings (e.g. CALM) name no profile.
            profile = self.encoder.embedding_profile
            if profile:
                self.embeds = EMBEDDING_REGISTRY[profile](config, encoder=self.encoder)
                self.encoder.set_embeddings(self.embeds)
        else:
            self.embeds = EMBEDDING_REGISTRY[config.block_type](config)
        self.decoder = DECODER_REGISTRY.get(config.decoder_type)(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        current_state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        losses = LossContainer()
        h_encoder = None
        patch_lengths = None

        # Compute EOS-aware block_ids on the raw input_ids regardless of
        # encoder presence, so the local byte-level encoder can isolate
        # packed documents. Encoders compress the sequence (patches),
        # which destroys these boundaries, so they return None for the
        # global decoder's block_ids - the patch-level "block" labels are
        # noisy and shouldn't gate the global attention.
        token_block_ids = create_block_ids(input_ids, self.config.eos_token_id)

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

        last_hidden_state, new_key_values, new_state, losses = self.decoder(
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            block_ids,
            losses,
            labels,
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
    model_type = "praxis"

    def __init__(self, config: PraxisConfig):
        config.causal = True
        super().__init__(config)

        # Build the LM head, passing the encoder reference so heads that
        # participate in encoder-mode forward (harmonic, crystal) can
        # size their own submodules. Encoder-agnostic heads (forward,
        # tied) ignore the reference and skip allocating an lm_head.
        encoder_ref = self.encoder if self.encoder else None
        # Standard-mode tying normally routes to the dedicated "tied" head;
        # heads that tie their own weights (crystal, and compositions ending in
        # it) keep their type and tie themselves in tie_weights(). Read the flag
        # off the registered class, unwrapping any functools.partial variant.
        entry = HEAD_REGISTRY.get(config.head_type)
        head_cls_decl = entry
        while isinstance(head_cls_decl, functools.partial):
            head_cls_decl = head_cls_decl.func
        self_ties = bool(getattr(head_cls_decl, "self_ties", False))
        if encoder_ref is None and config.tie_word_embeddings and not self_ties:
            head_type = "tied"
        else:
            head_type = config.head_type
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

        # Initialize RL policy if requested. rl_type may be a forward-path policy
        # key (reinforce/grpo/cot) or a weight-editing controller *profile* key
        # (which resolves to its underlying policy + bundled edit_mode/selector).
        self.policy = None
        rl_type = getattr(config, "rl_type", None)
        from praxis.policies import get_rl_profile

        _profile = get_rl_profile(rl_type)
        policy_key = _profile["policy"] if _profile else rl_type
        if policy_key and policy_key in RL_POLICIES_REGISTRY:
            policy_cls = RL_POLICIES_REGISTRY[policy_key]
            # Weight-editing controllers act on parameters from a training
            # callback, not on hidden states in the forward pass - so they are
            # not built as self.policy (the callback owns them).
            if not getattr(policy_cls, "is_weight_controller", False):
                self.policy = policy_cls(config)

        # Encoders that own their loss (e.g. CALM) bypass the main-CE path
        # entirely, so don't build a criterion we'd never call.
        if self.encoder and self.encoder.handles_loss:
            self.criterion = None
        else:
            self.criterion = get_loss_function(config.loss_func, config.vocab_size)

        # Per-task loss weighting. Identity (no-op) unless --task-weights
        # is set; the assistant mask from the chat template is always
        # applied when present. Learnable variants expose an anchor_loss
        # that gets folded into the combined objective by the strategy below.
        self.tasker = resolve_task_weighter(config.task_weights)

        # Set by the trainer to the task indices a live dataset produces;
        # get_metrics() uses it to skip charting weights for absent tasks.
        self.active_task_ids = None

        # SimCTG isotropy regularizer (additive; see forward). On by default.
        self.aux = None
        if getattr(config, "contrastive_isotropy", True):
            self.aux = ContrastiveIsotropyLoss(pad_id=config.pad_token_id)

        # The strategy for combining multiple losses into a single scalar objective.
        self.strategy = STRATEGIES_REGISTRY.get(config.strategy, "naive")()

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
        return compute_layer_wise_loss(
            hidden_states=hidden_states,
            labels=labels,
            head=self.head,
            criterion=self.criterion,
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
        is_cut_ce = self.criterion.__class__.__name__ == "CutCrossEntropyLoss"

        # Check if encoder outputs are already aligned
        if self.encoder and self.encoder.outputs_are_aligned:
            return self.criterion(
                logits=logits.contiguous(),
                embeddings=embeddings if is_cut_ce else embeddings,
                classifier=classifier,
                labels=labels,
                input_ids=input_ids,
                loss_weights=loss_weights,
            )
        else:
            return self.criterion(
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
        backward_loss = self.criterion(
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
        use_cache: bool = False,
        **kwargs,
    ) -> dict:
        if not use_cache:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        return {
            "input_ids": input_ids[:, -1:],
            "attention_mask": (
                attention_mask[:, -1:] if attention_mask is not None else None
            ),
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
        if (
            self.training
            and self.encoder
            and self.encoder.in_pretraining()
            and not self._has_uninitialized_params()
        ):
            self._set_pretraining_lock(True)
            return CausalLMOutputWithPast(
                loss=self.encoder.pretraining_loss(input_ids)
            )
        if self.training and self.encoder:
            self._set_pretraining_lock(False)

        outputs = super().forward(
            input_ids=input_ids,
            current_state=current_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

        # Get hidden states before computing logits
        hidden_states = outputs.last_hidden_state

        classifier = None
        backward_logits = None

        logits = hidden_states

        # Check if we're using cut_cross_entropy to optimize logits computation
        is_cut_ce = (
            self.criterion is not None
            and self.criterion.__class__.__name__ == "CutCrossEntropyLoss"
        )
        skip_logits_for_training = is_cut_ce and self.training and labels is not None

        if self.encoder:
            """Needs encoding:"""

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
                # same path as standalone mode. Skip materializing logits
                # under cut-CE training (the loss projects internally).
                if not skip_logits_for_training:
                    logits = self.head(decoder_embeds)
                classifier = self.head.classifier
            hidden_states = decoder_embeds

            # Encoders that manage their own losses (e.g. CALM) may emit
            # side-channel losses during decode(); fold them into the
            # shared loss container so the strategy can combine them.
            for key, value in self.encoder.consume_pending_losses().items():
                outputs.losses.add_loss(key, value)
        elif hidden_states.size(-1) != self.config.vocab_size:
            """Needs projection/classification:"""

            # Skip logits computation during training with cut_cross_entropy
            # The loss function will compute it internally without materializing the full matrix
            if not skip_logits_for_training:
                logits = self.head(hidden_states)
            # Always get classifier reference (needed for cut_cross_entropy)
            classifier = self.head.classifier

            # Compute backward logits if we have a separate backward head
            # (only when we need them - not during cut_ce training)
            if self.backward_head is not None and not skip_logits_for_training:
                backward_logits = self.backward_head(hidden_states)

        # Apply RL policy if enabled
        if self.policy is not None:
            # Different RL algorithms need different inputs
            rl_type = getattr(self.config, "rl_type", None)

            if rl_type == "grpo" and rewards is not None and labels is not None:
                # GRPO needs logits and labels for proper loss computation
                # For now, we'll need to compute reference logits in the training loop
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
                        outputs.losses.add_loss(f"rl_{key}", value)
            elif rl_type == "cot" and labels is not None:
                # Basic CoT uses supervised learning with weighted loss
                # Note: Pass shifted logits to match labels
                _, cot_losses = self.policy(
                    hidden_states,
                    logits=logits[..., :-1, :].contiguous(),
                    labels=labels,
                    attention_mask=attention_mask,
                    token_weights=token_weights,  # Pre-computed token weights from builder
                )
                if cot_losses is not None:
                    # Add CoT losses directly using LossContainer integration
                    outputs.losses.add_loss_container(cot_losses)
            elif rewards is not None and labels is not None:
                # REINFORCE and other methods
                hidden_states, rl_loss = self.policy(
                    hidden_states, rewards=rewards, mask=attention_mask
                )
                if rl_loss is not None:
                    outputs.losses.add_loss("rl_policy", rl_loss)

        loss = 0
        if labels is not None:
            loss_weights = self._build_loss_weights(
                labels=labels,
                task_type_ids=task_type_ids,
                assistant_mask=assistant_mask,
            )
            # Check if trainer already computed layer-wise losses (e.g., MonoForward trainer)
            if "_layer_wise_complete" in outputs.losses.loss_dict:
                # Trainer handled its own training, use strategy to combine losses
                layer_losses = [
                    v
                    for k, v in outputs.losses.loss_dict.items()
                    if k != "_layer_wise_complete" and k != "main"
                ]
                if layer_losses:
                    loss = self.strategy(layer_losses)
            elif self.encoder and self.encoder.handles_loss:
                # Encoder owns its loss bookkeeping (see CALMEncoder).
                # Registered losses will be combined by the strategy below.
                pass
            elif self.config.bidirectional:
                main_loss = self._compute_bidirectional_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=hidden_states,
                    classifier=classifier,
                    input_ids=input_ids,
                    backward_logits=backward_logits,
                )
                loss = outputs.losses.add_loss("main", main_loss)
            else:
                main_loss = self._compute_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=hidden_states,
                    classifier=classifier,
                    input_ids=input_ids,
                    loss_weights=loss_weights,
                )
                loss = outputs.losses.add_loss("main", main_loss)

        # Task-weight anchor loss (learnable weighters only). Folds in
        # alongside MTP / RL / encoder aux losses so the strategy sees it.
        if self.training and labels is not None:
            anchor = self.tasker.anchor_loss()
            if anchor is not None:
                outputs.losses.add_loss("task_weight_anchor", anchor)

        # Heads can emit named aux losses (e.g., HarmonicHead's
        # forward-shift smoothness loss, CrystalHead's centers-RMS
        # regularizer).
        if self.training and labels is not None and self.head is not None:
            aux = self.head.aux_losses()
            for name, value in aux.items():
                outputs.losses.add_loss(name, value)

        # MTP auxiliary loss (training only)
        if self.mtp is not None and self.training and labels is not None:
            mtp_inputs = self.mtp.prepare_inputs(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                embed_fn=self.get_input_embeddings(),
                head=self.head,
                patch_embeds=outputs.patch_embeds if self.encoder else None,
            )
            mtp_losses = self.mtp(mtp_inputs)
            outputs.losses.add_loss_container(mtp_losses)

        # Contrastive isotropy (SimCTG): additive regularizer on the last-layer
        # representations to keep the space discriminative for contrastive-search
        # decoding. Leaves the main objective untouched.
        if self.aux is not None and self.training and labels is not None:
            iso_loss = self.aux(hidden_states, input_ids)
            outputs.losses.add_loss("contrastive", iso_loss)

        # We omit auxiliary losses during validation and inference - except
        # for handles_loss encoders (CALM), where the encoder owns the main
        # loss and there's nothing else to fall back to. Without this their
        # val_loss would stay at 0.
        handles_loss_encoder = (
            self.encoder is not False
            and getattr(self.encoder, "handles_loss", False)
        )
        if labels is not None and (self.training or handles_loss_encoder):
            loss_values = outputs.losses.get_loss_values()
            if len(loss_values) > 1:
                # Multiple tagged losses — let the strategy combine them
                loss = self.strategy(loss_values)
            elif loss == 0 and len(loss_values) > 0:
                # Only auxiliary losses (no main) — combine via strategy
                loss = self.strategy(loss_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Generate tokens, dispatching to specialised paths when applicable."""
        # Encoders may own their generation loop (e.g. CALM's latent path).
        # The hook gets a callable to run the global transformer and returns
        # None to defer to the standard HF generate loop.
        if self.encoder and not self.training:
            result = self.encoder.custom_generate(
                inputs,
                base_forward=lambda ids: PraxisModel.forward(self, input_ids=ids),
                generation_config=generation_config,
                **kwargs,
            )
            if result is not None:
                return result
        if (
            self.mtp is not None
            and not self.encoder
            and not self.training
            and generation_config is not None
            and getattr(generation_config, "num_beams", 1) == 1
        ):
            return self._speculative_generate(inputs, generation_config, **kwargs)
        return super().generate(inputs, generation_config=generation_config, **kwargs)

    @torch.no_grad()
    def _speculative_generate(self, input_ids, generation_config, **kwargs):
        """MTP-based speculative decoding for faster inference.

        For each step:
        1. Run main model forward to get hidden states and next-token logits
        2. Draft N additional tokens greedily via MTP modules
        3. Verify all N+1 candidates in a single main model forward pass
        4. Accept the longest prefix where main model agrees with draft
        """
        from types import SimpleNamespace

        max_new_tokens = getattr(generation_config, "max_new_tokens", 100)
        do_sample = getattr(generation_config, "do_sample", False)
        temperature = getattr(generation_config, "temperature", 1.0) or 1.0
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        return_dict = kwargs.get("return_dict_in_generate", False)

        eos_set = set()
        if isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        elif isinstance(eos_token_id, (list, tuple)):
            eos_set = set(eos_token_id)

        generated = input_ids
        embed_fn = self.get_input_embeddings()
        num_new = 0

        while num_new < max_new_tokens:
            # Main model forward pass to get hidden states
            base_out = PraxisModel.forward(self, input_ids=generated)
            hidden_states = base_out.last_hidden_state
            main_logits = self.head(hidden_states)

            # Sample first token from main model
            last_logits = main_logits[:, -1, :]
            token_0 = self._sample_token(last_logits, do_sample, temperature)
            token_0_2d = token_0.unsqueeze(1)

            if token_0.item() in eos_set:
                generated = torch.cat([generated, token_0_2d], dim=1)
                num_new += 1
                break

            # Draft additional tokens with MTP
            draft_ids = self.mtp.draft_next_tokens(
                hidden_states[:, -1:, :], token_0_2d, embed_fn, self.head
            )

            # Combine: main prediction + drafts → [batch, 1+N]
            candidates = torch.cat([token_0_2d, draft_ids], dim=1)
            n_candidates = candidates.size(1)

            # Verify all candidates in one forward pass
            verify_input = torch.cat([generated, candidates], dim=1)
            verify_out = PraxisModel.forward(self, input_ids=verify_input)
            verify_logits = self.head(verify_out.last_hidden_state)

            # Check agreement at each position
            gen_len = generated.size(1)
            accepted = 0

            for i in range(n_candidates):
                v_logits = verify_logits[:, gen_len - 1 + i, :]
                v_token = self._sample_token(v_logits, do_sample, temperature)

                if v_token.item() == candidates[:, i].item():
                    accepted += 1
                    if v_token.item() in eos_set:
                        generated = torch.cat(
                            [generated, candidates[:, :accepted]], dim=1
                        )
                        num_new += accepted
                        if return_dict:
                            return SimpleNamespace(sequences=generated)
                        return generated
                else:
                    # Divergence: keep accepted prefix + verified token
                    parts = [generated]
                    if accepted > 0:
                        parts.append(candidates[:, :accepted])
                    parts.append(v_token.unsqueeze(1))
                    generated = torch.cat(parts, dim=1)
                    num_new += accepted + 1
                    break
            else:
                # All candidates accepted — also take a bonus token
                generated = torch.cat([generated, candidates], dim=1)
                num_new += n_candidates

                if num_new < max_new_tokens:
                    bonus_logits = verify_logits[:, gen_len - 1 + n_candidates, :]
                    bonus = self._sample_token(bonus_logits, do_sample, temperature)
                    generated = torch.cat([generated, bonus.unsqueeze(1)], dim=1)
                    num_new += 1
                    if bonus.item() in eos_set:
                        break

        if return_dict:
            return SimpleNamespace(sequences=generated)
        return generated

    def _sample_token(self, logits, do_sample, temperature):
        """Sample or greedily select a single token from logits."""
        if do_sample and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        return logits.argmax(dim=-1)

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


@dataclass
class PraxisModelOutput(BaseModelOutputWithPast):
    current_state: Optional[torch.LongTensor] = None
    h_encoder: Optional[torch.FloatTensor] = None
    patch_lengths: Optional[torch.LongTensor] = None
    patch_embeds: Optional[torch.FloatTensor] = None
    local_decoder_tokens: Optional[torch.LongTensor] = None
    token_block_ids: Optional[torch.LongTensor] = None
    losses: List[torch.LongTensor] = None
