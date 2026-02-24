from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from praxis import DECODER_REGISTRY, EMBEDDING_REGISTRY, ENCODER_REGISTRY, PraxisConfig
from praxis.containers import LossContainer
from praxis.heads import HEAD_REGISTRY
from praxis.losses import get_loss_function
from praxis.policies import RL_POLICIES_REGISTRY
from praxis.strategies import STRATEGIES_REGISTRY
from praxis.utils import create_block_ids


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig
    _supports_cache_class = True

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.encoder = False
        if config.encoder_type is not None:
            self.encoder = ENCODER_REGISTRY.get(config.encoder_type)(config)
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

        if self.encoder:
            (
                inputs,
                h_encoder,
                patch_lengths,
                block_ids,
                encoder_loss,
                local_decoder_tokens,
            ) = self.encoder.encode(input_ids)
            losses.add_loss("encoder", encoder_loss)
        else:
            block_ids = create_block_ids(input_ids, self.config.eos_token_id)
            inputs = self.embeds(input_ids)
            local_decoder_tokens = None

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

        # Initialize the language modeling head based on head_type
        if config.encoder_type is None:
            if config.tie_word_embeddings:
                # Use tied head and get embedding weight reference
                self.head = HEAD_REGISTRY["tied"](config)
                # Weight will be tied after model initialization
            else:
                self.head = HEAD_REGISTRY.get(config.head_type, "forward")(config)
        else:
            self.head = None

        # Initialize separate backward head if requested
        if config.bidirectional and config.encoder_type is None:
            self.backward_head = HEAD_REGISTRY.get(config.head_type, "forward")(config)
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

        # Initialize RL policy if requested
        self.policy = None
        rl_type = getattr(config, "rl_type", None)
        if rl_type and rl_type in RL_POLICIES_REGISTRY:
            self.policy = RL_POLICIES_REGISTRY[rl_type](config)

        self.criterion = get_loss_function(config.loss_func, config.vocab_size)
        self.strategy = STRATEGIES_REGISTRY.get(config.strategy, "naive")()

        # Tie weights if requested
        if config.tie_word_embeddings and self.head is not None:
            self.tie_weights()

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        classifier: Optional[nn.Module],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the main loss using the criterion."""
        # cut_cross_entropy needs FULL UNSHIFTED embeddings to avoid materializing shifted tensors
        # Check by class name to avoid hard dependency on integration
        is_cut_ce = self.criterion.__class__.__name__ == "CutCrossEntropyLoss"

        # Check if encoder outputs are already aligned
        if self.encoder and getattr(self.encoder, "outputs_are_aligned", False):
            return self.criterion(
                logits=logits.contiguous(),
                embeddings=embeddings if is_cut_ce else embeddings,
                classifier=classifier,
                labels=labels,
                input_ids=input_ids,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
        is_cut_ce = self.criterion.__class__.__name__ == "CutCrossEntropyLoss"
        skip_logits_for_training = is_cut_ce and self.training and labels is not None

        if self.encoder:
            """Needs encoding:"""

            logits = self.encoder.decode(
                hidden_states,
                outputs.h_encoder,
                input_ids,
                outputs.patch_lengths,
                outputs.local_decoder_tokens,
            )
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
                )
                loss = outputs.losses.add_loss("main", main_loss)

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

        # We omit auxiliary losses during validation and inference
        if self.training and labels is not None:
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
        """Generate tokens, using MTP speculative decoding when available."""
        if (
            self.mtp is not None
            and not self.encoder
            and not self.training
            and generation_config is not None
            and getattr(generation_config, "num_beams", 1) == 1
        ):
            return self._speculative_generate(inputs, generation_config, **kwargs)
        return super().generate(
            inputs, generation_config=generation_config, **kwargs
        )

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
        if hasattr(self, "embeds"):
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

    def tie_weights(self) -> None:
        """Tie the input and output embeddings weights."""
        if self.config.tie_word_embeddings and self.head is not None:
            input_embeddings = self.get_input_embeddings()
            if input_embeddings is not None and hasattr(self.head, "embedding_weight"):
                # For TiedWeights, set the embedding weight reference
                self.head.embedding_weight = input_embeddings.weight
            elif input_embeddings is not None and hasattr(self.head, "lm_head"):
                # For regular heads, tie the weights directly
                self.head.lm_head.weight = input_embeddings.weight

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
    losses: List[torch.LongTensor] = None
