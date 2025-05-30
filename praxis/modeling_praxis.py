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
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        losses = LossContainer()
        h_encoder = None
        patch_lengths = None

        if self.encoder:
            inputs, h_encoder, patch_lengths, block_ids, encoder_loss = (
                self.encoder.encode(input_ids)
            )
            losses.add_loss("encoder", encoder_loss)
        else:
            block_ids = create_block_ids(input_ids, self.config.eos_token_id)
            inputs = self.embeds(input_ids)

        last_hidden_state, new_key_values, new_state, losses = self.decoder(
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            block_ids,
            losses,
        )

        return PraxisModelOutput(
            last_hidden_state=last_hidden_state,
            past_key_values=new_key_values,
            hidden_states=None,
            attentions=None,
            current_state=new_state,
            h_encoder=h_encoder,
            patch_lengths=patch_lengths,
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
        return self.criterion(
            logits=logits[..., :-1, :].contiguous(),
            embeddings=embeddings,
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

        # Use separate backward logits if available, otherwise use the same logits
        if backward_logits is not None:
            # Use the separate backward head's logits
            backward_classifier = (
                self.backward_head.classifier
                if hasattr(self.backward_head, "classifier")
                else None
            )
            backward_loss = self.criterion(
                logits=backward_logits[..., 1:, :].contiguous(),
                embeddings=embeddings,
                classifier=backward_classifier,
                labels=backward_labels,
                input_ids=input_ids,
            )
        else:
            # Use the same logits as forward (original behavior)
            backward_loss = self.criterion(
                logits=logits[..., 1:, :].contiguous(),
                embeddings=embeddings,
                classifier=classifier,
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
        )

        # Get hidden states before computing logits
        hidden_states = outputs.last_hidden_state

        backward_logits = None
        if self.encoder:
            logits = self.encoder.decode(
                hidden_states,
                outputs.h_encoder,
                input_ids,
                outputs.patch_lengths,
            )
            classifier = None
        else:
            logits = self.head(hidden_states)
            classifier = self.head.classifier

            # Compute backward logits if we have a separate backward head
            if self.backward_head is not None:
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
                    for key, value in cot_losses.items():
                        outputs.losses.add_loss(f"cot_{key}", value)
            elif rl_type == "cot-reinforce" and labels is not None:
                # CoT with REINFORCE - can work with or without rewards
                _, cot_rl_losses = self.policy(
                    hidden_states,
                    logits=logits[..., :-1, :].contiguous(),
                    labels=labels,
                    rewards=rewards,
                    attention_mask=attention_mask,
                    # Note: generated_texts and ground_truths would be passed from training loop
                )
                if cot_rl_losses is not None:
                    for key, value in cot_rl_losses.items():
                        outputs.losses.add_loss(f"cot_rl_{key}", value)
            elif rewards is not None and labels is not None:
                # REINFORCE and other methods
                hidden_states, rl_loss = self.policy(
                    hidden_states, rewards=rewards, mask=attention_mask
                )
                if rl_loss is not None:
                    outputs.losses.add_loss("rl_policy", rl_loss)

        loss = 0
        if labels is not None:
            if self.config.bidirectional:
                main_loss = self._compute_bidirectional_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=hidden_states,
                    classifier=classifier,
                    input_ids=input_ids,
                    backward_logits=backward_logits,
                )
            else:
                main_loss = self._compute_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=hidden_states,
                    classifier=classifier,
                    input_ids=input_ids,
                )
            loss = outputs.losses.add_loss("main", main_loss)

        # We omit auxiliary losses during validation and inference
        if self.training and labels is not None:
            loss = self.strategy(outputs.losses.get_loss_values())

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
                # For TiedHead, set the embedding weight reference
                self.head.embedding_weight = input_embeddings.weight
            elif input_embeddings is not None and hasattr(self.head, "lm_head"):
                # For regular heads, tie the weights directly
                self.head.lm_head.weight = input_embeddings.weight


@dataclass
class PraxisModelOutput(BaseModelOutputWithPast):
    current_state: Optional[torch.LongTensor] = None
    h_encoder: Optional[torch.FloatTensor] = None
    patch_lengths: Optional[torch.LongTensor] = None
    losses: List[torch.LongTensor] = None
