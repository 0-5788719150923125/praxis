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
            self.head = HEAD_REGISTRY.get(config.head_type, "forward")(config)
        else:
            self.head = None

        self.criterion = get_loss_function(config.loss_func, config.vocab_size)
        self.strategy = STRATEGIES_REGISTRY.get(config.strategy, "naive")()

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
    ) -> torch.Tensor:
        """Compute bidirectional loss (forward and backward prediction).

        Note: labels are already shifted right (input_ids[..., 1:]) when passed in.
        """
        # Forward loss: predict next token (standard causal LM)
        forward_loss = self._compute_loss(logits, labels, embeddings, classifier, input_ids)

        # Backward loss: predict previous token
        # For backward prediction, we want logits[1:] to predict input_ids[:-1]
        backward_labels = input_ids[..., :-1].contiguous()
        backward_loss = self.criterion(
            logits=logits[..., 1:, :].contiguous(),
            embeddings=embeddings,
            classifier=classifier,
            labels=backward_labels,
            input_ids=input_ids,
        )

        # Combine losses (equal weighting by default)
        return (forward_loss + backward_loss) / 2

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = super().forward(
            input_ids=input_ids,
            current_state=current_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.encoder:
            logits = self.encoder.decode(
                outputs.last_hidden_state,
                outputs.h_encoder,
                input_ids,
                outputs.patch_lengths,
            )
            classifier = None
        else:
            logits = self.head(outputs.last_hidden_state)
            classifier = self.head.classifier

        loss = 0
        if labels is not None:
            if self.config.bidirectional:
                main_loss = self._compute_bidirectional_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=outputs.last_hidden_state,
                    classifier=classifier,
                    input_ids=input_ids,
                )
            else:
                main_loss = self._compute_loss(
                    logits=logits,
                    labels=labels,
                    embeddings=outputs.last_hidden_state,
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


@dataclass
class PraxisModelOutput(BaseModelOutputWithPast):
    current_state: Optional[torch.LongTensor] = None
    h_encoder: Optional[torch.FloatTensor] = None
    patch_lengths: Optional[torch.LongTensor] = None
    losses: List[torch.LongTensor] = None
