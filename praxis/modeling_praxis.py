from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from praxis import PraxisConfig
from praxis.losses import LOSS_REGISTRY
from praxis.modules import EMBEDDING_REGISTRY
from praxis.modules.decoder import PraxisDecoder
from praxis.modules.encoder import PraxisEncoder
from praxis.utils import create_block_ids


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig
    _supports_cache_class = True

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.encoder = False
        if config.byte_latent:
            self.encoder = PraxisEncoder(config)
        else:
            self.embeds = EMBEDDING_REGISTRY[config.block_type](config)
        self.decoder = PraxisDecoder(config)
        self.current_state = []
        self.aux_losses = []

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        current_state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        h_encoder = None
        patch_lengths = None
        if self.encoder:
            inputs, h_encoder, patch_lengths, block_ids, entropy_loss = (
                self.encoder.encode(input_ids)
            )
            self.aux_losses.append(entropy_loss)
        else:
            block_ids = create_block_ids(input_ids, self.config.pad_token_id)
            inputs = self.embeds(input_ids)

        last_hidden_state, new_key_values, new_state, aux_loss = self.decoder(
            inputs, current_state, attention_mask, past_key_values, block_ids
        )
        self.aux_losses.append(aux_loss)

        return PraxisModelOutput(
            last_hidden_state=last_hidden_state,
            past_key_values=new_key_values,
            hidden_states=None,
            attentions=None,
            current_state=new_state,
            h_encoder=h_encoder,
            patch_lengths=patch_lengths,
        )

    def get_addr(self):
        if self.decoder.manager:
            self.decoder.manager.get_visible_maddrs()

    def get_metrics(self):
        return dict(**self.decoder.get_metrics())


class PraxisForCausalLM(PraxisModel, GenerationMixin):
    model_type = "praxis"

    def __init__(self, config: PraxisConfig):
        config.causal = True
        super().__init__(config)
        if not config.byte_latent:
            self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = LOSS_REGISTRY[config.loss_func]()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        use_cache=False,
        **kwargs,
    ):
        if not use_cache:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        # First generation step
        if not isinstance(past_key_values, DynamicCache):
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": DynamicCache(),
                "current_state": None,
            }

        # Subsequent steps
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
        else:
            logits = self.head(outputs.last_hidden_state)

        loss = 0
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                input_ids.view(-1),
            )
            loss = loss + sum(self.aux_losses)

        self.aux_losses = []

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
