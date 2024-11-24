from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from praxis import PraxisConfig
from praxis.modules import EMBEDDING_REGISTRY
from praxis.modules.compression import PraxisCompressor
from praxis.modules.decoder import PraxisDecoder


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.embeds = EMBEDDING_REGISTRY[config.block_type](config)
        self.compression = (
            PraxisCompressor(
                config, compressed_seq_len=256, compressed_dim=config.num_dims
            )
            if config.compression
            else False
        )
        self.decoder = PraxisDecoder(config)
        self.aux_losses = []

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs = self.embeds(input_ids)

        if not torch.is_tensor(attention_mask):
            attention_mask = torch.ones(inputs.shape[:2], device=inputs.device)

        if self.compression:
            inputs, attention_mask = self.compression(inputs, attention_mask)

        last_hidden_state, aux_loss = self.decoder(inputs, attention_mask)
        self.aux_losses.append(aux_loss)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
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
        self.head = nn.Linear(config.num_dims, config.vocab_size, bias=False)

    def prepare_inputs_for_generation(self, input_ids, attention_mask, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.compression:
            hidden_states = self.compression.decode(hidden_states)

        logits = self.head(hidden_states)

        loss = 0
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
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
