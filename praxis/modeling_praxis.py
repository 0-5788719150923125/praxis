from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_praxis import PraxisConfig
from .modules.decoder import PraxisDecoder


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.n_dim = config.n_dim
        self.wte = nn.Embedding(config.vocab_size, config.n_emb)
        # Add projection layer if n_emb is larger than n_dim
        if config.n_emb > config.n_dim:
            self.reduce = nn.Linear(config.n_emb, config.n_dim)
        else:
            self.reduce = nn.Identity()
        self.decoder = PraxisDecoder(config)
        self.aux_losses = []

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        input_embeds = self.wte(input_ids)
        inputs_reduced = self.reduce(input_embeds)
        hidden_states = inputs_reduced

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=hidden_states.device)

        outputs = self.decoder(hidden_states, attention_mask)

        if self.training:
            self.aux_losses.append(outputs["aux_loss"])

        return BaseModelOutputWithPast(
            last_hidden_state=outputs["hidden_states"],
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class PraxisForCausalLM(PraxisModel):
    def __init__(self, config):
        config.causal = True
        super().__init__(config)
        self.head = nn.Linear(config.n_dim, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "attention_mask": kwargs.get("attention_mask", None),
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

        transformer_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss += sum(self.aux_losses)

        self.aux_losses = []

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
