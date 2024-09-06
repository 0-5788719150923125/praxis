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
from .modules.block import PraxisBlock


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [PraxisBlock(config) for _ in range(config.n_layer)]
        )
        self.post_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.extra_losses = []
        self.n_experts = config.n_experts
        self.register_buffer("ema_expert_utilization", torch.zeros(self.n_experts))
        self.ema_decay = 0.99  # Adjust this value to control the EMA decay rate
        self.forward_count = 0

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        input_embeds = self.wte(input_ids)
        hidden_states = input_embeds

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=hidden_states.device)

        if self.training:
            total_expert_counts = torch.zeros(
                self.n_experts, device=hidden_states.device
            )

        for block in self.blocks:
            hidden_states, extra_loss, expert_counts = block(
                hidden_states, attention_mask
            )
            if self.training:
                self.extra_losses.append(extra_loss)
                total_expert_counts += expert_counts

        # Update EMA expert utilization
        if self.training:
            if self.forward_count == 0:
                self.ema_expert_utilization = total_expert_counts
            else:
                self.ema_expert_utilization = (
                    self.ema_decay * self.ema_expert_utilization
                    + (1 - self.ema_decay) * total_expert_counts
                )
            self.forward_count += 1

        hidden_states = self.post_norm(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.view(*output_shape),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def get_expert_utilization(self):
        if self.forward_count == 0:
            return None
        utilization = self.ema_expert_utilization.cpu().numpy()
        # Convert to percentages
        utilization_percentages = (utilization / np.sum(utilization)) * 100
        return utilization_percentages

    def reset_expert_utilization(self):
        self.ema_expert_utilization.zero_()
        self.forward_count = 0


class PraxisForCausalLM(PraxisModel):
    def __init__(self, config):
        config.causal = True
        super().__init__(config)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": kwargs.get("attention_mask", None),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        transformer_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
            loss += sum(self.extra_losses)

        self.extra_losses = []

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past
        )
