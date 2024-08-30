import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import Optional, Tuple, Union
from .configuration_thorns import ThornsConfig


class ThornsModel(PreTrainedModel):
    config_class = ThornsConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    config.n_embd,
                    config.n_head,
                    dim_feedforward=4 * config.n_embd,
                    dropout=config.resid_pdrop,
                    activation=config.activation_function,
                    batch_first=True,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(
                input_shape[-1], dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, src_key_padding_mask=attention_mask)

        hidden_states = self.ln_f(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class ThornsForCausalLM(ThornsModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # Only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        transformer_outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        if past_key_values is not None:
            hidden_states = hidden_states[:, -1, :]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,)
            if use_cache:
                output += (hidden_states,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=transformer_outputs.hidden_states,
            attentions=None,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past)
