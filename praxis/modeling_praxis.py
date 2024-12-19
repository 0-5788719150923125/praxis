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
from praxis.losses import LOSS_REGISTRY
from praxis.modules import EMBEDDING_REGISTRY
from praxis.modules.decoder import PraxisDecoder


class PraxisModel(PreTrainedModel):
    config_class = PraxisConfig

    def __init__(self, config: PraxisConfig):
        super().__init__(config)
        self.encoder = False
        if config.byte_latent:
            from praxis.modules.encoder import PraxisByteLatentEncoder

            self.encoder = (
                PraxisByteLatentEncoder(config) if config.byte_latent else False
            )
        else:
            self.embeds = EMBEDDING_REGISTRY[config.block_type](config)
        self.decoder = PraxisDecoder(config)
        self.state = []
        self.aux_losses = []

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        embeds = None
        decoder_tokens = None
        patch_lengths = None
        if self.encoder:
            inputs, decoder_tokens, embeds, patch_lengths = self.encoder.encode(
                input_ids
            )
        else:
            inputs = self.embeds(input_ids)

        if not torch.is_tensor(attention_mask):
            attention_mask = torch.ones(inputs.shape[:2], device=inputs.device)

        state = self.get_initial_state() if state is None else state

        last_hidden_state, state, aux_loss = self.decoder(inputs, state, attention_mask)
        self.aux_losses.append(aux_loss)

        return (
            BaseModelOutputWithPast(
                last_hidden_state=last_hidden_state,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            ),
            state,
            embeds,
            decoder_tokens,
            patch_lengths,
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
        self.loss_func = LOSS_REGISTRY[config.loss_func]()

    def prepare_inputs_for_generation(self, input_ids, attention_mask, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        state: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs, state, embeds, decoder_tokens, patch_lengths = super().forward(
            input_ids=input_ids,
            state=state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.encoder:
            logits = self.encoder.decode(
                hidden_states, decoder_tokens, embeds, patch_lengths
            )
        else:
            logits = self.head(hidden_states)

        loss = 0
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            )
            loss = loss + sum(self.aux_losses)

        self.aux_losses = []

        self._save_state(state)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _save_state(self, state):
        self.state = state

    def get_initial_state(self) -> list[torch.Tensor]:
        if len(self.state) > 0:
            return [
                (
                    self.state[i].detach()
                    if torch.is_tensor(self.state[i])
                    else self.state[i]
                )
                for i in range(self.config.depth)
            ]
        else:
            return [None for _ in range(self.config.depth)]
