from typing import Optional, Tuple, Union

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
        self.wme = nn.Linear(config.n_emb, config.n_dim, bias=False)
        self.max_pca_k = min(
            config.n_dim, config.n_emb
        )  # Maximum number of principal components
        self.n_factors = config.n_factors
        self.pca = nn.Linear(config.n_dim + self.max_pca_k, config.n_dim, bias=True)
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

        # Word token embeddings
        input_embeds = self.wte(input_ids)

        # Linear projection (residual)
        inputs_reduced = self.wme(input_embeds)

        # Calculate pca_k dynamically
        q = min(self.max_pca_k, input_embeds.size(0), input_embeds.size(1)) - 1

        # PCA operation
        if q > 0:
            _, _, v = torch.pca_lowrank(
                input_embeds, q=q, center=True, niter=self.n_factors
            )
            pca_reduced = torch.matmul(input_embeds, v)
        else:
            # Fallback if PCA is not possible
            pca_reduced = torch.zeros(
                *input_embeds.shape[:-1], 0, device=input_embeds.device
            )

        # Pad pca_reduced if necessary
        if pca_reduced.size(-1) < self.max_pca_k:
            padding = torch.zeros(
                *pca_reduced.shape[:-1],
                self.max_pca_k - pca_reduced.size(-1),
                device=pca_reduced.device
            )
            pca_reduced = torch.cat([pca_reduced, padding], dim=-1)

        # Combine linear projection and PCA results
        combined = torch.cat([inputs_reduced, pca_reduced], dim=-1)
        hidden_states = self.pca(combined)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=hidden_states.device)

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
            loss = nn.CrossEntropyLoss()(
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
