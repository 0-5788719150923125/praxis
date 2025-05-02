from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig

from praxis.decoders.base import BaseDecoder
from praxis.decoders.checkpoint import create_forward, should_checkpoint

ConfigType = TypeVar("ConfigType", bound=PretrainedConfig)


class SequentialDecoder(BaseDecoder):
    def __init__(self, config: ConfigType) -> None:
        super().__init__(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
        current_state: Optional[List[Any]] = None,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor, Optional[Union[List[Any], Dict[str, Any]]], Optional[List[Any]], Tensor
    ]:
        """
        Forward pass through the sequential decoder.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/values for faster inference
            current_state: Optional current layer states
            block_ids: Optional block identification tensor

        Returns:
            Tuple containing:
                - Output hidden states
                - Updated past key values
                - Updated layer states
                - Combined auxiliary loss
        """
        controller_state = None
        sequential_experts: List[nn.Module] = list(self.locals) + list(self.remotes)
        ordered_experts: List[nn.Module] = self.controller.sort_experts(
            sequential_experts.copy()
        )
        current_route: List[int] = []
        aux_losses: List[Tensor] = []

        for i in range(self.depth):
            controller_state, aux_loss, current_route, next_expert_idx = (
                self.controller.get_next_expert(
                    hidden_states,
                    controller_state,
                    sequential_experts,
                    ordered_experts,
                    current_route,
                    current_depth=i,
                )
            )

            aux_losses.append(aux_loss)
            if next_expert_idx is None:
                break

            expert = ordered_experts[next_expert_idx]

            layer_state = (
                current_state[next_expert_idx] if current_state is not None else None
            )
            hidden_states, past_key_values, layer_state, aux_loss = create_forward(
                expert,
                self.controller,
                self.manager,
                hidden_states,
                attention_mask,
                past_key_values,
                layer_state,
                i,
                block_ids,
                should_checkpoint(self.training, i, self.checkpoint_every),
            )
            aux_losses.append(aux_loss)
            hidden_states = self.post_layer(hidden_states, i)
            if current_state is not None:
                current_state[next_expert_idx] = layer_state

        hidden_states = self.post_decoding(hidden_states)

        self.controller.post_forward(hidden_states, current_route)

        return hidden_states, past_key_values, current_state, sum(aux_losses)
