from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn

from praxis.containers import LossContainer
from praxis.decoders.base import BaseDecoder
from praxis.decoders.checkpoint import create_forward, should_checkpoint

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


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
        losses: LossContainer = None,
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
            losses: A storage class for auxiliary losses

        Returns:
            Tuple containing:
                - Output hidden states
                - Updated past key values
                - Updated layer states
                - Auxiliary loss container
        """

        _, seq_len, _ = hidden_states.shape

        controller_state = None
        sequential_experts: List[nn.Module] = list(self.locals) + list(self.remotes)
        ordered_experts: List[nn.Module] = self.controller.sort_experts(
            sequential_experts.copy()
        )
        current_route: List[int] = []

        for i in range(self.depth):
            current_depth = i
            hidden_states, controller_state, controller_loss, next_expert_idx = (
                self.controller.get_next_expert(
                    hidden_states,
                    controller_state,
                    sequential_experts,
                    ordered_experts,
                    current_route,
                    current_depth,
                )
            )

            losses.add_loss_container(controller_loss)
            if next_expert_idx is None:
                break

            current_route = self.controller.update_route(
                hidden_states, current_route, current_depth, next_expert_idx
            )

            expert = ordered_experts[next_expert_idx]

            layer_state = (
                current_state[next_expert_idx] if current_state is not None else None
            )
            hidden_states, past_key_values, layer_state, decoder_loss = create_forward(
                expert,
                self.controller,
                self.manager,
                hidden_states,
                attention_mask,
                past_key_values,
                layer_state,
                current_depth,
                block_ids,
                should_checkpoint(self.training, current_depth, self.checkpoint_every),
            )
            # Handle expert decoder loss (which is scalar/tensor, not LossContainer)
            losses.add_loss("decoder", decoder_loss)
            hidden_states = self.compressor.reduce_sequence(hidden_states)
            block_ids = self.compressor.reduce_block_ids(block_ids)
            hidden_states = self.post_layer(hidden_states, current_depth)
            if current_state is not None:
                current_state[next_expert_idx] = layer_state

        hidden_states = self.compressor.expand_sequence(hidden_states, seq_len)

        hidden_states = self.post_decoding(hidden_states)

        self.controller.post_forward(hidden_states, current_route)

        return hidden_states, past_key_values, current_state, losses
