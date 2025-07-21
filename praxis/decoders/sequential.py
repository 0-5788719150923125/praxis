import math
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn

from praxis.containers import LossContainer
from praxis.decoders.base import BaseDecoder
from praxis.decoders.checkpoint import create_forward, should_checkpoint

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class SequentialDecoder(BaseDecoder):
    def __init__(self, config: ConfigType) -> None:
        # Heuristically determine reasoning steps based on depth and num_experts
        if "use_reason" in config.meta:
            # Calculate how many "cycles" through the expert pool we make
            # Each complete cycle is considered a reasoning step
            self.steps = max(1, math.ceil(config.depth / config.num_experts))
        else:
            self.steps = 1
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

        current_route: List[int] = []

        controller_state = None
        sequential_experts: List[nn.Module] = list(self.locals) + list(self.remotes)
        ordered_experts: List[nn.Module] = self.controller.sort_experts(
            sequential_experts.copy()
        )

        for total_calls, (reason_step, current_depth) in enumerate(
            product(range(self.steps), range(self.depth))
        ):
            # Enforce depth budget - exit if we've made enough expert calls
            if total_calls >= self.depth:
                break

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

            expert = ordered_experts[next_expert_idx]

            # Handle current_state access with modulo for multiple reasoning steps
            state_idx = (
                next_expert_idx % len(ordered_experts)
                if current_state is not None
                else None
            )
            layer_state = (
                current_state[state_idx] if current_state is not None else None
            )
            (
                hidden_states,
                past_key_values,
                layer_state,
                decoder_loss,
                exit_signal,
            ) = create_forward(
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

            # Update route immediately after expert execution
            current_route = self.controller.update_route(
                hidden_states, current_route, current_depth, next_expert_idx
            )

            # Handle expert decoder loss (can be scalar/tensor or LossContainer)
            if isinstance(decoder_loss, LossContainer):
                losses.add_loss_container(decoder_loss)
            else:
                losses.add_loss("decoder", decoder_loss)

            # Check for Taxus early exit signal (passed directly, not through LossContainer)
            if exit_signal is not None:
                # Debug: Show exit signal during inference and training
                if hasattr(self, "config") and getattr(self.config, "debug", False):
                    mode = "training" if self.training else "inference"
                    print(
                        f"DEBUG: Decoder got exit_signal at depth {current_depth} ({mode}): {exit_signal}"
                    )

                if exit_signal:
                    # Early exit signaled - stop decoding
                    if hasattr(self, "config") and getattr(self.config, "debug", False):
                        mode = "training" if self.training else "inference"
                        print(f"DEBUG: Early exit at depth {current_depth} ({mode})!")
                    break
            hidden_states = self.compressor.reduce_sequence(hidden_states)
            block_ids = self.compressor.reduce_block_ids(block_ids)
            hidden_states = self.post_layer(hidden_states, current_depth)
            if current_state is not None:
                # Use the same state index for updating
                current_state[state_idx] = layer_state

        hidden_states = self.compressor.expand_sequence(hidden_states, seq_len)

        hidden_states = self.post_decoding(hidden_states)

        self.controller.post_forward(hidden_states, current_route)

        # Apply feature sorting
        hidden_states = self.order(hidden_states)

        return hidden_states, past_key_values, current_state, losses
