from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError


def create_forward(
    expert: nn.Module,
    controller: nn.Module,
    manager: nn.Module,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor],
    past_key_values: Optional[Union[List[Any], Dict[str, Any]]],
    current_state: Optional[Any],
    current_depth: int,
    block_ids: Optional[Tensor],
    should_checkpoint: bool = False,
) -> Optional[Tuple[Tensor, Any, Any, Tensor, Optional[bool]]]:
    """
    Create and execute a forward pass function for an expert module with optional checkpointing.

    Args:
        expert: Expert module to execute the forward pass on
        controller: Controller module that determines routing decisions
        manager: A network management module used for Hivemind
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
        attention_mask: Optional attention mask tensor
        past_key_values: Optional cached key/values for faster inference
        current_state: Optional current layer state
        current_depth: Current depth in the network
        block_ids: Optional block identification tensor
        should_checkpoint: Whether to use gradient checkpointing to save memory

    Returns:
        Optional tuple containing:
            - Output hidden states
            - Updated past key values
            - Updated layer state
            - Auxiliary loss
            - Early exit signal
    """

    def custom_forward(
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]],
        current_state: Optional[Any],
        current_depth: int,
        block_ids: Optional[Tensor],
    ) -> Tuple[Tensor, Any, Any, Tensor, Optional[bool]]:
        """
        Custom forward function that can be used with gradient checkpointing.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/values for faster inference
            current_state: Optional current layer state
            current_depth: Current depth in the network
            block_ids: Optional block identification tensor

        Returns:
            Tuple containing:
                - Output hidden states
                - Updated past key values
                - Updated layer state
                - Auxiliary loss
                - Early exit signal
        """
        # Add positional context to both hidden states and attention mask
        hidden_states, attention_mask = controller.add_context(
            hidden_states, attention_mask, current_depth
        )
        # Forward pass
        states, layer_kv, state_update, aux_loss, exit_signal = expert(
            hidden_states,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )
        # Remove context from both hidden states and attention mask
        states, attention_mask = controller.remove_context(states, attention_mask)

        return states, layer_kv, state_update, aux_loss, exit_signal

    try:
        if should_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
                use_reentrant=False,
            )
        else:
            return custom_forward(
                hidden_states,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
    except (P2PDaemonError, P2PHandlerError) as e:
        print(e)
        manager.handle_failure(expert)
        return None


def should_checkpoint(
    training: bool, current_depth: int, checkpoint_every: Optional[int]
) -> bool:
    """
    Determine whether to use gradient checkpointing for a given layer.

    Args:
        training: Whether the model is in training mode
        current_depth: Current layer depth
        checkpoint_every: Frequency to apply checkpointing (None or 0 means no checkpointing)

    Returns:
        Boolean indicating whether to apply checkpointing
    """
    return (
        training
        and current_depth != 0
        and checkpoint_every is not None
        and checkpoint_every != 0
        and current_depth % checkpoint_every
    )
