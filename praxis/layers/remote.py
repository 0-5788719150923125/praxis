"""Remote layer implementation for Praxis."""

from typing import List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.layers.local import LocalLayer

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class RemoteLayer(LocalLayer):
    """
    A module for handling remote layers in a mixture-of-experts architecture.
    This extends LocalLayer with additional functionality for remote execution.
    """

    def __init__(
        self,
        config: ConfigType,
        block: Union[nn.Module, bool] = False,
        router: Union[nn.Module, bool] = False,
    ) -> None:
        """
        Initialize remote layer wrapper.

        Args:
            config: Configuration object with model parameters
            block: Block module to wrap
            router: Router module (or False to use default router)
        """
        super().__init__(config, block, router)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor] = None,
        current_state: Optional[Tensor] = None,
        current_depth: int = 0,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, float]:
        """
        Forward pass through remote layer.

        Args:
            inputs: Input tensor
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/value tensors (not used in remote)
            current_state: Optional current state tensor (not used in remote)
            current_depth: Current depth in the network (not used in remote)
            block_ids: Optional block IDs for structured attention (not used in remote)

        Returns:
            Tuple containing:
                - Hidden states tensor
                - Auxiliary loss value
        """
        return self._remote_forward(inputs, attention_mask)

    def _remote_forward(
        self, inputs: Tensor, attention_mask: Optional[Tensor]
    ) -> Tuple[Tensor, float]:
        """
        Forward pass implementation for remote layers.

        Args:
            inputs: Input tensor
            attention_mask: Optional attention mask tensor

        Returns:
            Tuple containing:
                - Hidden states tensor
                - Auxiliary loss value
        """
        # because we would otherwise break gradient flow
        residual = inputs
        aux_losses: List[float] = []

        # Move to CPU for remote execution
        inputs = inputs.to("cpu")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cpu")

        if self.router and isinstance(self.router, nn.Module):
            hidden_states, aux_loss = self.router(self.block, inputs, attention_mask)
            aux_losses.append(aux_loss)
        elif isinstance(self.block, nn.Module):
            # because hivemind cannot receive undefined arguments in the forward pass
            dummy_router_weights = torch.zeros_like(inputs)
            # because we do not backpropagate through remote experts
            hidden_states = self.block(
                inputs,
                attention_mask,
                dummy_router_weights,
            )
        else:
            raise ValueError("Neither router nor block is a valid module")

        # TODO: we could possibly add some differentiable noise here; perhaps as a penalty on slow experts?
        hidden_states = hidden_states.to(residual.device) + residual
        return hidden_states, sum(aux_losses)
