from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.routers import ROUTER_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class LocalExpert(nn.Module):
    """
    A module for handling local experts in a mixture-of-experts architecture.
    """

    __version__ = "0.2.0"

    def __init__(
        self,
        config: ConfigType,
        block: Union[nn.Module, bool] = False,
        router: Union[nn.Module, bool] = False,
    ) -> None:
        """
        Initialize local expert wrapper.

        Args:
            config: Configuration object with model parameters
            block: Block module to wrap
            router: Router module (or False to use default router)
        """
        super().__init__()
        self.block: Union[nn.Module, bool] = block
        self.router: Union[nn.Module, bool] = router
        if config.router_type is not None and not self.router:
            self.router = ROUTER_REGISTRY.get("mixture_of_depths")(config)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor],
        current_state: Optional[Tensor],
        current_depth: int,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        """
        Forward pass through local expert.

        Args:
            inputs: Input tensor
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/value tensors
            current_state: Optional current state tensor
            current_depth: Current depth in the network
            block_ids: Optional block IDs for structured attention

        Returns:
            Tuple containing:
                - Hidden states tensor
                - Updated key/value cache
                - Updated state tensor
                - Auxiliary loss value
        """
        return self._forward(
            inputs,
            current_state,
            attention_mask,
            past_key_values,
            current_depth,
            block_ids,
        )

    def _forward(
        self,
        inputs: Tensor,
        current_state: Optional[Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor],
        current_depth: int,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        """
        Forward pass implementation for local experts.

        Args:
            inputs: Input tensor
            current_state: Current state tensor
            attention_mask: Attention mask tensor
            past_key_values: Cached key/value tensors
            current_depth: Current depth in the network
            block_ids: Optional block IDs for structured attention

        Returns:
            Tuple containing:
                - Hidden states tensor
                - Updated key/value cache
                - Updated state tensor
                - Auxiliary loss value
        """
        aux_losses: List[float] = []
        if self.router and isinstance(self.router, nn.Module):
            hidden_states, layer_kv, state_update, aux_loss = self.router(
                self.block,
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
            aux_losses.append(aux_loss)
        elif isinstance(self.block, nn.Module):
            hidden_states, layer_kv, state_update, aux_loss = self.block(
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
            aux_losses.append(aux_loss)
        else:
            raise ValueError("Neither router nor block is a valid module")

        return hidden_states, layer_kv, state_update, sum(aux_losses)


class RemoteExpert(LocalExpert):
    """
    A module for handling remote experts in a mixture-of-experts architecture.
    This extends LocalExpert with additional functionality for remote execution.
    """

    def __init__(
        self,
        config: ConfigType,
        block: Union[nn.Module, bool] = False,
        router: Union[nn.Module, bool] = False,
    ) -> None:
        """
        Initialize remote expert wrapper.

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
        Forward pass through remote expert.

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
        Forward pass implementation for remote experts.

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
