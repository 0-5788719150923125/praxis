from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly, PraxisScatter
from praxis.modules.kan import PraxisKAN
from praxis.modules.peer import PraxisPEER
from praxis.modules.recurrent import PraxisRecurrent
from praxis.routers import ROUTER_REGISTRY

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

EXPERT_REGISTRY: Dict[str, Type[nn.Module]] = {
    "glu": PraxisGLU,
    "kan": PraxisKAN,
    "mlp": PraxisMLP,
    "peer": PraxisPEER,
    "poly": PraxisPoly,
    "recurrent": PraxisRecurrent,
    "scatter": PraxisScatter,
}

EXPERT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "glu": {},
    "kan": {},
    "mlp": {},
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "poly": {},
    "recurrent": {},
    "scatter": {},
}


def get_expert_config(expert: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get configuration for the specified expert.

    Args:
        expert: Expert name as string or config as dictionary

    Returns:
        Expert configuration dictionary

    Raises:
        ValueError: If expert is not a string or dictionary, or if expert name is unknown
    """
    # Handle expert configuration
    if isinstance(expert, str):
        if expert not in EXPERT_CONFIGS:
            raise ValueError(f"Unknown expert type: {expert}")
        return {"type": expert, **EXPERT_CONFIGS[expert]}
    elif isinstance(expert, dict):
        return expert
    else:
        raise ValueError("Expert must be either a string or a dictionary")


class PraxisExpert(nn.Module):
    """
    This class is a wrapper around the orchestration of both local and remote experts.
    TODO: There is some unreliable routing in this class. We need to fix the local/remote parity, while
    also addressing the depth and state handling.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: ConfigType,
        block: Union[nn.Module, bool] = False,
        router: Union[nn.Module, bool] = False,
        is_remote: bool = False,
    ) -> None:
        """
        Initialize expert wrapper.

        Args:
            config: Configuration object with model parameters
            block: Block module to wrap
            router: Router module (or False to use default router)
            is_remote: Whether this is a remote expert
        """
        super().__init__()
        self.is_remote: bool = is_remote
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
    ) -> Union[
        Tuple[Tensor, Optional[Tensor], Optional[Tensor], float], Tuple[Tensor, float]
    ]:
        """
        Forward pass through expert wrapper.

        Args:
            inputs: Input tensor
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/value tensors
            current_state: Optional current state tensor
            current_depth: Current depth in the network
            block_ids: Optional block IDs for structured attention

        Returns:
            For local experts:
                - Hidden states tensor
                - Updated key/value cache
                - Updated state tensor
                - Auxiliary loss value

            For remote experts:
                - Hidden states tensor
                - Auxiliary loss value
        """
        if self.is_remote:
            return self._remote_forward(inputs, attention_mask)
        else:
            return self._local_forward(
                inputs,
                current_state,
                attention_mask,
                past_key_values,
                current_depth,
                block_ids,
            )

    def _local_forward(
        self,
        inputs: Tensor,
        current_state: Optional[Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Tensor],
        current_depth: int,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        """
        Forward pass for local experts.

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

    def _remote_forward(
        self, inputs: Tensor, attention_mask: Optional[Tensor]
    ) -> Tuple[Tensor, float]:
        """
        Forward pass for remote experts.

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
