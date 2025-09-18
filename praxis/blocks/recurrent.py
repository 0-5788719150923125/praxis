from typing import Any, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.recurrent import RECURRENT_REGISTRY
from praxis.routers.smear import SMEAR

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class RecurrentBlock(nn.Module):
    """
    A recurrent block type.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        """
        Initialize recurrent block.

        Args:
            config: Model configuration
        """
        super().__init__()
        # Use num_experts for SMEAR routing
        num_experts_for_smear = getattr(config, "num_experts", 3)

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.experts = SMEAR(
            config,
            experts=nn.ModuleList(
                [
                    RECURRENT_REGISTRY["min_gru"](config.hidden_size, proj_out=True)
                    for _ in range(num_experts_for_smear)
                ]
            ),
        )

    def forward(
        self,
        inputs: Tensor,
        current_state: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
        """
        Forward pass through the recurrent block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            current_state: Optional current recurrent state

        Returns:
            Tuple containing:
                - Output tensor
                - None (no past key values)
                - New recurrent state
                - Auxiliary loss
        """
        outputs, new_state, aux_loss = self.experts(self.norm(inputs), current_state)
        return outputs + inputs, None, new_state, aux_loss
