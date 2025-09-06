from typing import Any, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from praxis.recurrent import RECURRENT_REGISTRY
from praxis.routers.smear import SMEAR

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class GRUBlock(nn.Module):
    """
    A GRU block with SMEAR routing for multi-expert processing.
    Cleaner implementation without complex masking.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        """
        Initialize GRU block.

        Args:
            config: Model configuration
        """
        super().__init__()
        # Use num_smear if available, otherwise default to 3
        num_smear = getattr(config, "num_smear", 3)

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.experts = SMEAR(
            config,
            experts=nn.ModuleList(
                [
                    RECURRENT_REGISTRY["gru"](
                        config.hidden_size, expansion_factor=1.25, proj_out=True
                    )
                    for _ in range(num_smear)
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
        Forward pass through the GRU block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            current_state: Optional current recurrent state

        Returns:
            Tuple containing:
                - Output tensor with residual connection
                - None (no past key values for recurrent blocks)
                - New recurrent state
                - Auxiliary loss from SMEAR routing
        """
        # Apply layer norm and process through SMEAR-routed GRU experts
        normed_inputs = self.norm(inputs)
        outputs, new_state, aux_loss = self.experts(normed_inputs, current_state)

        # Apply dropout before residual connection
        outputs = self.dropout(outputs)

        # Apply residual connection
        outputs = outputs + inputs

        return outputs, None, new_state, aux_loss
