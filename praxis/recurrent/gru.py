from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GRU(nn.Module):
    """
    GRU wrapper using PyTorch's native implementation for use in SMEAR routing.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        dim_out: Optional[int] = None,
        proj_out: Optional[bool] = None,
    ) -> None:
        """
        Initialize GRU module using PyTorch's native GRU.

        Args:
            dim: Input dimension
            expansion_factor: Factor to scale hidden dimension
            dim_out: Output dimension (defaults to input dimension)
            proj_out: Whether to project output (defaults to True if expansion_factor != 1.0)
        """
        super().__init__()
        
        dim_out = dim_out or dim
        dim_inner = int(dim * expansion_factor)
        proj_out = proj_out if proj_out is not None else expansion_factor != 1.0
        
        # Use PyTorch's native GRU
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim_inner,
            num_layers=1,
            batch_first=True,
            bias=True
        )
        
        self.to_out = (
            nn.Linear(dim_inner, dim_out, bias=False) if proj_out else nn.Identity()
        )
        
        self.dim = dim
        self.dim_inner = dim_inner
    
    def forward(
        self,
        x: Tensor,
        prev_hidden: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through GRU.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            prev_hidden: Optional previous hidden state [batch_size, 1, dim_inner]
            
        Returns:
            Tuple containing:
                - Output tensor [batch_size, seq_len, dim_out]
                - Next hidden state [batch_size, 1, dim_inner]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Prepare hidden state for nn.GRU (expects [num_layers, batch_size, hidden_size])
        if prev_hidden is None:
            h = torch.zeros(1, batch_size, self.dim_inner, device=device)
        else:
            # Convert from [batch_size, 1, dim_inner] to [1, batch_size, dim_inner]
            if prev_hidden.dim() == 3:
                h = prev_hidden.transpose(0, 1).contiguous()  # [1, batch_size, dim_inner]
            elif prev_hidden.dim() == 2:
                h = prev_hidden.unsqueeze(0)  # [1, batch_size, dim_inner]
            else:
                h = prev_hidden
                
            # Ensure correct shape
            if h.shape != (1, batch_size, self.dim_inner):
                h = torch.zeros(1, batch_size, self.dim_inner, device=device)
        
        # Run through PyTorch's GRU
        output, h_final = self.gru(x, h)
        
        # Project output if needed
        output = self.to_out(output)
        
        # Convert hidden state back to expected format [batch_size, 1, dim_inner]
        next_hidden = h_final.transpose(0, 1).contiguous()  # [batch_size, 1, dim_inner]
        
        return output, next_hidden