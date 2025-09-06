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
            bias=True,
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
        # Input validation with detailed error messages
        if x.dim() != 3:
            raise ValueError(
                f"GRU: Expected 3D input tensor, got {x.dim()}D with shape {x.shape}"
            )

        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # Validate input dimension
        if input_dim != self.dim:
            raise ValueError(
                f"GRU: Input dimension mismatch. Expected {self.dim}, got {input_dim}. "
                f"Input shape: {x.shape}"
            )

        # Prepare hidden state for nn.GRU (expects [num_layers, batch_size, hidden_size])
        if prev_hidden is None:
            h = None  # Let GRU initialize it with zeros
        else:
            # Check if we have a properly shaped hidden state
            valid_state = False

            if prev_hidden.dim() == 2:
                # Expected: [batch_size, dim_inner]
                if prev_hidden.shape == (batch_size, self.dim_inner):
                    h = prev_hidden.unsqueeze(0)  # -> [1, batch_size, dim_inner]
                    valid_state = True
            elif prev_hidden.dim() == 3:
                # Check various 3D formats
                if prev_hidden.shape == (batch_size, 1, self.dim_inner):
                    # [batch_size, 1, dim_inner] -> [1, batch_size, dim_inner]
                    h = prev_hidden.transpose(0, 1).contiguous()
                    valid_state = True
                elif prev_hidden.shape == (1, batch_size, self.dim_inner):
                    # Already in correct shape
                    h = prev_hidden
                    valid_state = True

            # If we don't have a valid state, reinitialize
            if not valid_state:
                h = None

        # Ensure h is on the same device and dtype as x if not None
        if h is not None:
            if h.device != x.device:
                h = h.to(x.device)
            if h.dtype != x.dtype:
                h = h.to(x.dtype)

        # Run through PyTorch's GRU
        output, h_final = self.gru(x, h)

        # Project output if needed
        output = self.to_out(output)

        # Convert hidden state back to expected format [batch_size, 1, dim_inner]
        next_hidden = h_final.transpose(0, 1).contiguous()  # [batch_size, 1, dim_inner]

        return output, next_hidden
