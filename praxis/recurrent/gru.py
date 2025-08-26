from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GRU(nn.Module):
    """
    Standard GRU implementation for use in SMEAR routing.
    Cleaner implementation without the complex masking logic.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        dim_out: Optional[int] = None,
        proj_out: Optional[bool] = None,
    ) -> None:
        """
        Initialize GRU module.

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

        # Standard GRU gates: reset and update
        # Input: concat of x[dim] and h[dim_inner] -> gates[dim_inner * 2]
        self.to_gates = nn.Linear(dim + dim_inner, dim_inner * 2, bias=True)
        # New hidden state computation
        # Input: concat of x[dim] and (reset_gate * h)[dim_inner] -> h_candidate[dim_inner]
        self.to_hidden = nn.Linear(dim + dim_inner, dim_inner, bias=True)

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
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Validate input dimension
        if input_dim != self.dim:
            raise ValueError(f"Input dimension {input_dim} doesn't match expected dimension {self.dim}")

        # Initialize hidden state if not provided
        if prev_hidden is None:
            h = torch.zeros(batch_size, self.dim_inner, device=device)
        else:
            # Ensure hidden state has correct shape
            h = prev_hidden.squeeze(1) if prev_hidden.dim() == 3 else prev_hidden

        outputs = []

        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, dim]
            
            # Ensure h has correct shape
            if h.shape[-1] != self.dim_inner:
                # If h has wrong dimensions, reinitialize it
                h = torch.zeros(batch_size, self.dim_inner, device=device)

            # Compute gates
            gates_input = torch.cat([x_t, h], dim=-1)
            gates = self.to_gates(gates_input)
            reset_gate, update_gate = gates.chunk(2, dim=-1)
            reset_gate = torch.sigmoid(reset_gate)
            update_gate = torch.sigmoid(update_gate)

            # Compute candidate hidden state
            hidden_input = torch.cat([x_t, reset_gate * h], dim=-1)
            h_candidate = torch.tanh(self.to_hidden(hidden_input))

            # Update hidden state
            h = (1 - update_gate) * h + update_gate * h_candidate

            outputs.append(h)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, dim_inner]

        # Project output if needed
        output = self.to_out(output)

        # Return output and final hidden state
        next_hidden = h.unsqueeze(1)  # [batch_size, 1, dim_inner]

        return output, next_hidden
