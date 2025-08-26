import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MinGRU(nn.Module):
    """
    This is a log-space implementation of minGRU:
    https://arxiv.org/abs/2410.01201
    https://github.com/lucidrains/minGRU-pytorch/blob/main/minGRU_pytorch/minGRU.py
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 1.0,
        dim_out: Optional[int] = None,
        proj_out: Optional[bool] = None,
    ) -> None:
        """
        Initialize minGRU module.

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

        self.to_hidden_and_gate: nn.Linear = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out: nn.Module = (
            nn.Linear(dim_inner, dim_out, bias=False) if proj_out else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        prev_hidden: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through minGRU.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            prev_hidden: Optional previous hidden state
            input_ids: Optional input token IDs for masking

        Returns:
            Tuple containing:
                - Output tensor
                - Next hidden state
        """
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        # Create pad mask (True where tokens are padding)
        if input_ids is not None:
            pad_mask = input_ids == 0  # Assuming pad token is 0
        else:
            # Create default mask of all False (no padding)
            pad_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool, device=x.device
            )

        if seq_len == 1:
            # Sequential mode
            hidden = g(hidden)
            gate = gate.sigmoid()
            # Reset prev_hidden where we had a pad token
            if prev_hidden is not None and pad_mask.any():
                prev_hidden = prev_hidden.masked_fill(pad_mask.unsqueeze(-1), 0)
            out = (
                torch.lerp(prev_hidden.float(), hidden.float(), gate.float()).to(hidden.dtype)
                if prev_hidden is not None
                else (hidden * gate)
            )
        else:
            # Parallel mode
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if prev_hidden is not None:
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
                # Extend pad_mask for prev_hidden
                if pad_mask.any():
                    pad_mask = F.pad(pad_mask, (1, 0), value=False)

            # Use modified scan that respects boundaries
            out = heinsen_associative_scan_log_with_reset(
                log_coeffs, log_values, pad_mask
            )
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        # Ensure next_prev_hidden is zero if we ended on a pad token
        if pad_mask is not None:
            next_prev_hidden = next_prev_hidden.masked_fill(
                pad_mask[:, -1:].unsqueeze(-1), 0
            )

        out = self.to_out(out)

        return out, next_prev_hidden


# https://github.com/glassroom/heinsen_sequence
def heinsen_associative_scan_log_with_reset(
    log_coeffs: Tensor, log_values: Tensor, pad_mask: Tensor
) -> Tensor:
    """
    Modified Heinsen scan in log space that respects reset boundaries.

    Args:
        log_coeffs: Log coefficients tensor
        log_values: Log values tensor
        pad_mask: Padding mask (True where tokens should be masked)

    Returns:
        Output tensor after scan operation
    """
    # Instead of masking with -inf, we'll modify the coefficients
    batch_size, seq_len, hidden_dim = log_coeffs.shape

    # Create reset boundaries
    reset_mask = F.pad(pad_mask[:, :-1], (1, 0), value=False)
    reset_mask = reset_mask.unsqueeze(-1)

    # Instead of using masked_fill with 0, use a very negative number for log space
    # This effectively makes the coefficient near zero after exp
    penalty = -100
    log_coeffs_masked = torch.where(
        reset_mask, torch.full_like(log_coeffs, penalty), log_coeffs
    )

    # Compute cumulative sum in log space
    a_star = log_coeffs_masked.cumsum(dim=1)

    # For the values, we'll use the same masking technique
    log_values_masked = torch.where(
        reset_mask, torch.full_like(log_values, penalty), log_values
    )

    # Compute the scan
    log_h0_plus_b_star = (log_values_masked - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star

    return log_h.exp()


def g(x: Tensor) -> Tensor:
    """
    Activation function for minGRU.

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


# they enforce the hidden states to be positive
def log_g(x: Tensor) -> Tensor:
    """
    Log-space activation function for minGRU.

    Args:
        x: Input tensor

    Returns:
        Log-space activated tensor
    """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
