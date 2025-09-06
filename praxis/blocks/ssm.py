from typing import Any, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.residuals import RESIDUAL_REGISTRY
from praxis.utils import norm_scaling

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class SSMBlock(nn.Module):
    """
    A simplified State Space Model (SSM) block for causal language modeling.

    This implements a basic S4-style architecture with:
    - Discretized state space dynamics
    - Causal convolution
    - Gated architecture similar to Mamba
    """

    def __init__(self, config: ConfigType, *args: Any, **kwargs: Any) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.state_size = getattr(config, "ssm_state_size", 16)
        self.conv_size = getattr(config, "ssm_conv_size", 4)
        self.expand_factor = getattr(config, "ssm_expand_factor", 2)
        self.dt_rank = getattr(config, "ssm_dt_rank", self.hidden_size // 16)
        self.use_scaler = config.scaled

        # Residual connection
        self.residual = RESIDUAL_REGISTRY.get(config.residual_type)(self.hidden_size)

        # Layer norm
        self.norm = nn.RMSNorm(self.hidden_size, eps=config.epsilon)

        # Input projection with gating
        self.in_proj = nn.Linear(
            self.hidden_size, self.hidden_size * self.expand_factor * 2
        )

        # Causal convolution
        self.conv1d = nn.Conv1d(
            self.hidden_size * self.expand_factor,
            self.hidden_size * self.expand_factor,
            kernel_size=self.conv_size,
            padding=self.conv_size - 1,
            groups=self.hidden_size * self.expand_factor,
        )

        # SSM parameters
        self.x_proj = nn.Linear(
            self.hidden_size * self.expand_factor, self.dt_rank + self.state_size * 2
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.hidden_size * self.expand_factor)

        # Initialize A matrix (state transition) - negative for stability
        A = (
            torch.arange(1, self.state_size + 1)
            .float()
            .repeat(self.hidden_size * self.expand_factor, 1)
        )
        self.A_log = nn.Parameter(torch.log(A))

        # Initialize D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.hidden_size * self.expand_factor))

        # Output projection
        self.out_proj = nn.Linear(
            self.hidden_size * self.expand_factor, self.hidden_size
        )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        current_state: Optional[Tensor] = None,
        current_depth: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, Optional[Tensor], Tensor]:
        """
        Forward pass through the SSM block.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (not used in SSM)
            past_key_values: Not used (for compatibility)
            current_state: Optional SSM hidden state [batch_size, hidden_size * expand_factor, state_size]
            current_depth: Current depth in the network

        Returns:
            Tuple containing:
                - Output tensor
                - None (no past key values)
                - Updated SSM state
                - Auxiliary loss (0 for SSM)
        """
        batch_size, seq_len, _ = inputs.shape

        # Residual connection setup
        residual, beta = self.residual.connect_width(inputs)

        # Normalize input
        x = self.norm(self.residual.format_state(residual))

        if self.use_scaler:
            x = norm_scaling(x, current_depth)

        # Input projection with gating
        x_and_gate = self.in_proj(x)
        x, gate = x_and_gate.chunk(2, dim=-1)

        # Apply convolution (causal)
        x_conv = x.transpose(1, 2)  # [B, D, L]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal masking
        x_conv = x_conv.transpose(1, 2)  # [B, L, D]

        # Apply SiLU activation
        x = F.silu(x_conv)

        # SSM computation
        x_proj_out = self.x_proj(x)
        delta, B, C = x_proj_out.split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )

        # Compute dt (discretization timestep)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, D]

        # Discretize A
        A = -torch.exp(self.A_log)  # [D, N]

        # Initialize state if needed
        if current_state is None:
            current_state = torch.zeros(
                batch_size,
                self.hidden_size * self.expand_factor,
                self.state_size,
                device=inputs.device,
                dtype=inputs.dtype,
            )

        # Fully vectorized SSM using parallel associative scan
        # Memory-efficient implementation without creating L×L matrices

        # Discretize continuous parameters
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, D, N]
        deltaB_x = (
            delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        )  # [B, L, D, N]

        # Use parallel scan algorithm
        # Key insight: we can express the recurrence as an associative binary operation
        # (A1, B1) • (A2, B2) = (A1*A2, A1*B2 + B1)

        # Prepare for parallel scan
        # We'll use PyTorch's built-in operations for efficiency

        # Method: Use cumulative sum in log space for numerical stability
        log_deltaA = deltaA.log()
        log_cumsum_deltaA = log_deltaA.cumsum(dim=1)
        cumsum_deltaA = log_cumsum_deltaA.exp()  # [B, L, D, N]

        # For initial state contribution
        h0_contribution = current_state.unsqueeze(1) * cumsum_deltaA

        # For input contributions, we use the fact that:
        # h[n] = sum_{k=0}^{n} (prod_{j=k+1}^{n} A[j]) * B[k] * x[k]
        # This can be computed as: reverse_cumsum(B*x / cumsum_A) * cumsum_A

        # Prepare shifted cumsum for division
        cumsum_deltaA_shifted = torch.cat(
            [torch.ones_like(cumsum_deltaA[:, :1]), cumsum_deltaA[:, :-1]], dim=1
        )

        # Scale inputs by inverse cumulative product
        deltaB_x_scaled = deltaB_x / cumsum_deltaA_shifted.clamp(min=1e-10)

        # Compute cumulative sum of scaled inputs
        h_inputs_cumsum = deltaB_x_scaled.cumsum(dim=1)

        # Rescale by cumulative product to get final contributions
        h_from_inputs = h_inputs_cumsum * cumsum_deltaA_shifted

        # Combine all contributions
        h = h0_contribution + h_from_inputs  # [B, L, D, N]

        # Update state
        current_state = h[:, -1]  # [B, D, N]

        # Compute outputs
        y = (h * C.unsqueeze(2)).sum(dim=-1)  # [B, L, D]

        # Add skip connection
        y = y + x * self.D

        # Apply gate
        y = y * F.silu(gate)

        # Output projection
        output = self.out_proj(y)

        # Residual connection
        output = self.residual.connect_depth(residual, output, beta)
        output = self.residual.format_state(output)

        return output, None, current_state, torch.tensor(0.0, device=inputs.device)
