from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import device as torch_device

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PraxisCompressiveMemory(nn.Module):
    """
    This module implements a simplified version of Infini-Attention, which can offer
    substantial VRAM savings at longer sequence lengths.
    https://arxiv.org/abs/2404.07143
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize compressive memory module.

        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.num_queries: int = config.num_queries
        self.num_query_heads: int = self.num_heads * self.num_queries
        self.head_dim: int = config.head_size
        self.use_delta: bool = True
        self.betas: nn.Parameter = nn.Parameter(
            torch.zeros(1, self.num_query_heads, 1, self.head_dim)
        )
        self._states_buffer: List[Tuple[Tensor, Tensor]] = []
        self.init_state_learnable: bool = True

        if self.init_state_learnable:
            self.init_mem: Optional[nn.Parameter] = nn.Parameter(
                torch.randn(
                    1,
                    self.num_query_heads,
                    self.head_dim,
                    self.head_dim,
                )
            )
            self.init_z: Optional[nn.Parameter] = nn.Parameter(
                torch.ones(1, self.num_query_heads, self.head_dim, 1)
            )
        else:
            self.init_mem: Optional[nn.Parameter] = None
            self.init_z: Optional[nn.Parameter] = None

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attention_output: Tensor
    ) -> Tensor:
        """
        Forward pass for compressive memory.

        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            attention_output: Attention output tensor

        Returns:
            Blended output tensor after applying memory mechanism
        """
        batch_size = q.size(0)

        # Get states - either initialize or pop from buffer
        if not self._states_buffer:
            memory_states, memory_z = self._init_states(batch_size, q.device)
        else:
            memory_states, memory_z = self._states_buffer.pop()

        # Compute memory output
        sigma_q = F.elu(q) + 1.0
        memory_output = (sigma_q @ memory_states) / (sigma_q @ memory_z)

        # Compute updates
        sigma_k = F.elu(k) + 1.0
        if self.use_delta:
            retrieved = (sigma_k @ memory_states) / (sigma_k @ memory_z)
            value_delta = v - retrieved
            new_states = memory_states + sigma_k.transpose(-2, -1) @ value_delta
        else:
            new_states = memory_states + sigma_k.transpose(-2, -1) @ v

        new_z = memory_z + sigma_k.sum(dim=-2, keepdim=True).transpose(-2, -1)

        # Store single new state
        self._states_buffer.append((new_states, new_z))

        return self._blend_outputs(memory_output, attention_output)

    def _blend_outputs(self, memory_output: Tensor, attention_output: Tensor) -> Tensor:
        """
        Blend memory output with attention output using learned gates.

        Args:
            memory_output: Output from memory mechanism
            attention_output: Output from attention mechanism

        Returns:
            Blended output tensor
        """
        gate = torch.sigmoid(self.betas)
        # if attention_output.dim() == 3:
        #     attention_output = attention_output.unsqueeze(1)
        # print(gate.shape, memory_output.shape, attention_output.shape)
        return gate * memory_output + (1 - gate) * attention_output

    def _init_states(
        self, batch_size: int, device: torch_device
    ) -> Tuple[Tensor, Tensor]:
        """
        Initialize memory states.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple containing:
                - Memory states tensor
                - Memory normalization tensor
        """
        if (
            self.init_state_learnable
            and self.init_mem is not None
            and self.init_z is not None
        ):
            # Use learnable initial states
            memory_states = self.init_mem.expand(batch_size, -1, -1, -1).to(device)
            memory_z = self.init_z.expand(batch_size, -1, -1, -1).to(device)
        else:
            # Use standard initialization
            memory_states = torch.zeros(
                batch_size,
                self.num_query_heads,
                self.head_dim,
                self.head_dim,
                device=device,
            )
            memory_z = (
                torch.ones(
                    batch_size,
                    self.num_query_heads,
                    self.head_dim,
                    1,
                    device=device,
                )
                / self.head_dim
            )
        return memory_states, memory_z

    def reset_states(self) -> None:
        """Clear the states buffer"""
        self._states_buffer.clear()
