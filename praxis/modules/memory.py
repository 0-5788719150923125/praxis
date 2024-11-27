from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class CompressiveMemory(nn.Module):
    """
    This module implements a simplified version of Infini-Attention, which can offer
    substantial VRAM savings at longer sequence lengths.
    https://arxiv.org/abs/2404.07143
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.num_queries = config.num_queries
        self.num_query_heads = self.num_heads * self.num_queries
        self.head_dim = self.hidden_size // self.num_heads
        self.multiplier = 2 if config.differential else 1
        self.use_delta = True
        self.betas = nn.Parameter(
            torch.zeros(1, self.num_query_heads, 1, self.head_dim)
        )
        self._states_buffer = []
        self.init_state_learnable = True
        if self.init_state_learnable:
            self.init_mem = nn.Parameter(
                torch.randn(
                    1,
                    self.num_query_heads,
                    self.head_dim * self.multiplier,
                    self.head_dim,
                )
            )
            self.init_z = nn.Parameter(
                torch.ones(1, self.num_query_heads, self.head_dim * self.multiplier, 1)
            )
        else:
            self.init_mem = None
            self.init_z = None

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attention_output: Tensor
    ) -> Tensor:
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
        gate = torch.sigmoid(self.betas)
        return gate * memory_output + (1 - gate) * attention_output

    def _init_states(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        if self.init_state_learnable:
            # Use learnable initial states
            memory_states = self.init_mem.expand(batch_size, -1, -1, -1).to(device)
            memory_z = self.init_z.expand(batch_size, -1, -1, -1).to(device)
        else:
            # Use standard initialization
            memory_states = torch.zeros(
                batch_size,
                self.num_query_heads,
                self.head_dim * self.multiplier,
                self.head_dim,
                device=device,
            )
            memory_z = (
                torch.ones(
                    batch_size,
                    self.num_query_heads,
                    self.head_dim * self.multiplier,
                    1,
                    device=device,
                )
                / self.head_dim
            )
        return memory_states, memory_z

    def reset_states(self):
        """Clear the states buffer"""
        self._states_buffer.clear()
