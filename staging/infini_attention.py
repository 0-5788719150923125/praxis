import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class PraxisMemory(nn.Module):
    """
    This module implements a simplified version of Infini-Attention. Whereas the original
    research would initialize a new "memory state" for every forward pass, and use segmentation
    to process iterations over a sequence length (in chunks), our approach attempts to persist
    memory between entirely different sequences:
    https://arxiv.org/abs/2404.07143
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.epsilon = 1e-8
        self.use_delta = True
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        multiplier = 2 if config.differential else 1
        self.betas = nn.Parameter(torch.ones(self.num_heads, 1, self.head_dim))
        self.memory_states = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim * multiplier, self.head_dim)
        )
        self.memory_z = nn.Parameter(
            torch.ones(self.num_heads, self.head_dim * multiplier) / self.head_dim
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor, output: Tensor):
        # Expand memory states and z to match the batch size
        current_states, current_z = self.memory_states, self.memory_z
        # Blend with intermediate states
        new_states, new_z = self._compute_updates(key, value, current_states, current_z)
        # Retrieve compressed memories from the memory state
        memory_output = self._retrieve_memory(query, new_states, new_z)
        # Accumulated memory states to persist representations between forward passes
        self.memory_states.add(new_states)
        self.memory_z.add(new_z)
        # Combine with attention
        return self._focus_attention(memory_output, output)

    def _compute_updates(self, key, value, current_states, current_z):
        # Compute memory updates
        sigma_k = F.elu(key) + 1.0
        value_update = value
        if self.use_delta:
            # Retrieve compressed memory states
            retrieved_value = torch.matmul(sigma_k, current_states) / (
                torch.matmul(sigma_k, current_z.unsqueeze(-1)) + self.epsilon
            )
            value_update = value - retrieved_value
        # Update the memory states
        memory_update = current_states + torch.matmul(
            sigma_k.transpose(-2, -1), value_update
        )
        # Accumulate states
        z_update = sigma_k.sum(dim=-2)
        memory_z = current_z + z_update
        return memory_update, memory_z

    def _retrieve_memory(self, query, memory_states, memory_z):
        # Retrieve using accumulated state
        sigma_q = F.elu(query) + 1.0
        retrieved_memories = torch.matmul(sigma_q, memory_states)
        norm_factor = torch.matmul(sigma_q, memory_z.unsqueeze(-1)) + self.epsilon
        return retrieved_memories / norm_factor

    def _focus_attention(self, memory_output, attention_output):
        gate = torch.sigmoid(self.betas)
        return gate * memory_output + (1 - gate) * attention_output
