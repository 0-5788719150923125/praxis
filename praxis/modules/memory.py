import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class PraxisMemory(nn.Module):
    """
    This code implements a simple, non-differential, external memory module, which
    performs a KNN lookup of previous keys/values used in the attention mechanism.
    Inspired by Memorizing Transformers:
    https://arxiv.org/abs/2203.08913
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.num_dims // config.num_heads
        self.k = 16  # max KNN vectors to lookup
        self.max_memories = 1024  # max k/v vectors to store
        self.epsilon = 1e-8  # for numerical stability
        # Gating parameter: one gate per head
        self.gate = nn.Parameter(torch.zeros(self.num_heads))
        # Initialize key_memories and value_memories for each head
        multiplier = 2 if config.differential else 1
        self.register_buffer(
            "key_memories", torch.empty(self.num_heads, 0, self.head_dim * multiplier)
        )
        self.register_buffer(
            "value_memories", torch.empty(self.num_heads, 0, self.head_dim)
        )

    def forward(
        self, inputs: Tensor, query: Tensor, key: Tensor, value: Tensor, outputs: Tensor
    ) -> Tensor:
        batch_size, seq_len, d_model = inputs.size()

        # Prepare queries, keys, and values for memory: [num_heads, Q, dim]
        multiplier = query.size(-1) // self.head_dim
        q = (
            query.view(batch_size, self.num_heads, seq_len, self.head_dim * multiplier)
            .transpose(0, 1)
            .reshape(self.num_heads, batch_size * seq_len, self.head_dim * multiplier)
        )  # [num_heads, Q, d_k]
        k = (
            key.view(batch_size, self.num_heads, seq_len, self.head_dim * multiplier)
            .transpose(0, 1)
            .reshape(self.num_heads, batch_size * seq_len, self.head_dim * multiplier)
        )  # [num_heads, Q, d_k]
        v = (
            value.view(batch_size, self.num_heads, seq_len, self.head_dim)
            .transpose(0, 1)
            .reshape(self.num_heads, batch_size * seq_len, self.head_dim)
        )  # [num_heads, Q, d_k]

        # Detach q, k, v for non-differentiable memory operations
        q_detached = q.detach()
        k_detached = k.detach()
        v_detached = v.detach()

        # Look up KNN memories without tracking gradients
        with torch.no_grad():
            scores_mem, indices_mem = self._find_knn(q_detached)

        # If no memory found, use standard attention output
        combined_output = outputs  # [batch_size, num_heads, seq_len, dim]
        if scores_mem is not None:
            # Retrieve memory values without tracking gradients
            with torch.no_grad():
                memory_values = self._get_values(indices_mem)  # [num_heads, Q, k, dim]

            # Compute weighted sum: [num_heads, Q, dim]
            # Note: scores_mem is detached; no gradients will flow through it
            weighted_memory = memory_values * scores_mem.unsqueeze(
                -1
            )  # [num_heads, Q, k, dim]
            weighted_memory = weighted_memory.sum(dim=2)  # [num_heads, Q, dim]

            # Reshape to [num_heads, batch_size, seq_len, dim]
            weighted_memory = weighted_memory.view(
                self.num_heads, batch_size, seq_len, self.head_dim
            )  # [num_heads, batch_size, seq_len, dim]

            # Permute to [batch_size, num_heads, seq_len, dim] to align with attn_out
            weighted_memory = weighted_memory.permute(
                1, 0, 2, 3
            ).contiguous()  # [batch_size, num_heads, seq_len, dim]

            # Apply per-head gating
            gate = (
                torch.sigmoid(self.gate)
                .view(1, self.num_heads, 1, 1)
                .to(outputs.device)
            )  # [1, num_heads, 1, 1]

            output_dim = outputs.size(-1)
            weighted_memory = weighted_memory[..., :output_dim]

            # Combine attention and memory outputs using the gate
            combined_output = (
                gate * weighted_memory + (1 - gate) * outputs
            )  # [batch_size, num_heads, seq_len, dim]

        # Update memory with current keys and values without tracking gradients
        with torch.no_grad():
            self._update_memory(
                k_detached, v_detached
            )  # Correct dimensions: [num_heads, Q, dim]

        return combined_output

    def _find_knn(self, queries: Tensor) -> tuple:
        """
        Finds the k-nearest neighbors for each query across all heads using cosine similarity.
        """
        if self.key_memories.size(1) == 0:
            return None, None

        # Normalize queries and keys using F.normalize for numerical stability
        queries_norm = F.normalize(
            queries, p=2, dim=-1, eps=self.epsilon
        )  # [num_heads, Q, dim]
        keys_norm = F.normalize(
            self.key_memories, p=2, dim=-1, eps=self.epsilon
        )  # [num_heads, K, dim]

        # Compute cosine similarity: [num_heads, Q, K]
        # Since vectors are normalized, cosine similarity is equivalent to the dot product
        similarities = torch.bmm(queries_norm, keys_norm.transpose(1, 2)) / math.sqrt(
            self.head_dim
        )

        # Get top-k similarities and their indices
        k = min(self.k, self.key_memories.size(1))
        scores, indices = similarities.topk(k, dim=-1)  # [num_heads, Q, k]

        return scores, indices

    def _get_values(self, indices: Tensor) -> Tensor:
        """
        Retrieves the values corresponding to the nearest neighbors.
        """
        # Gather values for each head
        gathered_values = torch.gather(
            self.value_memories.unsqueeze(1).expand(-1, indices.size(1), -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
        )  # [num_heads, Q, k, dim]
        return gathered_values

    def _update_memory(self, keys: Tensor, values: Tensor):
        """
        Updates the memory with new keys and values.
        """
        # Concatenate new keys and values
        self.key_memories = torch.cat(
            [self.key_memories, keys], dim=1
        )  # [num_heads, K_new, hidden_size]
        self.value_memories = torch.cat(
            [self.value_memories, values], dim=1
        )  # [num_heads, K_new, hidden_size]

        # Trim memory if exceeding max_memories
        if self.key_memories.size(1) > self.max_memories:
            excess = self.key_memories.size(1) - self.max_memories
            self.key_memories = self.key_memories[:, excess:, :]
            self.value_memories = self.value_memories[:, excess:, :]
