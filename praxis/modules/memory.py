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
    We also use a form of Locally-Sensitive Hashing (LSH), to reduce the computational
    requirements of the KNN algorithm:
    https://github.com/jinyeom/lsh-knn/tree/main
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.num_dims // config.num_heads
        self.k = 8  # max KNN vectors to lookup
        self.max_memories = 4 * 4096  # max k/v vectors to store
        self.epsilon = 1e-8  # for numerical stability
        # Gating parameter: one gate per head
        self.gate = nn.Parameter(torch.zeros(self.num_heads))
        # Initialize key_memories and value_memories for each head
        multiplier = 2 if config.differential else 1
        # Pre-allocate full memory, circular buffers
        self.write_pos = 0
        self.register_buffer(
            "key_memories",
            torch.zeros(self.num_heads, self.max_memories, self.head_dim * multiplier),
        )
        self.register_buffer(
            "value_memories",
            torch.zeros(self.num_heads, self.max_memories, self.head_dim),
        )

    def forward(
        self, inputs: Tensor, query: Tensor, key: Tensor, value: Tensor, outputs: Tensor
    ) -> Tensor:
        batch_size, seq_len, _ = inputs.size()

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
        Finds k-nearest neighbors using batched processing to reduce peak memory usage.
        """
        if self.key_memories.size(1) == 0:
            return None, None

        # Normalize queries and keys
        queries_norm = F.normalize(queries, p=2, dim=-1, eps=self.epsilon)
        keys_norm = F.normalize(self.key_memories, p=2, dim=-1, eps=self.epsilon)

        batch_size = 512  # Adjust based on available memory
        num_queries = queries_norm.size(1)
        k = min(self.k, self.key_memories.size(1))
        device = queries.device

        # Pre-allocate output tensors
        all_scores = torch.zeros(self.num_heads, num_queries, k, device=device)
        all_indices = torch.zeros(
            self.num_heads, num_queries, k, dtype=torch.long, device=device
        )

        # Process queries in batches
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_queries = queries_norm[
                :, start_idx:end_idx
            ]  # [num_heads, batch, dim]

            # Compute similarities for this batch
            batch_similarities = torch.bmm(
                batch_queries, keys_norm.transpose(1, 2)
            ) / math.sqrt(
                self.head_dim
            )  # [num_heads, batch, num_keys]

            # Get top k for this batch
            batch_scores, batch_indices = batch_similarities.topk(
                k, dim=-1
            )  # [num_heads, batch, k]

            # Store results
            all_scores[:, start_idx:end_idx] = batch_scores
            all_indices[:, start_idx:end_idx] = batch_indices

        return all_scores, all_indices

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
        Updates the memory using a circular buffer approach.
        """
        batch_size = keys.size(1)

        # Calculate positions to write to
        end_pos = self.write_pos + batch_size
        if end_pos <= self.max_memories:
            # Simple case: just write to next positions
            self.key_memories[:, self.write_pos : end_pos] = keys
            self.value_memories[:, self.write_pos : end_pos] = values
        else:
            # Wrap around case: split the write
            first_part = self.max_memories - self.write_pos
            second_part = batch_size - first_part

            # Write first part
            self.key_memories[:, self.write_pos :] = keys[:, :first_part]
            self.value_memories[:, self.write_pos :] = values[:, :first_part]

            # Write second part at beginning
            self.key_memories[:, :second_part] = keys[:, first_part:]
            self.value_memories[:, :second_part] = values[:, first_part:]

        # Update positions
        self.write_pos = (self.write_pos + batch_size) % self.max_memories
