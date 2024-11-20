import math
import random

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
        self.debug = config.debug
        self.num_heads = config.num_heads
        self.head_dim = config.num_dims // config.num_heads
        self.k = 8  # max KNN vectors to lookup
        max_memories = 4 * 4096  # max k/v vectors to store
        self.epsilon = 1e-8  # for numerical stability
        # Gating parameter: one gate per head
        num_query_heads = self.num_heads * config.num_queries
        self.gate = nn.Parameter(torch.zeros(num_query_heads))
        # Initialize key_memories and value_memories for each head
        self.multiplier = 2 if config.differential else 1
        # Pre-allocate full memory banks
        self.register_buffer(
            "key_memories",
            torch.zeros(self.num_heads, max_memories, self.head_dim * self.multiplier),
        )
        self.register_buffer(
            "value_memories",
            torch.zeros(self.num_heads, max_memories, self.head_dim),
        )

    def forward(
        self, inputs: Tensor, query: Tensor, key: Tensor, value: Tensor, outputs: Tensor
    ) -> Tensor:
        batch_size, seq_len, _ = inputs.size()
        num_heads = query.size(1)
        # Prepare queries, keys, and values for memory: [num_heads, Q, dim]
        multiplier = query.size(-1) // self.head_dim
        q = (
            query.view(batch_size, num_heads, seq_len, self.head_dim * multiplier)
            .transpose(0, 1)
            .reshape(num_heads, batch_size * seq_len, self.head_dim * multiplier)
        )  # [num_heads, Q, d_k]
        k = (
            key.view(batch_size, num_heads, seq_len, self.head_dim * multiplier)
            .transpose(0, 1)
            .reshape(num_heads, batch_size * seq_len, self.head_dim * multiplier)
        )  # [num_heads, Q, d_k]
        v = (
            value.view(batch_size, num_heads, seq_len, self.head_dim)
            .transpose(0, 1)
            .reshape(num_heads, batch_size * seq_len, self.head_dim)
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
                num_heads, batch_size, seq_len, self.head_dim
            )  # [num_heads, batch_size, seq_len, dim]

            # Permute to [batch_size, num_heads, seq_len, dim] to align with attn_out
            weighted_memory = weighted_memory.permute(
                1, 0, 2, 3
            ).contiguous()  # [batch_size, num_heads, seq_len, dim]

            # Apply per-head gating
            gate = (
                torch.sigmoid(self.gate).view(1, num_heads, 1, 1).to(outputs.device)
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
        Handles GQA where num_query_heads > num_heads (key/value heads)
        """
        if self.key_memories.size(1) == 0:
            return None, None

        # Normalize queries and keys
        queries_norm = F.normalize(queries, p=2, dim=-1, eps=self.epsilon)
        keys_norm = F.normalize(self.key_memories, p=2, dim=-1, eps=self.epsilon)

        batch_size = 512
        num_query_heads, num_queries, _ = queries_norm.shape
        k = min(self.k, self.key_memories.size(1))
        device = queries.device
        queries_per_key = num_query_heads // self.num_heads

        all_scores = torch.zeros(num_query_heads, num_queries, k, device=device)
        all_indices = torch.zeros(
            num_query_heads, num_queries, k, dtype=torch.long, device=device
        )

        # Process each key head's group of query heads
        for i in range(self.num_heads):
            # Get the query heads for this key head
            query_start = i * queries_per_key
            query_end = (i + 1) * queries_per_key
            group_queries = queries_norm[
                query_start:query_end
            ]  # [queries_per_key, num_queries, dim]

            # Process in batches to manage memory
            for start_idx in range(0, num_queries, batch_size):
                end_idx = min(start_idx + batch_size, num_queries)
                batch_queries = group_queries[
                    :, start_idx:end_idx
                ]  # [queries_per_key, batch_size, dim]

                # Compute similarities for each query in the group
                # Expand keys to match batch_queries shape for bmm
                key_head = (
                    keys_norm[i].unsqueeze(0).expand(batch_queries.size(0), -1, -1)
                )  # [queries_per_key, num_memories, dim]

                batch_similarities = torch.bmm(
                    batch_queries, key_head.transpose(1, 2)
                ) / math.sqrt(self.head_dim)

                # Get top k for this batch
                batch_scores, batch_indices = batch_similarities.topk(k, dim=-1)

                # Store results for this group of query heads
                all_scores[query_start:query_end, start_idx:end_idx] = batch_scores
                all_indices[query_start:query_end, start_idx:end_idx] = batch_indices

        return all_scores, all_indices

    def _get_values(self, indices: Tensor) -> Tensor:
        """
        Retrieves the values corresponding to the nearest neighbors.
        Handles GQA where num_query_heads > num_heads (key/value heads)
        """
        num_query_heads = indices.size(0)
        queries_per_key = num_query_heads // self.num_heads
        gathered_values = []

        # Process each key head's group of query heads
        for i in range(self.num_heads):
            query_start = i * queries_per_key
            query_end = (i + 1) * queries_per_key
            group_indices = indices[query_start:query_end]  # [queries_per_key, Q, k]

            # Expand memory values for this head
            head_memory = self.value_memories[i]  # [num_memories, head_dim]

            # Reshape indices for gather: [queries_per_key * Q, k]
            flat_indices = group_indices.reshape(-1, group_indices.size(-1))

            # Gather values and reshape
            values = head_memory[flat_indices]  # [queries_per_key * Q, k, head_dim]

            # Reshape back to [queries_per_key, Q, k, head_dim]
            head_values = values.reshape(
                group_indices.size(0),  # queries_per_key
                group_indices.size(1),  # Q
                group_indices.size(2),  # k
                self.head_dim,
            )

            gathered_values.append(head_values)

        # Combine all gathered values
        gathered_values = torch.cat(
            gathered_values, dim=0
        )  # [num_query_heads, Q, k, dim]
        return gathered_values

    def _update_memory(self, keys: Tensor, values: Tensor):
        """Updates memory by keeping most surprising/novel information."""
        num_query_heads = keys.size(0)
        queries_per_key = num_query_heads // self.num_heads
        surprise_threshold = 0.5

        # Process each group of queries_per_key heads
        for i in range(queries_per_key):
            # Take the corresponding slice of heads
            start_idx = i * self.num_heads
            end_idx = (i + 1) * self.num_heads
            group_keys = keys[start_idx:end_idx]
            group_values = values[start_idx:end_idx]

            # Replace least useful memories
            batch_keys_norm = F.normalize(group_keys, dim=-1)
            existing_keys_norm = F.normalize(self.key_memories, dim=-1)

            for h in range(self.num_heads):
                # Compare new memories against existing ones
                sims = torch.mm(batch_keys_norm[h], existing_keys_norm[h].t())
                max_sims = sims.max(dim=1)[0]

                # Only consider truly surprising memories
                surprising_indices = torch.where(max_sims < (1 - surprise_threshold))[0]
                if self.debug and random.random() < 0.001:
                    print(f"DEBUG: found {len(surprising_indices)} memories")

                if len(surprising_indices) > 0:
                    # Compare surprising new memories against existing ones
                    relevant_sims = sims[surprising_indices]

                    # Find memories least relevant to our new surprising content
                    min_relevance = relevant_sims.min(dim=0)[
                        0
                    ]  # How relevant is each memory to new content?

                    # Replace least relevant memories
                    _, replace_positions = min_relevance.topk(
                        k=len(surprising_indices),
                        largest=False,  # Take lowest relevance scores
                    )

                    # Replace redundant memories with surprising ones
                    self.key_memories[h, replace_positions] = group_keys[
                        h, surprising_indices
                    ]
                    self.value_memories[h, replace_positions] = group_values[
                        h, surprising_indices
                    ]
