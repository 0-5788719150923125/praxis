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
        self.epsilon = 1e-11  # for numerical stability
        self.chunk_size = 512
        max_memories = self.chunk_size * 8  # max k/v vectors to store
        head_dim = config.num_dims // config.num_heads
        num_query_heads = self.num_heads * config.num_queries
        self.k = num_query_heads  # max KNN vectors to lookup
        # Gating parameter: one gate per query head
        self.gate = nn.Parameter(
            torch.full((num_query_heads,), -1.0)
        )  # sigmoid(-1) ≈ 0.27
        # Initialize key_memories and value_memories for each head
        multiplier = 2 if config.differential else 1
        self.register_buffer(
            "key_memories",
            torch.randn(self.num_heads, max_memories, head_dim * multiplier),
        )
        self.register_buffer(
            "value_memories",
            torch.randn(self.num_heads, max_memories, head_dim),
        )
        # Add memory churn tracking
        self.memory_decay = 0.99  # EMA decay factor
        self.register_buffer("memory_churn", torch.zeros(1))

    def forward(
        self, inputs: Tensor, query: Tensor, key: Tensor, value: Tensor, outputs: Tensor
    ) -> Tensor:
        batch_size, num_heads, seq_len, query_dim = query.shape
        # Prepare queries, keys, and values for memory: [num_heads, Q, dim]
        q = query.transpose(0, 1).reshape(
            num_heads, batch_size * seq_len, -1
        )  # [num_heads, Q, d_k]
        k = key.transpose(0, 1).reshape(
            num_heads, batch_size * seq_len, -1
        )  # [num_heads, Q, d_k]
        v = value.transpose(0, 1).reshape(
            num_heads, batch_size * seq_len, -1
        )  # [num_heads, Q, d_k]

        # Look up KNN memories without tracking gradients
        scores_mem, indices_mem = self._find_knn(q)

        # Retrieve memory values without tracking gradients
        memory_values = self._get_values(indices_mem)  # [num_heads, Q, k, dim]

        # Compute weighted sum: [num_heads, Q, dim]
        weighted_memory = memory_values * scores_mem.unsqueeze(
            -1
        )  # [num_heads, Q, k, dim]
        weighted_memory = weighted_memory.sum(dim=2)  # [num_heads, Q, dim]

        # Reshape to [num_heads, batch_size, seq_len, dim]
        weighted_memory = weighted_memory.view(
            batch_size, num_heads, seq_len, -1
        )  # [num_heads, batch_size, seq_len, dim]

        # Apply per-head gating
        gate = torch.sigmoid(self.gate).view(1, num_heads, 1, 1)  # [1, num_heads, 1, 1]

        # Combine attention and memory outputs using the gate
        combined_output = (
            gate * weighted_memory + (1 - gate) * outputs
        )  # [batch_size, num_heads, seq_len, dim]

        # Update memory with current keys and values without tracking gradients
        self._update_memory(k, v)  # [num_heads, Q, dim]

        return combined_output

    def get_metrics(self):
        return {"churn": self.memory_churn.item()}

    @torch.no_grad()
    def _find_knn(self, queries: Tensor) -> tuple:
        """
        Finds k-nearest neighbors using batched processing to reduce peak memory usage.
        Handles GQA where num_query_heads > num_heads (key/value heads)
        """
        # Normalize queries and keys
        queries_norm = F.normalize(
            queries, p=2, dim=-1, eps=self.epsilon
        )  # [num_query_heads, num_queries, dim]
        keys_norm = F.normalize(
            self.key_memories, p=2, dim=-1, eps=self.epsilon
        )  # [num_heads, num_memories, dim]

        k = min(self.k, self.key_memories.size(1))
        num_query_heads, num_queries, queries_dim = queries_norm.shape
        num_heads = self.num_heads
        queries_per_key = num_query_heads // num_heads  # Assume divisible

        # Reshape queries_norm to [num_heads, queries_per_key, num_queries, dim]
        queries_norm = queries_norm.view(
            num_heads, queries_per_key, num_queries, queries_dim
        )

        all_scores = torch.zeros(num_query_heads, num_queries, k, device=queries.device)
        all_indices = torch.zeros(
            num_query_heads, num_queries, k, dtype=torch.long, device=queries.device
        )

        # Process in batches to manage memory
        for start_idx in range(0, num_queries, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_queries)
            batch_queries = queries_norm[
                :, :, start_idx:end_idx, :
            ]  # [num_heads, queries_per_key, batch_size, dim]

            # Compute similarities
            batch_similarities = torch.einsum(
                "hqbd,hnd->hqbn", batch_queries, keys_norm
            )  # [num_heads, queries_per_key, batch_size, num_memories]

            # Flatten similarities for top-k selection
            similarities_flat = batch_similarities.reshape(
                -1, batch_similarities.size(-1)
            )  # [num_heads * queries_per_key * batch_size, num_memories]

            # Get top k for each query
            batch_scores, batch_indices = similarities_flat.topk(k, dim=-1)
            # batch_scores, batch_indices: [num_heads * queries_per_key * batch_size, k]

            # Reshape back to [num_query_heads, batch_size, k]
            batch_scores = batch_scores.view(num_query_heads, end_idx - start_idx, k)
            batch_indices = batch_indices.view(num_query_heads, end_idx - start_idx, k)

            # Store results
            all_scores[:, start_idx:end_idx, :] = batch_scores
            all_indices[:, start_idx:end_idx, :] = batch_indices

        return all_scores, all_indices

    @torch.no_grad()
    def _get_values(self, indices: Tensor) -> Tensor:
        """
        Retrieves the values corresponding to the nearest neighbors.
        Handles GQA where num_query_heads > num_heads (key/value heads)
        """
        num_query_heads, Q, k = indices.size()
        queries_per_key = num_query_heads // self.num_heads  # Assuming divisible

        # Compute key head indices for each query head
        key_head_indices = (
            torch.arange(num_query_heads, device=indices.device) // queries_per_key
        )  # [num_query_heads]

        # Expand key_head_indices to match indices shape
        key_head_indices = key_head_indices.view(num_query_heads, 1, 1).expand(
            -1, Q, k
        )  # [num_query_heads, Q, k]

        # Flatten key_head_indices and indices
        flat_key_head_indices = key_head_indices.reshape(
            -1
        )  # [num_query_heads * Q * k]
        flat_indices = indices.reshape(-1)  # [num_query_heads * Q * k]

        # Gather values
        # self.value_memories is [num_heads, num_memories, head_dim]
        # values will be [num_query_heads * Q * k, head_dim]
        values = self.value_memories[
            flat_key_head_indices, flat_indices
        ]  # [N, head_dim]

        # Reshape values to [num_query_heads, Q, k, head_dim]
        gathered_values = values.view(num_query_heads, Q, k, -1)

        return gathered_values

    @torch.no_grad()
    def _update_memory(self, keys: Tensor, values: Tensor):
        """Updates memory by keeping most surprising/novel information."""
        num_heads = self.num_heads
        num_query_heads = keys.size(0)
        queries_per_key = num_query_heads // num_heads
        surprise_threshold = 0.5

        # Track total surprising memories this batch
        total_surprising = 0
        total_capacity = self.key_memories.size(1) * num_heads

        # Process each group of queries_per_key heads
        for i in range(queries_per_key):
            # Take the corresponding slice of heads
            start_idx = i * num_heads
            end_idx = (i + 1) * num_heads
            group_keys = keys[start_idx:end_idx]
            group_values = values[start_idx:end_idx]

            # Replace least useful memories
            batch_keys_norm = F.normalize(group_keys, dim=-1, eps=self.epsilon)
            existing_keys_norm = F.normalize(
                self.key_memories, dim=-1, eps=self.epsilon
            )

            for h in range(num_heads):
                # Compare new memories against existing ones
                sims = torch.mm(batch_keys_norm[h], existing_keys_norm[h].t())
                max_sims = sims.max(dim=1)[0]

                # Count ALL surprising memories before capping
                surprising_indices = torch.where(max_sims < (1 - surprise_threshold))[0]
                total_surprising += len(
                    surprising_indices
                )  # Use full count, not capped

                # Sort surprising_indices by their similarity scores
                if len(surprising_indices) > 0:
                    # Sort by similarity (ascending), so most surprising (lowest similarity) first
                    similarities = max_sims[surprising_indices]
                    sorted_indices = torch.argsort(similarities)
                    surprising_indices = surprising_indices[sorted_indices]

                    # Cap replacements for actual memory update
                    num_memories = existing_keys_norm[h].size(0)
                    update_cap = int(num_memories * 0.01)
                    num_replacements = min(
                        len(surprising_indices), num_memories, update_cap
                    )
                    surprising_indices = surprising_indices[
                        :num_replacements
                    ]  # Now takes most surprising ones

                    if self.training and self.debug and random.random() < 0.005:
                        print(
                            f"DEBUG: found {len(surprising_indices)} surprising memories"
                        )

                    if num_replacements > 0:
                        # Compare surprising new memories against existing ones
                        relevant_sims = sims[surprising_indices]

                        # Find memories least relevant to our new surprising content
                        min_relevance = relevant_sims.min(dim=0)[
                            0
                        ]  # How relevant is each memory to new content?

                        # Replace least relevant memories
                        _, replace_positions = min_relevance.topk(
                            k=num_replacements,
                            largest=False,  # Take lowest relevance scores
                        )

                        # Replace redundant memories with surprising ones
                        self.key_memories[h, replace_positions] = group_keys[
                            h, surprising_indices
                        ]
                        self.value_memories[h, replace_positions] = group_values[
                            h, surprising_indices
                        ]

        # Update memory churn metric
        churn_percent = (total_surprising / total_capacity) * 100
        self.memory_churn.mul_(self.memory_decay).add_(
            churn_percent * (1 - self.memory_decay)
        )
