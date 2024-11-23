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

        self.k = 16  # max KNN vectors to lookup
        self.chunk_size = 512
        max_memories = self.chunk_size * 64  # max k/v vectors to store
        self.sample_size = 1.0
        # Intended to reduce VRAM, but it doesn't help very much
        # At 32768 memories, memory buffers only use ~0.05GB of VRAM already
        if "subsample" in config.meta:
            max_memories = self.chunk_size * 64
            self.sample_size = 0.25

        head_dim = config.num_dims // config.num_heads
        num_query_heads = self.num_heads * config.num_queries

        # Gating parameter: one gate per query head
        self.gate = nn.Parameter(torch.zeros(num_query_heads))
        nn.init.normal_(self.gate, mean=0, std=0.01)
        # Initialize key_memories and value_memories for each head
        multiplier = 2 if config.differential else 1
        self.register_buffer(
            "key_memories",
            F.normalize(
                torch.randn(self.num_heads, max_memories, head_dim * multiplier),
                dim=-1,
            ),
        )
        self.register_buffer(
            "value_memories",
            torch.randn(self.num_heads, max_memories, head_dim),
        )
        # Memory churn tracking
        self.memory_decay = 0.99
        self.register_buffer("memory_churn", torch.zeros(1))
        self.register_buffer(
            "update_counts", torch.zeros_like(self.key_memories[:, :, 0])
        )  # [num_heads, num_memories]

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

        # Original non-compressed path
        scores_mem, indices_mem = self._find_knn(q)
        memory_values = self._get_values(indices_mem)
        self._update_memory(k, v)

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

        if self.debug and random.random() < 0.001:
            gate_values = gate.squeeze()
            gate_str = ", ".join([f"{g:.2f}" for g in gate_values.cpu().tolist()])
            print(
                f"DEBUG: head gates: {gate_str} | "
                f"min={gate_values.min().item():.2f} "
                f"max={gate_values.max().item():.2f} "
                f"mean={gate_values.mean().item():.2f}"
            )

        # Combine attention and memory outputs using the gate
        combined_output = (
            gate * weighted_memory + (1 - gate) * outputs
        )  # [batch_size, num_heads, seq_len, dim]

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
            queries, p=2, dim=-1
        )  # [num_query_heads, num_queries, dim]
        keys = self.key_memories

        k = min(self.k, self.key_memories.size(1))
        num_query_heads, num_queries, queries_dim = queries_norm.shape
        num_heads = self.num_heads
        queries_per_key = num_query_heads // num_heads  # Assuming divisible

        # Reshape queries_norm to [num_heads, queries_per_key, num_queries, dim]
        queries_norm = queries_norm.view(
            num_heads, queries_per_key, num_queries, queries_dim
        )

        # If sampling is enabled, sample a subset of keys per head
        if self.sample_size < 1.0:
            num_samples = max(int(keys.size(1) * self.sample_size), 64)
            sampled_key_indices = torch.randint(
                0, keys.size(1), (num_heads, num_samples), device=keys.device
            )

            # Gather sampled keys for each head
            keys_sampled = []
            for h in range(num_heads):
                sampled_indices = sampled_key_indices[h]
                sampled_keys = keys[h, sampled_indices, :]
                keys_sampled.append(sampled_keys)
            keys = torch.stack(keys_sampled, dim=0)
        else:
            num_samples = keys.size(1)
            sampled_key_indices = None  # All keys are used

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

            # Compute similarities with sampled keys
            batch_similarities = torch.einsum(
                "hqbd,hkd->hqbk", batch_queries, keys
            )  # [num_heads, queries_per_key, batch_size, num_samples]

            # Flatten similarities for top-k selection
            similarities_flat = batch_similarities.reshape(
                -1, batch_similarities.size(-1)
            )  # [num_heads * queries_per_key * batch_size, num_samples]

            # Get top k for each query
            batch_scores, batch_indices = similarities_flat.topk(k, dim=-1)
            # batch_scores, batch_indices: [num_query_heads * batch_size, k]

            # Adjust indices if sampling was done
            if sampled_key_indices is not None:
                # Map indices back to original key indices
                adjusted_indices = torch.zeros_like(batch_indices)
                for h in range(num_heads):
                    # Determine the range of query indices for this head
                    head_start = h * queries_per_key * (end_idx - start_idx)
                    head_end = (h + 1) * queries_per_key * (end_idx - start_idx)

                    # Get the batch indices for this head
                    h_batch_indices = batch_indices[head_start:head_end, :]
                    h_sampled_indices = sampled_key_indices[h][h_batch_indices]
                    adjusted_indices[head_start:head_end, :] = h_sampled_indices
                batch_indices = adjusted_indices

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
        """
        Simple FIFO memory update - newest entries replace oldest ones.
        """
        batch_size = keys.size(1)
        max_memories = self.key_memories.size(1)

        # For each head
        for h in range(self.num_heads):
            # Calculate how many new entries we can add
            space_available = min(batch_size, max_memories)

            # Roll the existing memories back
            self.key_memories[h] = torch.roll(
                self.key_memories[h], shifts=space_available, dims=0
            )
            self.value_memories[h] = torch.roll(
                self.value_memories[h], shifts=space_available, dims=0
            )
            # Add new memories at the start
            self.key_memories[h, :space_available] = F.normalize(
                keys[h, :space_available], dim=-1
            )
            self.value_memories[h, :space_available] = values[h, :space_available]

    # @torch.no_grad()
    # def _update_memory(self, keys: Tensor, values: Tensor):
    #     """
    #     Updates memory by replacing least useful memories with novel information.
    #     Uses similarity scores to identify both novel content and redundant memories.
    #     """
    #     num_heads = self.num_heads
    #     num_query_heads = keys.size(0)
    #     queries_per_key = num_query_heads // num_heads

    #     # Initialize thresholds if not already present
    #     if not hasattr(self, "redundancy_threshold"):
    #         self.register_buffer("redundancy_threshold", torch.tensor(0.95))
    #     if not hasattr(self, "surprise_threshold"):
    #         self.register_buffer("surprise_threshold", torch.tensor(0.80))

    #     max_memories = self.key_memories.size(1)
    #     max_replacements = int(max_memories * 0.01)  # Cap at 1% per update

    #     total_memories = max_memories * num_heads
    #     total_surprising = 0
    #     total_processed = 0

    #     # Process each group of heads
    #     for i in range(queries_per_key):
    #         start_idx = i * num_heads
    #         end_idx = (i + 1) * num_heads
    #         group_keys = F.normalize(keys[start_idx:end_idx], dim=-1)
    #         group_values = values[start_idx:end_idx]

    #         # For each batch/sequence position, track how many tokens were processed
    #         total_processed += group_keys.size(1) * num_heads

    #         # Update memories for each head
    #         for h in range(num_heads):
    #             # Compute similarities between new keys and stored memories
    #             similarities = torch.mm(group_keys[h], self.key_memories[h].t())
    #             max_similarities = similarities.max(dim=1)[0]

    #             # Update running statistics for both thresholds
    #             low_percentile = 5  # Bottom 5% are "surprising"
    #             high_percentile = 95  # Top 5% are "redundant"

    #             current_low_sim = torch.quantile(max_similarities, low_percentile / 100)
    #             current_high_sim = torch.quantile(
    #                 max_similarities, high_percentile / 100
    #             )

    #             # Update both thresholds with exponential moving average
    #             self.surprise_threshold = (
    #                 0.9 * self.surprise_threshold + 0.1 * current_low_sim
    #             ).clamp(0.3, 0.8)

    #             self.redundancy_threshold = (
    #                 0.9 * self.redundancy_threshold + 0.1 * current_high_sim
    #             ).clamp(0.5, 0.95)

    #             # Find novel content
    #             novel_mask = max_similarities < self.surprise_threshold
    #             novel_indices = torch.where(novel_mask)[0]

    #             # Track surprises relative to batch size
    #             total_surprising += len(novel_indices)

    #             # Find redundant memories (check similarity to all input keys)
    #             memory_similarities = torch.mm(self.key_memories[h], group_keys[h].t())
    #             memory_max_sims = memory_similarities.max(dim=1)[0]
    #             redundant_mask = memory_max_sims >= self.redundancy_threshold
    #             redundant_indices = torch.where(redundant_mask)[0]

    #             if len(novel_indices) == 0 or len(redundant_indices) == 0:
    #                 continue

    #             # Sort by similarity scores
    #             novel_scores = max_similarities[novel_indices]
    #             sorted_novel = novel_indices[torch.argsort(novel_scores)]

    #             redundant_scores = memory_max_sims[redundant_indices]
    #             sorted_redundant = redundant_indices[
    #                 torch.argsort(redundant_scores, descending=True)
    #             ]

    #             # Determine number of replacements
    #             num_replacements = min(
    #                 len(sorted_novel), len(sorted_redundant), max_replacements
    #             )

    #             # Update memories
    #             replace_slots = sorted_redundant[:num_replacements]
    #             new_content = sorted_novel[:num_replacements]

    #             self.key_memories[h, replace_slots] = group_keys[h, new_content]
    #             self.value_memories[h, replace_slots] = group_values[h, new_content]

    #             # Reset update counts for replaced memories
    #             if self.training and self.debug:
    #                 self.update_counts[h, replace_slots] = 0

    #             # Debug logging
    #             if self.debug and random.random() < 0.001:
    #                 print(
    #                     f"DEBUG: head #{h} replaced {num_replacements} memories, "
    #                     f"redundant_thresh={self.redundancy_threshold:.3f}, "
    #                     f"surprise_thresh={self.surprise_threshold:.3f}"
    #                 )

    #     # Calculate normalized memory surprise (0-100%)
    #     churn_percent = min(100.0, (total_surprising / total_processed) * 100)
    #     self.memory_churn.mul_(self.memory_decay).add_(
    #         churn_percent * (1 - self.memory_decay)
    #     )

    #     # Debug age statistics
    #     if self.training and self.debug:
    #         self.update_counts += 1
    #         if random.random() < 0.001:
    #             counts = self.update_counts.view(-1)

    #             age_buckets = {
    #                 "new": (0, 1000),  # Recently added
    #                 "settled": (1000, 10000),  # Moderately stable
    #                 "mature": (10000, 100000),  # Well-established
    #                 "permanent": (100000, None),  # Very stable memories
    #             }

    #             percentages = []
    #             for name, (min_age, max_age) in age_buckets.items():
    #                 if max_age is None:
    #                     mask = counts >= min_age
    #                 else:
    #                     mask = (counts >= min_age) & (counts < max_age)
    #                 percentage = mask.float().mean() * 100
    #                 percentages.append(f"{name}(<{max_age or '∞'})={percentage:.1f}%")

    #             print("DEBUG: memory age:", ", ".join(percentages))
