import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.encoders import PraxisVAE


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
        self.epsilon = 1e-6  # for numerical stability

        self.chunk_size = 512
        max_memories = self.chunk_size * 8  # max k/v vectors to store
        self.sample_size = 1.0
        # Intended to reduce VRAM, but it doesn't help very much
        # At 32768 memories, memory buffers only use ~0.05GB of VRAM already
        if "subsample" in config.meta:
            max_memories = self.chunk_size * 64
            self.sample_size = 0.25

        head_dim = config.num_dims // config.num_heads
        num_query_heads = self.num_heads * config.num_queries
        self.k = num_query_heads  # max KNN vectors to lookup
        # Gating parameter: one gate per query head
        self.gate = nn.Parameter(
            torch.full((num_query_heads,), 0.0)
        )  # sigmoid(-1) ≈ 0.5
        # Determine if we're using compression
        self.compressed = True if "compressed" in config.meta else False

        memory_dim = head_dim
        if self.compressed:
            memory_dim = int(head_dim * 0.25)
            self.key_vae = PraxisVAE(
                config, input_dim=head_dim, output_dim=memory_dim, beta=0.1
            )
            self.value_vae = PraxisVAE(
                config,
                input_dim=head_dim,
                output_dim=memory_dim,
                beta=0.1,
                requires_projection=True,
            )

        # Initialize key_memories and value_memories for each head
        multiplier = 2 if config.differential else 1
        self.register_buffer(
            "key_memories",
            torch.zeros(self.num_heads, max_memories, memory_dim * multiplier),
        )
        self.register_buffer(
            "value_memories",
            torch.zeros(self.num_heads, max_memories, memory_dim),
        )
        # self.register_buffer(
        #     "key_memories",
        #     torch.randn(self.num_heads, max_memories, memory_dim * multiplier),
        # )
        # self.register_buffer(
        #     "value_memories",
        #     torch.randn(self.num_heads, max_memories, memory_dim),
        # )
        # Memory churn tracking
        self.aux_losses = []
        self.memory_decay = 0.99
        self.register_buffer("memory_churn", torch.zeros(1))

    def get_aux_loss(self):
        if len(self.aux_losses) > 0:
            return self.aux_losses.pop()
        else:
            return 0

    def set_aux_loss(self, value):
        self.aux_losses.append(value)

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

        aux_loss = 0
        if self.compressed:
            # Compress for memory operations - keeping gradients for VAE training
            q_compressed, q_kl = self.key_vae(
                q
            )  # Valid: [num_heads, Q, d_k] -> [batch, seq, feat]
            k_compressed, k_kl = self.key_vae(k)
            v_compressed, v_kl = self.value_vae(v)
            aux_loss = q_kl + k_kl + v_kl
            self.set_aux_loss(aux_loss)

            # Use compressed versions for memory operations
            scores_mem, indices_mem = self._find_knn(q_compressed)
            # Memory values are in compressed form
            memory_values_compressed = self._get_values(
                indices_mem
            )  # [num_query_heads, Q, k, compressed_dim]
            num_query_heads, Q, k, compressed_dim = memory_values_compressed.shape

            # Reshape: [num_query_heads, Q, k, compressed_dim] -> [num_query_heads * Q * k, 1, compressed_dim]
            memory_values_reshaped = memory_values_compressed.view(
                -1, 1, compressed_dim
            )

            # These values are already compressed - send directly to decoder
            memory_values_expanded = self.value_vae.decode(
                memory_values_reshaped,
                compressed_input=True,
                project_to_input=True,
            )

            # Reshape back to original structure
            memory_values = memory_values_expanded.reshape(num_query_heads, Q, k, -1)

            # Update memory with compressed versions
            self._update_memory(k_compressed, v_compressed)
        else:
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
            mean_gate = gate.mean(1).squeeze().item()
            print(f"DEBUG: average memory contribution: {mean_gate:.4f}%")

        # Combine attention and memory outputs using the gate
        combined_output = (
            gate * weighted_memory + (1 - gate) * outputs
        )  # [batch_size, num_heads, seq_len, dim]

        return combined_output, aux_loss

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
        queries_per_key = num_query_heads // num_heads  # Assuming divisible

        # Reshape queries_norm to [num_heads, queries_per_key, num_queries, dim]
        queries_norm = queries_norm.view(
            num_heads, queries_per_key, num_queries, queries_dim
        )

        # If sampling is enabled, sample a subset of keys per head
        if self.sample_size < 1.0:
            num_samples = max(int(keys_norm.size(1) * self.sample_size), 64)
            sampled_key_indices = torch.randint(
                0, keys_norm.size(1), (num_heads, num_samples), device=keys_norm.device
            )

            # Gather sampled keys for each head
            keys_norm_sampled = []
            for h in range(num_heads):
                sampled_indices = sampled_key_indices[h]
                sampled_keys = keys_norm[h, sampled_indices, :]
                keys_norm_sampled.append(sampled_keys)
            keys_norm = torch.stack(keys_norm_sampled, dim=0)
        else:
            num_samples = keys_norm.size(1)
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
                "hqbd,hkd->hqbk", batch_queries, keys_norm
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
        """Updates memory by keeping most surprising/novel information."""
        num_heads = self.num_heads
        num_query_heads = keys.size(0)
        queries_per_key = num_query_heads // num_heads

        # Initialize or get current threshold (stored as buffer to persist between calls)
        if not hasattr(self, "surprise_threshold"):
            self.register_buffer("surprise_threshold", torch.tensor(0.9))

        total_surprising = 0
        total_capacity = self.key_memories.size(1) * num_heads
        total_checked = 0

        # Process each group of queries_per_key heads
        for i in range(queries_per_key):
            start_idx = i * num_heads
            end_idx = (i + 1) * num_heads
            group_keys = keys[start_idx:end_idx]
            group_values = values[start_idx:end_idx]

            batch_keys_norm = F.normalize(group_keys, dim=-1, eps=self.epsilon)
            existing_keys_norm = F.normalize(
                self.key_memories, dim=-1, eps=self.epsilon
            )

            for h in range(num_heads):
                sims = torch.mm(batch_keys_norm[h], existing_keys_norm[h].t())
                max_sims = sims.max(dim=1)[0]
                total_checked += len(max_sims)

                # Update threshold based on current similarities distribution
                target_percentile = 95  # Keep top 5% most surprising
                current_threshold = torch.quantile(max_sims, target_percentile / 100)
                self.surprise_threshold = (
                    0.9 * self.surprise_threshold + 0.1 * current_threshold
                ).clamp(0.5, 0.95)

                # Use current threshold
                surprising_indices = torch.where(max_sims < self.surprise_threshold)[0]
                total_surprising += len(surprising_indices)

                if self.debug and random.random() < 0.001:
                    thresh = self.surprise_threshold.item()
                    mean = max_sims.mean().item()
                    stddv = max_sims.std().item()
                    mn = self.key_memories[h].norm(dim=-1).mean().item()
                    kn = keys[h].norm(dim=-1).mean().item()
                    print(
                        f"DEBUG: memory thresh: {thresh:.3f}, similarities: {mean:.3f} ± {stddv:.3f}"
                    )
                    print(f"DEBUG: memory norm: {mn:.3f}, key norm: {kn:.3f}")

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

                    if self.training and self.debug and random.random() < 0.005:
                        print(f"DEBUG: found {num_replacements} surprising memories")

                    surprising_indices = surprising_indices[
                        :num_replacements
                    ]  # Now takes most surprising ones

                    if num_replacements > 0:
                        # Calculate how many random replacements we'll do
                        percent_random = 0.001
                        num_random = max(
                            1, math.ceil(num_replacements * percent_random)
                        )  # 5% random replacement
                        num_similarity = (
                            num_replacements - num_random
                        )  # Reduce similarity-based by random count

                        # Compare surprising new memories against existing ones
                        relevant_sims = sims[surprising_indices]

                        # Find memories least relevant to our new surprising content
                        min_relevance = relevant_sims.min(dim=0)[0]

                        # Get positions for similarity-based replacement (reduced count)
                        _, replace_positions_sim = min_relevance.topk(
                            k=num_similarity,
                            largest=False,
                        )

                        # Get random positions for random replacement
                        replace_positions_random = torch.randint(
                            0,
                            num_memories,
                            (num_random,),
                            device=self.key_memories.device,
                        )

                        # Combine position indices
                        replace_positions = torch.cat(
                            [replace_positions_sim, replace_positions_random]
                        )

                        # Take corresponding number of surprising indices
                        # (we're still only using num_replacements total values)
                        surprising_indices = surprising_indices[:num_replacements]

                        # Perform replacements
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
