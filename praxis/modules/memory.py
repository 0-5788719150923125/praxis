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
        self.k = 16  # max KNN vectors to lookup
        self.max_memories = 8192  # max k/v vectors to store
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
        Finds the k-nearest neighbors for each query across all heads using cosine similarity.
        """
        if self.key_memories.size(1) == 0:
            return None, None

        # Normalize queries and keys
        queries_norm = F.normalize(queries, p=2, dim=-1, eps=self.epsilon)
        keys_norm = F.normalize(self.key_memories, p=2, dim=-1, eps=self.epsilon)

        # Compute cosine similarity: [num_heads, Q, K]
        # Since vectors are normalized, cosine similarity is equivalent to the dot product
        similarities = torch.bmm(queries_norm, keys_norm.transpose(1, 2)) / math.sqrt(
            self.head_dim
        )
        k = min(self.k, self.key_memories.size(1))
        scores, indices = similarities.topk(k, dim=-1)
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


# import math

# import faiss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoConfig


# class PraxisMemory(nn.Module):
#     """
#     This code implements a simple, non-differential, external memory module, which
#     performs a KNN lookup of previous keys/values used in the attention mechanism.
#     Inspired by Memorizing Transformers:
#     https://arxiv.org/abs/2203.08913
#     We also use a form of Locally-Sensitive Hashing (LSH), to reduce the computational
#     requirements of the KNN algorithm:
#     https://github.com/jinyeom/lsh-knn/tree/main
#     """

#     __version__ = "0.1.0"

#     def __init__(self, config: AutoConfig):
#         super().__init__()
#         self.num_heads = config.num_heads
#         self.head_dim = config.num_dims // config.num_heads
#         self.k = 16  # max KNN vectors to lookup
#         self.max_memories = 8192  # max k/v vectors to store
#         self.epsilon = 1e-8  # for numerical stability
#         # Gating parameter: one gate per head
#         self.gate = nn.Parameter(torch.zeros(self.num_heads))
#         # Initialize key_memories and value_memories for each head
#         multiplier = 2 if config.differential else 1
#         # Create FAISS index for each head
#         self.indexes = [
#             faiss.IndexFlatIP(self.head_dim * multiplier)  # Inner product similarity
#             for _ in range(self.num_heads)
#         ]
#         self.register_buffer(
#             "value_memories", torch.empty(self.num_heads, 0, self.head_dim)
#         )
#         # LSH configuration
#         self.approximation = False
#         if self.approximation:
#             self.num_hyperplanes = 16  # number of random hyperplanes for hashing
#             self.num_hash_tables = 8  # number of hash tables for multiple probing
#             # Register hyperplanes for each hash table
#             self.register_buffer(
#                 "hyperplanes",
#                 torch.randn(self.num_hash_tables, self.num_hyperplanes, self.head_dim),
#             )

#     def forward(
#         self, inputs: Tensor, query: Tensor, key: Tensor, value: Tensor, outputs: Tensor
#     ) -> Tensor:
#         batch_size, seq_len, _ = inputs.size()

#         # Prepare queries, keys, and values for memory: [num_heads, Q, dim]
#         multiplier = query.size(-1) // self.head_dim
#         q = (
#             query.view(batch_size, self.num_heads, seq_len, self.head_dim * multiplier)
#             .transpose(0, 1)
#             .reshape(self.num_heads, batch_size * seq_len, self.head_dim * multiplier)
#         )  # [num_heads, Q, d_k]
#         k = (
#             key.view(batch_size, self.num_heads, seq_len, self.head_dim * multiplier)
#             .transpose(0, 1)
#             .reshape(self.num_heads, batch_size * seq_len, self.head_dim * multiplier)
#         )  # [num_heads, Q, d_k]
#         v = (
#             value.view(batch_size, self.num_heads, seq_len, self.head_dim)
#             .transpose(0, 1)
#             .reshape(self.num_heads, batch_size * seq_len, self.head_dim)
#         )  # [num_heads, Q, d_k]

#         # Detach q, k, v for non-differentiable memory operations
#         q_detached = q.detach()
#         k_detached = k.detach()
#         v_detached = v.detach()

#         # Look up KNN memories without tracking gradients
#         with torch.no_grad():
#             scores_mem, indices_mem = self._find_knn(q_detached)

#         # If no memory found, use standard attention output
#         combined_output = outputs  # [batch_size, num_heads, seq_len, dim]
#         if scores_mem is not None:
#             # Retrieve memory values without tracking gradients
#             with torch.no_grad():
#                 memory_values = self._get_values(indices_mem)  # [num_heads, Q, k, dim]

#             # Compute weighted sum: [num_heads, Q, dim]
#             # Note: scores_mem is detached; no gradients will flow through it
#             weighted_memory = memory_values * scores_mem.unsqueeze(
#                 -1
#             )  # [num_heads, Q, k, dim]
#             weighted_memory = weighted_memory.sum(dim=2)  # [num_heads, Q, dim]

#             # Reshape to [num_heads, batch_size, seq_len, dim]
#             weighted_memory = weighted_memory.view(
#                 self.num_heads, batch_size, seq_len, self.head_dim
#             )  # [num_heads, batch_size, seq_len, dim]

#             # Permute to [batch_size, num_heads, seq_len, dim] to align with attn_out
#             weighted_memory = weighted_memory.permute(
#                 1, 0, 2, 3
#             ).contiguous()  # [batch_size, num_heads, seq_len, dim]

#             # Apply per-head gating
#             gate = (
#                 torch.sigmoid(self.gate)
#                 .view(1, self.num_heads, 1, 1)
#                 .to(outputs.device)
#             )  # [1, num_heads, 1, 1]

#             output_dim = outputs.size(-1)
#             weighted_memory = weighted_memory[..., :output_dim]

#             # Combine attention and memory outputs using the gate
#             combined_output = (
#                 gate * weighted_memory + (1 - gate) * outputs
#             )  # [batch_size, num_heads, seq_len, dim]

#         # Update memory with current keys and values without tracking gradients
#         with torch.no_grad():
#             self._update_memory(
#                 k_detached, v_detached
#             )  # Correct dimensions: [num_heads, Q, dim]

#         return combined_output

#     def _find_knn(self, queries: Tensor) -> tuple:
#         """
#         Finds k-nearest neighbors using FAISS.
#         """
#         if self.value_memories.size(1) == 0:
#             return None, None

#         # Normalize queries for cosine similarity
#         queries_norm = F.normalize(queries, p=2, dim=-1, eps=self.epsilon)

#         all_scores = []
#         all_indices = []

#         # Process each head separately
#         for head_idx in range(self.num_heads):
#             # Convert query to numpy for FAISS
#             head_queries = queries_norm[head_idx].cpu().numpy().astype("float32")

#             # Get scores and indices from FAISS
#             k = min(self.k, self.indexes[head_idx].ntotal)
#             if k == 0:
#                 continue

#             scores, indices = self.indexes[head_idx].search(head_queries, k)

#             # Convert back to torch
#             all_scores.append(torch.from_numpy(scores).to(queries.device))
#             all_indices.append(torch.from_numpy(indices).to(queries.device))

#         if not all_scores:
#             return None, None

#         # Stack results for all heads
#         scores = torch.stack(all_scores)
#         indices = torch.stack(all_indices)

#         return scores, indices

#     def _get_values(self, indices: Tensor) -> Tensor:
#         """
#         Retrieves the values corresponding to the nearest neighbors.
#         """
#         # Gather values for each head
#         gathered_values = torch.gather(
#             self.value_memories.unsqueeze(1).expand(-1, indices.size(1), -1, -1),
#             2,
#             indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
#         )  # [num_heads, Q, k, dim]
#         return gathered_values

#     def _update_memory(self, keys: Tensor, values: Tensor):
#         """
#         Updates both FAISS indexes and value memories.
#         """
#         # First concatenate new values
#         new_values = torch.cat([self.value_memories, values], dim=1)

#         # If we exceed max_memories, trim the oldest ones
#         if new_values.size(1) > self.max_memories:
#             start_idx = new_values.size(1) - self.max_memories
#             new_values = new_values[:, start_idx:, :]

#         # Update value memories in PyTorch
#         self.value_memories = new_values

#         # Update FAISS indexes
#         for head_idx in range(self.num_heads):
#             # Normalize keys for cosine similarity
#             head_keys = F.normalize(keys[head_idx], p=2, dim=-1, eps=self.epsilon)
#             head_keys = head_keys.cpu().numpy().astype("float32")

#             # Get total number of vectors after addition
#             total_vectors = self.indexes[head_idx].ntotal + head_keys.shape[0]

#             # If we'll exceed max_memories, create new index with most recent vectors
#             if total_vectors > self.max_memories:
#                 # Create new index
#                 new_index = faiss.IndexFlatIP(self.indexes[head_idx].d)

#                 if self.indexes[head_idx].ntotal > 0:
#                     # Get most recent vectors from old index
#                     retain_count = self.max_memories - head_keys.shape[0]
#                     if retain_count > 0:
#                         # Reconstruct the most recent vectors
#                         old_vectors = self.indexes[head_idx].reconstruct_n(
#                             self.indexes[head_idx].ntotal - retain_count, retain_count
#                         )
#                         # Add retained old vectors
#                         new_index.add(old_vectors)

#                 # Add new vectors
#                 new_index.add(head_keys)
#                 self.indexes[head_idx] = new_index
#             else:
#                 # Just add new vectors
#                 self.indexes[head_idx].add(head_keys)

#     def _hash_vectors(self, vectors: Tensor, table_idx: int) -> Tensor:
#         """
#         Convert vectors to hash codes using the specified hash table's hyperplanes.
#         """
#         # Get binary codes based on hyperplane positions
#         # vectors: [num_heads, Q, dim]
#         # hyperplanes: [num_hyperplanes, dim]
#         hash_bits = (
#             vectors @ self.hyperplanes[table_idx].T
#         ) >= 0  # [num_heads, Q, num_hyperplanes]

#         # Convert to decimal for bucketing
#         powers = 2 ** torch.arange(
#             self.num_hyperplanes, device=vectors.device, dtype=torch.long
#         )  # [num_hyperplanes]

#         return (hash_bits * powers).sum(dim=-1)  # [num_heads, Q]
