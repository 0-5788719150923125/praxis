import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class PraxisMemory(nn.Module):
    """
    This module implements human-like Episodic Memory, which allows for nearly-
    infinite contexts lengths:
    https://arxiv.org/abs/2407.09450
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.dropout = config.dropout
        self.surprise_threshold = 0.5
        self.max_memory_length = 32
        self.max_num_memories = 512
        self.max_retrieval_size = 128  # Set an appropriate limit
        self.similarity_buffer_size = 3  # number of similar memories retrieved
        self.contiguity_buffer_size = 2  # number of temporally-close memories retrieved
        self.window_size = (
            32  # used in event boundary detection for computing rolling statistics
        )
        self.gamma = 2.0  # changes event segmentation granularity

        self.compressed = False
        if self.compressed:
            self.memory_dim = self.hidden_dim // 8
            self.compress = nn.Linear(self.hidden_dim, self.memory_dim)
            self.decompress = nn.Linear(self.memory_dim, self.hidden_dim)
        else:
            self.memory_dim = self.hidden_dim

        # Memory integration
        self.storage = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.ReLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )

        # Language modeling head
        self.clustering = nn.Linear(self.memory_dim, config.vocab_size, bias=False)

        # Initialize memory stores
        self.current_timestamp = 0

        # Networks for surprise computation and similarity projection
        self.surprise = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim // 2),
            nn.ReLU(),
            nn.Linear(self.memory_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.similarity = nn.Linear(self.memory_dim, self.memory_dim)

        self.register_buffer(
            "memory_keys", torch.zeros(self.max_num_memories, self.memory_dim)
        )
        self.register_buffer(
            "memory_values",
            torch.zeros(self.max_num_memories, self.max_memory_length, self.memory_dim),
        )
        self.register_buffer(
            "memory_timestamps",
            torch.zeros(self.max_num_memories, dtype=torch.float32),
        )
        self.memory_index = 0  # Pointer to current memory position

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        target_tokens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate timestamp dynamically
        timestamp = self.current_timestamp
        self.current_timestamp += 1

        if self.compressed:
            # Project to lower dimension for memory operations
            query = self.compress(query)
            key = self.compress(key)
            value = self.compress(value)

        # Generate logits for the next token predictions
        logits = self.clustering(query)

        # Compute surprise scores
        if target_tokens is not None:
            # Training: Use target_tokens to compute negative log-likelihood
            surprise_scores = self.compute_surprise_with_targets(logits, target_tokens)
        else:
            # Inference: Use entropy-based surprise
            surprise_scores = self.compute_surprise(logits)

        # Identify event boundaries using surprise_scores
        boundaries = self.identify_event_boundaries(surprise_scores)

        refined_boundaries = self.refine_boundaries(boundaries, query)

        # Store events based on boundaries
        self.store_events(query, refined_boundaries, timestamp)

        # Memory retrieval and integration
        sim_buffer, cont_buffer = self.retrieve_memories(query)

        if sim_buffer is not None and cont_buffer is not None:
            sim_buffer = self.storage(sim_buffer)
            cont_buffer = self.storage(cont_buffer)

            # Combine buffers with adaptive weighting
            key_context = torch.cat([sim_buffer, cont_buffer, key], dim=1)
            value_context = torch.cat([sim_buffer, cont_buffer, value], dim=1)

            # Extend attention mask
            attention_mask = self.extend_attention_mask(
                attention_mask, query.size(1), key_context.size(1), query.device
            )

            # Update key and value
            key = key_context
            value = value_context

        if self.compressed:
            query = self.decompress(query)
            key = self.decompress(key)
            value = self.decompress(value)

        # Return modified query, key, value, attention_mask
        return query, key, value, attention_mask

    def store_events(self, token_representations, boundaries, timestamp):
        batch_size, seq_length, embed_dim = token_representations.shape

        # Append a boundary at the end to capture the last event
        boundaries = torch.cat(
            [boundaries, torch.ones(batch_size, 1, device=boundaries.device)], dim=1
        )  # Shape: [batch_size, seq_length + 1]

        # Compute event labels
        event_labels = torch.cumsum(boundaries, dim=1)[
            :, :-1
        ].long()  # Shape: [batch_size, seq_length]

        # Flatten batch and sequence dimensions
        flat_tokens = token_representations.reshape(
            -1, embed_dim
        )  # Shape: [batch_size * seq_length, embed_dim]
        flat_event_labels = event_labels.reshape(-1)  # Shape: [batch_size * seq_length]
        flat_batch_indices = (
            torch.arange(batch_size, device=token_representations.device)
            .unsqueeze(1)
            .expand(-1, seq_length)
            .reshape(-1)
        )  # Shape: [batch_size * seq_length]

        # Combine batch indices and event labels to create unique event IDs
        max_event_label = event_labels.max() + 1
        unique_event_ids = (
            flat_batch_indices * max_event_label + flat_event_labels
        )  # Shape: [batch_size * seq_length]

        # Get unique event IDs and inverse indices
        unique_event_ids, inverse_indices = unique_event_ids.unique(return_inverse=True)
        num_events = unique_event_ids.size(0)

        # Sort inverse_indices to group tokens by event
        sorted_indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[sorted_indices]

        # Count tokens per event
        event_counts = torch.bincount(
            inverse_indices, minlength=num_events
        )  # Shape: [num_events]

        # Compute positions within events
        positions_in_events = torch.zeros_like(inverse_indices)
        positions_in_events[sorted_indices] = torch.arange(
            inverse_indices.size(0), device=inverse_indices.device
        ) - torch.cumsum(
            torch.cat(
                [torch.tensor([0], device=event_counts.device), event_counts[:-1]]
            ),
            dim=0,
        ).repeat_interleave(
            event_counts
        )

        # Prepare events tensor
        max_event_length = event_counts.max().item()
        events = torch.zeros(
            num_events, max_event_length, embed_dim, device=token_representations.device
        )  # Shape: [num_events, max_event_length, embed_dim]

        # Scatter tokens into events tensor
        events[inverse_indices, positions_in_events] = flat_tokens

        # Get event lengths
        event_lengths = event_counts

        # Pad or truncate events to self.max_memory_length
        if max_event_length < self.max_memory_length:
            padding = torch.zeros(
                num_events,
                self.max_memory_length - max_event_length,
                embed_dim,
                device=events.device,
            )
            events = torch.cat([events, padding], dim=1)
        elif max_event_length > self.max_memory_length:
            events = events[:, : self.max_memory_length, :]
            event_lengths = torch.clamp(event_lengths, max=self.max_memory_length)

        # Store events
        self.store_event(events, event_lengths, timestamp)

    def store_event(self, events, event_lengths, timestamp):
        num_events = events.size(0)
        indices = (
            torch.arange(num_events, device=events.device) + self.memory_index
        ) % self.max_num_memories

        representative_tokens = events[:, 0, :]

        # Update memory buffers
        self.memory_keys[indices] = representative_tokens
        self.memory_values[indices] = events.detach()
        self.memory_timestamps[indices] = timestamp

        self.memory_index = (self.memory_index + num_events) % self.max_num_memories

    def compute_conductance(
        self, similarity_matrix: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute average conductance over communities."""
        N = similarity_matrix.size(0)
        degrees = similarity_matrix.sum(dim=1)  # Node degrees (shape: [N])
        total_volume = degrees.sum()

        # Create community assignments
        communities = torch.cumsum(boundaries, dim=0)  # Shape: [N]
        unique_communities = torch.unique(communities)
        num_communities = unique_communities.size(0)

        # Initialize tensors to store volumes and cuts
        volumes = torch.zeros(num_communities, device=similarity_matrix.device)
        cuts = torch.zeros(num_communities, device=similarity_matrix.device)

        # Create a mapping from community labels to indices
        comm2idx = {comm.item(): idx for idx, comm in enumerate(unique_communities)}

        # Build a one-hot encoding of communities
        community_one_hot = F.one_hot(
            communities.long(), num_classes=num_communities
        ).float()  # Shape: [N, num_communities]

        # Compute volumes for each community
        volumes = community_one_hot.t().matmul(degrees)  # Shape: [num_communities]

        # Compute cuts between communities
        # First, compute the adjacency between communities
        inter_community = (
            community_one_hot.t().matmul(similarity_matrix).matmul(community_one_hot)
        )
        # Set diagonal to zero to exclude intra-community edges
        inter_community.fill_diagonal_(0)
        # Sum over rows to get cuts for each community
        cuts = inter_community.sum(dim=1)  # Shape: [num_communities]

        # Compute conductance for each community
        min_volumes = torch.min(volumes, total_volume - volumes)
        conductance = cuts / (min_volumes + 1e-10)  # Shape: [num_communities]

        # Compute average conductance
        average_conductance = conductance.mean()

        return average_conductance

    def compute_conductance_batch(self, similarity_matrices, boundaries):
        """
        similarity_matrices: [batch_size, N, N]
        boundaries: [batch_size, N]
        """
        batch_size = similarity_matrices.size(0)
        N = similarity_matrices.size(1)

        degrees = similarity_matrices.sum(dim=2)  # [batch_size, N]
        total_volumes = degrees.sum(dim=1)  # [batch_size]

        communities = torch.cumsum(boundaries, dim=1).long()  # [batch_size, N]
        num_communities = communities.max(dim=1)[0] + 1  # [batch_size]

        max_num_communities = num_communities.max().item()

        # One-hot encoding of communities
        # Shape: [batch_size, N, max_num_communities]
        community_one_hot = F.one_hot(
            communities, num_classes=max_num_communities
        ).float()

        # Compute volumes: [batch_size, max_num_communities]
        volumes = torch.bmm(
            community_one_hot.transpose(1, 2), degrees.unsqueeze(2)
        ).squeeze(2)

        # Compute inter-community edge weights: [batch_size, max_num_communities, max_num_communities]
        inter_community = torch.bmm(
            torch.bmm(community_one_hot.transpose(1, 2), similarity_matrices),
            community_one_hot,
        )

        # Zero out the diagonal (intra-community edges)
        inter_community = inter_community - torch.diag_embed(
            torch.diagonal(inter_community, dim1=1, dim2=2)
        )

        # Compute cuts: [batch_size, max_num_communities]
        cuts = inter_community.sum(dim=2)

        # Compute min_volumes: [batch_size, max_num_communities]
        total_volumes = total_volumes.unsqueeze(1)  # [batch_size, 1]
        min_volumes = torch.min(volumes, total_volumes - volumes)

        # Compute conductance for each community
        cond = cuts / (min_volumes + 1e-10)  # [batch_size, max_num_communities]

        # Create a mask for valid communities
        # Shape: [batch_size, max_num_communities]
        valid_communities_mask = (
            torch.arange(max_num_communities, device=communities.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        ) < num_communities.unsqueeze(1)

        # Mask invalid communities
        cond = cond * valid_communities_mask.float()

        # Sum conductance over valid communities and compute the average
        cond_sum = cond.sum(dim=1)  # [batch_size]
        num_valid_communities = num_communities.float()  # [batch_size]
        conductance = cond_sum / num_valid_communities

        return conductance  # [batch_size]

    def compute_modularity_batch(self, similarity_matrices, boundaries):
        batch_size, N, _ = similarity_matrices.shape

        degrees = similarity_matrices.sum(dim=2)  # [batch_size, N]
        total_edge_weights = degrees.sum(dim=1, keepdim=True)  # [batch_size, 1]

        communities = torch.cumsum(boundaries, dim=1).long()  # [batch_size, N]

        degrees_expanded = degrees.unsqueeze(2)  # [batch_size, N, 1]
        degrees_transposed = degrees.unsqueeze(1)  # [batch_size, 1, N]

        expected_weights = torch.bmm(degrees_expanded, degrees_transposed) / (
            total_edge_weights.unsqueeze(1) + 1e-10
        )
        modularity_matrices = similarity_matrices - expected_weights

        # Create community masks
        community_masks = (communities.unsqueeze(2) == communities.unsqueeze(1)).float()

        Q = (modularity_matrices * community_masks).sum(dim=(1, 2)) / (
            total_edge_weights.squeeze(1) + 1e-10
        )

        return Q  # [batch_size]

    def refine_boundaries(self, boundaries, representations):
        batch_size, seq_length, _ = representations.shape
        refined_boundaries = boundaries.clone()

        # Compute similarity matrices
        sim_matrices = torch.matmul(
            representations, representations.transpose(1, 2)
        )  # Shape: [batch_size, seq_length, seq_length]

        # Pre-compute initial modularity and conductance
        current_modularity = self.compute_modularity_batch(
            sim_matrices, boundaries
        )  # Shape: [batch_size]
        current_conductance = self.compute_conductance_batch(
            sim_matrices, boundaries
        )  # Shape: [batch_size]

        # Find boundary positions
        boundary_positions = (boundaries == 1).nonzero(
            as_tuple=False
        )  # Shape: [num_boundaries, 2]

        if boundary_positions.size(0) == 0:
            # No boundaries to refine
            return refined_boundaries

        num_boundaries = boundary_positions.size(0)

        # Extract batch indices and positions of boundaries
        batch_indices = boundary_positions[:, 0]  # Shape: [num_boundaries]
        positions = boundary_positions[:, 1]  # Shape: [num_boundaries]

        # For each boundary, create a copy of the boundaries and remove the boundary at the position
        temp_boundaries_expanded = boundaries[
            batch_indices
        ].clone()  # Shape: [num_boundaries, seq_length]
        temp_boundaries_expanded[torch.arange(num_boundaries), positions] = (
            0  # Remove the boundary
        )

        # Extract corresponding similarity matrices
        sim_matrices_expanded = sim_matrices[
            batch_indices
        ]  # Shape: [num_boundaries, seq_length, seq_length]

        # Compute new modularity and conductance
        new_modularity = self.compute_modularity_batch(
            sim_matrices_expanded, temp_boundaries_expanded
        )  # Shape: [num_boundaries]
        new_conductance = self.compute_conductance_batch(
            sim_matrices_expanded, temp_boundaries_expanded
        )  # Shape: [num_boundaries]

        # Get current modularity and conductance for these batches
        current_modularity_expanded = current_modularity[
            batch_indices
        ]  # Shape: [num_boundaries]
        current_conductance_expanded = current_conductance[
            batch_indices
        ]  # Shape: [num_boundaries]

        # Compare and find improvements
        improvements = (new_modularity > current_modularity_expanded) | (
            new_conductance < current_conductance_expanded
        )  # Shape: [num_boundaries]

        # Update refined boundaries where improvements are found
        refined_boundaries[batch_indices[improvements], positions[improvements]] = 0

        # Enforce minimum event size
        refined_boundaries = self.enforce_minimum_event_size(refined_boundaries)

        return refined_boundaries

    def enforce_minimum_event_size(self, boundaries):
        batch_size, seq_length = boundaries.shape
        min_size = 3

        # Compute event labels
        event_labels = torch.cumsum(
            boundaries, dim=1
        ).long()  # [batch_size, seq_length]

        # Compute number of events per batch
        num_events_per_batch = event_labels.max(dim=1)[0] + 1  # [batch_size]
        max_num_events = num_events_per_batch.max().item()

        # Compute event sizes
        event_sizes = torch.zeros(batch_size, max_num_events, device=boundaries.device)
        event_sizes.scatter_add_(
            1, event_labels, torch.ones_like(event_labels, dtype=event_sizes.dtype)
        )

        # Identify small events
        small_events_mask = event_sizes < min_size  # [batch_size, max_num_events]
        small_events_indices = small_events_mask.nonzero(
            as_tuple=False
        )  # [num_small_events, 2], each row is [batch_index, event_index]

        # Compute boundary positions per event
        # boundaries == 1 at positions where event_labels increase
        event_labels_padded = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, device=boundaries.device, dtype=event_labels.dtype
                ),
                event_labels,
            ],
            dim=1,
        )  # [batch_size, seq_length + 1]
        diff_event_labels = torch.diff(
            event_labels_padded, dim=1
        )  # [batch_size, seq_length]
        boundary_positions = (diff_event_labels == 1).nonzero(
            as_tuple=False
        )  # [num_boundaries, 2]

        # Map event indices to boundary positions
        boundary_positions_per_event = torch.full(
            (batch_size, max_num_events), -1, device=boundaries.device, dtype=torch.long
        )
        boundary_positions_per_event[
            boundary_positions[:, 0],
            event_labels[boundary_positions[:, 0], boundary_positions[:, 1]],
        ] = boundary_positions[:, 1]

        # Remove boundaries before small events
        small_events_boundary_positions = boundary_positions_per_event[
            small_events_indices[:, 0], small_events_indices[:, 1]
        ]
        valid_mask = small_events_boundary_positions >= 0
        batch_indices_to_update = small_events_indices[valid_mask][:, 0]
        positions_to_update = small_events_boundary_positions[valid_mask]

        # Set boundaries at these positions to 0
        boundaries[batch_indices_to_update, positions_to_update] = 0

        return boundaries

    def compute_surprise(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def compute_surprise_with_targets(self, logits, target_tokens):
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        surprise = -target_log_probs
        return surprise

    def identify_event_boundaries(self, surprise_scores):
        T = self.compute_surprise_threshold(surprise_scores)
        boundaries = (surprise_scores > T).float()
        return boundaries

    def compute_surprise_threshold(self, surprise_values):
        batch_size, seq_length = surprise_values.shape
        # Calculate rolling statistics
        pad_size = self.window_size - 1
        padded_surprise = F.pad(surprise_values, (pad_size, 0), mode="replicate")
        windows = padded_surprise.unfold(1, self.window_size, 1)
        mu = windows.mean(dim=2)
        sigma = windows.std(dim=2, unbiased=False)
        threshold = mu + self.gamma * sigma
        return threshold

    def retrieve_memories(
        self, query: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.memory_index == 0:
            return None, None

        # Limit the number of memories retrieved
        num_memories = min(
            self.memory_index, self.max_num_memories, self.max_retrieval_size
        )
        memory_keys = self.memory_keys[
            :num_memories
        ]  # Shape: [num_memories, embed_dim]
        memory_values = self.memory_values[
            :num_memories
        ]  # Shape: [num_memories, max_seq_length, embed_dim]
        memory_timestamps = self.memory_timestamps[:num_memories]

        # Compute similarity
        query_mean = query.mean(dim=1)
        query_proj = self.similarity(query_mean)
        keys_proj = self.similarity(memory_keys)

        # Compute cosine similarities
        similarity_scores = torch.matmul(
            query_proj, keys_proj.T
        )  # [batch_size, num_memories]

        # Get top-k similar memories
        k = min(self.similarity_buffer_size, num_memories)
        top_k_sim, top_k_indices = torch.topk(similarity_scores, k, dim=1)

        # Gather the top_k_indices for each batch
        selected_memories = []
        for i in range(query.size(0)):
            indices = top_k_indices[i]
            selected_memories.append(memory_values[indices])
        selected_memories = torch.stack(
            selected_memories
        )  # Shape: [batch_size, k, max_seq_length, embed_dim]

        # Reshape and return
        similarity_buffer = selected_memories.view(query.size(0), -1, self.memory_dim)

        # Process temporal contiguity
        timestamps = memory_timestamps  # Shape: [num_memories]
        current_timestamp = timestamps[-1]

        temporal_similarity = -torch.abs(
            timestamps - current_timestamp
        )  # Negative distance

        k_temporal = min(self.contiguity_buffer_size, num_memories)
        top_k_temporal_sim, temporal_indices = torch.topk(
            temporal_similarity, k_temporal
        )

        temporal_memories = memory_values[
            temporal_indices
        ]  # Shape: [k_temporal, max_seq_length, embed_dim]
        # Expand to batch dimension
        contiguity_buffer = temporal_memories.unsqueeze(0).expand(
            query.size(0), -1, -1, -1
        )
        contiguity_buffer = contiguity_buffer.reshape(
            query.size(0), -1, self.memory_dim
        )
        # contiguity_buffer has shape: [batch_size, k_temporal * max_seq_length, embed_dim]

        return similarity_buffer, contiguity_buffer

    def extend_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = attention_mask.size(0)
        memory_len = key_len - seq_len

        # Create a mask for the memory tokens (allowing full attention)
        memory_mask = torch.zeros((batch_size, seq_len, memory_len), device=device)

        # Adjust attention_mask to 3D if it's 2D
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, :].expand(-1, seq_len, -1)

        # Concatenate memory mask and attention_mask
        extended_attention_mask = torch.cat([memory_mask, attention_mask], dim=2)

        # Create causal mask for the combined sequence
        causal_mask = torch.tril(torch.ones((seq_len, key_len), device=device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine masks
        combined_mask = extended_attention_mask * causal_mask

        # Convert to additive mask
        extended_attention_mask = (1.0 - combined_mask) * -1e9

        return extended_attention_mask


def test_praxis_memory():
    from attention import ModularAttention

    class SimpleBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.memory = PraxisMemory(config)
            self.attn_norm = nn.LayerNorm(config.hidden_size)
            self.attn = ModularAttention(config)

        def forward(
            self,
            inputs: torch.Tensor,
            attention_mask: torch.Tensor,
            target_tokens: torch.Tensor = None,
        ):

            residual = inputs
            normalized = self.attn_norm(inputs)
            query, key, value = normalized, normalized, normalized
            if self.memory:
                query, key, value, attention_mask = self.memory(
                    query, key, value, attention_mask, target_tokens
                )

            outputs = self.attn(query, key, value, attention_mask)
            return outputs + residual

    def create_attention_mask(batch_size, seq_len, device):
        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))

        # Convert to additive mask
        attention_mask = (1.0 - causal_mask) * -1e9

        # Expand to batch size
        attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)

        return attention_mask

    print("Initializing test...")
    # Test parameters
    batch_size = 2
    seq_length = 10
    embed_dim = 256
    num_heads = 8
    dropout = 0.1

    # Create random input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Create attention mask
    attention_mask = create_attention_mask(batch_size, seq_length, x.device)

    # Initialize model with proper parameters
    config = PretrainedConfig(
        hidden_size=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        max_length=512,
        causal=True,
        differential=False,
        vocab_size=8192,
    )
    model = SimpleBlock(config)

    print("\nTesting forward pass...")

    # Simulate training
    target_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    outputs = model(x, attention_mask, target_tokens)
    print(f"First forward pass output shape: {outputs.shape}")

    # Simulate inference
    outputs2 = model(x, attention_mask)
    print(f"Second forward pass output shape: {outputs2.shape}")

    # Test if outputs are different (they should be due to memory)
    print("\nChecking if memory affects outputs:")
    output_diff = (outputs - outputs2).abs().mean().item()
    print(f"Average difference between outputs: {output_diff:.6f}")


if __name__ == "__main__":
    test_praxis_memory()
