import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class PraxisMemory(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # Components
        self.hidden_dim = config.num_dims
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.surprise_threshold = 0.5
        self.max_memory_length = 16
        self.num_total_memories = 256
        self.similarity_buffer_size = 8  # number of similar memories retrieved
        self.contiguity_buffer_size = 4  # number of temporally-close memories retrieved
        self.window_size = (
            20  # used in event boundary detection for computing rolling statistics
        )
        self.gamma = 2.0  # changes event segmentation granularity

        # Memory integration
        self.storage = nn.Sequential(
            # nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Language modeling head
        self.brain = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # Initialize memory stores
        self.current_timestamp = 0

        # Networks for surprise computation and similarity projection
        self.surprise = nn.Sequential(
            # nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.similarity = nn.Sequential(
            # nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.register_buffer(
            "memory_keys", torch.zeros(self.num_total_memories, self.hidden_dim)
        )
        self.register_buffer(
            "memory_values",
            torch.zeros(
                self.num_total_memories, self.max_memory_length, self.hidden_dim
            ),
        )
        self.register_buffer(
            "memory_lengths", torch.zeros(self.num_total_memories, dtype=torch.long)
        )
        self.register_buffer(
            "memory_timestamps",
            torch.zeros(self.num_total_memories, dtype=torch.float32),
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

        # Generate logits for the next token predictions
        logits = self.brain(query)

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
            context = torch.cat([sim_buffer, cont_buffer, key], dim=1)

            # Extend attention mask
            extended_attention_mask = self.extend_attention_mask(
                attention_mask, query.size(1), context.size(1), query.device
            )

            # Update key and value
            key = context
            value = context
            attention_mask = extended_attention_mask

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

        # Compute indices for storing events in the memory buffers
        indices = (
            torch.arange(num_events, device=events.device) + self.memory_index
        ) % self.num_total_memories

        # Select representative tokens (e.g., the first token of each event)
        representative_tokens = events[:, 0, :]

        # Update memory buffers
        self.memory_keys[indices] = representative_tokens.detach()
        self.memory_values[indices] = events.detach()
        self.memory_lengths[indices] = event_lengths
        self.memory_timestamps[indices] = timestamp
        self.memory_index = (self.memory_index + num_events) % self.num_total_memories

    def compute_surprise_threshold(self, surprise_values: torch.Tensor) -> torch.Tensor:
        """Compute dynamic surprise threshold based on local statistics."""
        # Ensure enough context for window
        if surprise_values.size(1) < self.window_size:
            return self.surprise_threshold * torch.ones_like(surprise_values)

        # Calculate rolling statistics
        windows = surprise_values.unfold(1, self.window_size, 1)
        mu = windows.mean(dim=-1, keepdim=True)
        sigma = windows.std(dim=-1, keepdim=True)

        # Compute threshold
        threshold = mu + self.gamma * sigma

        # Pad beginning where we don't have enough context
        pad_size = self.window_size - 1
        if pad_size > 0:
            padding = threshold[:, :1].expand(-1, pad_size, -1)
            threshold = torch.cat([padding, threshold], dim=1)

        return threshold

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

        conductance_list = []
        for i in range(batch_size):
            num_com = num_communities[i].item()
            com = communities[i]
            degree = degrees[i]
            sim_matrix = similarity_matrices[i]
            total_volume = total_volumes[i]

            # One-hot encoding
            community_one_hot = F.one_hot(
                com, num_classes=num_com
            ).float()  # [N, num_com]

            # Volumes
            volumes = community_one_hot.t().matmul(degree)  # [num_com]

            # Cuts
            inter_community = (
                community_one_hot.t().matmul(sim_matrix).matmul(community_one_hot)
            )
            inter_community.fill_diagonal_(0)
            cuts = inter_community.sum(dim=1)  # [num_com]

            # Conductance
            min_volumes = torch.min(volumes, total_volume - volumes)
            cond = cuts / (min_volumes + 1e-10)
            conductance_list.append(cond.mean())

        conductance = torch.stack(conductance_list)  # [batch_size]

        return conductance

    def refine_boundaries(self, boundaries, representations):
        batch_size, seq_length, _ = representations.shape
        refined_boundaries = boundaries.clone()

        # Compute similarity matrices
        sim_matrices = torch.bmm(representations, representations.transpose(1, 2))

        # Pre-compute initial modularity and conductance
        current_modularity = self.compute_modularity_batch(sim_matrices, boundaries)
        current_conductance = self.compute_conductance_batch(sim_matrices, boundaries)

        # Find boundary positions
        boundary_positions = (boundaries == 1).nonzero(
            as_tuple=False
        )  # Shape: [num_boundaries, 2]

        if boundary_positions.size(0) == 0:
            # No boundaries to refine
            return refined_boundaries

        # Prepare new boundary configurations
        temp_boundaries = boundaries.unsqueeze(0).repeat(
            boundary_positions.size(0), 1, 1
        )
        temp_boundaries[
            torch.arange(boundary_positions.size(0)),
            boundary_positions[:, 0],
            boundary_positions[:, 1],
        ] = 0

        # Reshape tensors for batch processing
        sim_matrices_expanded = sim_matrices[
            boundary_positions[:, 0]
        ]  # [num_boundaries, seq_length, seq_length]
        temp_boundaries_expanded = temp_boundaries[
            torch.arange(boundary_positions.size(0)), boundary_positions[:, 0], :
        ]  # [num_boundaries, seq_length]

        # Compute new modularity and conductance
        new_modularity = self.compute_modularity_batch(
            sim_matrices_expanded, temp_boundaries_expanded
        )
        new_conductance = self.compute_conductance_batch(
            sim_matrices_expanded, temp_boundaries_expanded
        )

        # Compare and update boundaries
        improvements = (
            new_modularity > current_modularity[boundary_positions[:, 0]]
        ) | (new_conductance < current_conductance[boundary_positions[:, 0]])

        # Update refined boundaries in a vectorized way
        refined_boundaries[
            boundary_positions[improvements][:, 0],
            boundary_positions[improvements][:, 1],
        ] = 0

        # Enforce minimum event size
        refined_boundaries = self.enforce_minimum_event_size(refined_boundaries)

        return refined_boundaries

    def enforce_minimum_event_size(self, boundaries):
        batch_size, seq_length = boundaries.shape
        min_size = 3

        for b in range(batch_size):
            # Get positions of boundaries
            boundary_positions = boundaries[b].nonzero().squeeze(-1)
            # Include start and end positions
            event_starts = torch.cat(
                [torch.tensor([0], device=boundaries.device), boundary_positions + 1]
            )
            event_ends = torch.cat(
                [
                    boundary_positions,
                    torch.tensor([seq_length - 1], device=boundaries.device),
                ]
            )
            event_sizes = event_ends - event_starts + 1

            # Find events smaller than min_size
            small_events = (event_sizes < min_size).nonzero().squeeze(-1)

            # Merge small events
            for idx in small_events:
                pos = boundary_positions[idx - 1] if idx > 0 else 0
                if pos < seq_length:
                    boundaries[b, pos] = 0

        return boundaries

    def compute_surprise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute surprise as the entropy of the model's prediction distribution.
        logits: (batch_size, seq_length, vocab_size)
        """
        probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_length, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(
            probs * log_probs, dim=-1
        )  # Shape: (batch_size, seq_length)
        return entropy

    def compute_surprise_with_targets(
        self, logits: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute surprise as negative log-likelihood of the target tokens.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        surprise = -target_log_probs
        return surprise

    def compute_modularity_batch(self, similarity_matrices, boundaries):
        """
        similarity_matrices: [batch_size, N, N]
        boundaries: [batch_size, N]
        """
        batch_size = similarity_matrices.size(0)
        N = similarity_matrices.size(1)

        degrees = similarity_matrices.sum(dim=2)  # [batch_size, N]
        total_edge_weights = degrees.sum(dim=1, keepdim=True)  # [batch_size, 1]

        communities = torch.cumsum(boundaries, dim=1)  # [batch_size, N]

        degrees_expanded = degrees.unsqueeze(2)  # [batch_size, N, 1]
        degrees_transposed = degrees.unsqueeze(1)  # [batch_size, 1, N]
        expected_weights = torch.bmm(degrees_expanded, degrees_transposed) / (
            total_edge_weights.unsqueeze(1) + 1e-10
        )
        modularity_matrices = similarity_matrices - expected_weights

        community_masks = (
            communities.unsqueeze(2) == communities.unsqueeze(1)
        ).float()  # [batch_size, N, N]

        Q = (modularity_matrices * community_masks).sum(dim=(1, 2)) / (
            total_edge_weights.squeeze(1) + 1e-10
        )

        return Q  # [batch_size]

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

    def identify_event_boundaries(
        self, surprise_scores: torch.Tensor, window_size: int = 10, gamma: float = 1.0
    ) -> torch.Tensor:
        batch_size, seq_length = surprise_scores.shape
        # Pad surprise_scores to handle the window at the start
        pad_size = window_size - 1
        padded_surprise = F.pad(surprise_scores, (pad_size, 0), mode="replicate")
        # Compute rolling mean and std
        unfolded = padded_surprise.unfold(
            1, window_size, 1
        )  # Shape: (batch_size, seq_length, window_size)
        mu = unfolded.mean(dim=2)
        sigma = unfolded.std(dim=2, unbiased=False)
        T = mu + gamma * sigma
        boundaries = (surprise_scores > T).float()
        return boundaries

    def pad_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, int]:
        batch_size, seq_len, embed_dim = sequence.shape
        actual_length = seq_len
        if seq_len < self.max_memory_length:
            padding = torch.zeros(
                batch_size,
                self.max_memory_length - seq_len,
                embed_dim,
                device=sequence.device,
            )
            sequence = torch.cat([sequence, padding], dim=1)
        elif seq_len > self.max_memory_length:
            sequence = sequence[:, : self.max_memory_length, :]
            actual_length = self.max_memory_length
        return sequence, actual_length

    def retrieve_memories(
        self, query: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.memory_index == 0:
            return None, None

        # Get valid memory entries
        num_memories = min(self.memory_index, self.num_total_memories)
        memory_keys = self.memory_keys[
            :num_memories
        ]  # Shape: [num_memories, embed_dim]
        memory_values = self.memory_values[
            :num_memories
        ]  # Shape: [num_memories, max_seq_length, embed_dim]
        memory_timestamps = self.memory_timestamps[:num_memories]

        # Normalize query and compute similarity
        # query_mean = F.layer_norm(query.mean(dim=1), query.shape[-1:])
        query_mean = query.mean(dim=1)
        query_proj = self.similarity(query_mean)
        # query_proj_norm = F.normalize(query_proj, p=2, dim=-1)
        query_proj_norm = query_proj
        keys_proj = self.similarity(memory_keys)
        # keys_proj_norm = F.normalize(keys_proj, p=2, dim=-1)
        keys_proj_norm = keys_proj

        # Compute cosine similarities
        similarity_scores = torch.matmul(
            query_proj_norm, keys_proj_norm.T
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
        similarity_buffer = selected_memories.view(query.size(0), -1, self.hidden_dim)

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
            query.size(0), -1, self.hidden_dim
        )
        # contiguity_buffer has shape: [batch_size, k_temporal * max_seq_length, embed_dim]

        # return (
        #     F.layer_norm(similarity_buffer, similarity_buffer.shape[-1:]),
        #     F.layer_norm(contiguity_buffer, contiguity_buffer.shape[-1:]),
        # )
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
    from attention import PraxisAttention

    class SimpleBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.memory = PraxisMemory(config)
            self.attn_norm = nn.LayerNorm(config.num_dims)
            self.attn = PraxisAttention(config)

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
        num_dims=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        context_length=512,
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
