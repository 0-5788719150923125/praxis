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

    def store_events(
        self,
        token_representations: torch.Tensor,
        boundaries: torch.Tensor,
        timestamp: int,
    ):
        """
        Store events based on identified boundaries.

        token_representations: (batch_size, seq_length, embed_dim)
        boundaries: (batch_size, seq_length)
        """
        batch_size, seq_length, _ = token_representations.shape

        for b in range(batch_size):
            event_start = 0
            for t in range(seq_length):
                if boundaries[b, t] == 1 or t == seq_length - 1:
                    if t >= event_start:
                        event = token_representations[
                            b, event_start : t + 1, :
                        ]  # Shape: [seq_length_event, embed_dim]
                        self.store_event(
                            event.unsqueeze(0), timestamp
                        )  # Shape: [1, seq_length_event, embed_dim]
                    event_start = t + 1

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

    def compute_modularity_batch(
        self, similarity_matrices: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute modularity score for given boundaries in similarity matrix for all batches at once."""
        batch_size = similarity_matrices.size(0)
        N = similarity_matrices.size(1)

        # [batch_size, N]
        degrees = similarity_matrices.sum(dim=2)
        # [batch_size, 1]
        total_edge_weights = degrees.sum(dim=1, keepdim=True)

        # [batch_size, N]
        communities = torch.cumsum(boundaries, dim=1)

        # [batch_size, N, N]
        expected_weights = torch.bmm(
            degrees.unsqueeze(2), degrees.unsqueeze(1)
        ) / total_edge_weights.unsqueeze(1)
        modularity_matrices = similarity_matrices - expected_weights

        # [batch_size, N, N]
        community_masks = communities.unsqueeze(2) == communities.unsqueeze(1)

        # [batch_size]
        Q = (modularity_matrices * community_masks).sum(
            dim=(1, 2)
        ) / total_edge_weights.squeeze(1)

        return Q

    def compute_conductance_batch(
        self, similarity_matrices: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute conductance for all batches at once."""
        batch_size = similarity_matrices.size(0)
        N = similarity_matrices.size(1)

        # [batch_size, N]
        degrees = similarity_matrices.sum(dim=2)
        # [batch_size]
        total_volumes = degrees.sum(dim=1)

        # [batch_size, N]
        communities = torch.cumsum(boundaries, dim=1)

        # Remap community IDs to be consecutive integers starting from 0
        for b in range(batch_size):
            # Get unique values and create mapping
            unique_vals = communities[b].unique()
            mapping = torch.zeros(
                int(unique_vals.max().item()) + 1,
                dtype=torch.long,
                device=communities.device,
            )
            mapping[unique_vals.long()] = torch.arange(
                len(unique_vals), dtype=torch.long, device=communities.device
            )
            # Apply mapping
            communities[b] = mapping[communities[b].long()]

        num_communities = communities.max().long() + 1

        # [batch_size, N, num_communities]
        community_one_hot = F.one_hot(
            communities.long(), num_classes=int(num_communities)
        ).float()

        # Rest of the function remains the same
        volumes = torch.bmm(degrees.unsqueeze(1), community_one_hot).squeeze(1)

        inter_community = torch.bmm(
            torch.bmm(community_one_hot.transpose(1, 2), similarity_matrices),
            community_one_hot,
        )

        inter_community.diagonal(dim1=1, dim2=2).zero_()
        cuts = inter_community.sum(dim=2)

        other_volumes = total_volumes.unsqueeze(1) - volumes
        min_volumes = torch.min(volumes, other_volumes)
        conductance = cuts / (min_volumes + 1e-10)

        average_conductance = conductance.mean(dim=1)

        return average_conductance

    def refine_boundaries(
        self, boundaries: torch.Tensor, representations: torch.Tensor
    ) -> torch.Tensor:
        batch_size = representations.size(0)
        refined_boundaries = boundaries.clone()

        # Batch compute similarity matrices
        sim_matrices = torch.bmm(representations, representations.transpose(1, 2))
        sim_matrices = F.normalize(sim_matrices, p=2, dim=-1)

        # Pre-compute initial modularity/conductance for all batches
        current_modularity = self.compute_modularity_batch(sim_matrices, boundaries)
        current_conductance = self.compute_conductance_batch(sim_matrices, boundaries)

        # Still need some iteration for boundary refinement
        diff_tensor = torch.tensor([0], device=boundaries.device)
        for b in range(batch_size):
            boundary_positions = boundaries[b].nonzero().squeeze(-1)
            for pos in boundary_positions:
                temp_boundaries = boundaries.clone()
                temp_boundaries[b, pos] = 0

                # These would now operate on all batches at once
                new_modularity = self.compute_modularity_batch(
                    sim_matrices, temp_boundaries
                )
                new_conductance = self.compute_conductance_batch(
                    sim_matrices, temp_boundaries
                )

                if (
                    new_modularity[b] > current_modularity[b]
                    or new_conductance[b] < current_conductance[b]
                ):
                    refined_boundaries[b, pos] = 0
                    current_modularity[b] = new_modularity[b]
                    current_conductance[b] = new_conductance[b]

            # Ensure minimum event size
            event_sizes = torch.diff(
                torch.cat(
                    [
                        diff_tensor,
                        refined_boundaries[b].nonzero().squeeze(-1),
                    ]
                )
            )
            min_size = 3  # Minimum event size

            # Merge small events
            for i in range(len(event_sizes)):
                if event_sizes[i] < min_size:
                    if i > 0:  # Not first event
                        refined_boundaries[b, i] = 0

        return refined_boundaries

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

    def compute_modularity(
        self, similarity_matrix: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute modularity score for given boundaries in similarity matrix."""
        N = similarity_matrix.size(0)
        degrees = similarity_matrix.sum(dim=1)  # Node degrees (shape: [N])
        total_edge_weight = degrees.sum()  # Total edge weight (scalar)

        # Create community assignments based on boundaries
        communities = torch.cumsum(boundaries, dim=0)  # (shape: [N])

        # Expected edge weights
        expected_weights = (
            torch.outer(degrees, degrees) / total_edge_weight
        )  # Shape: [N, N]

        # Modularity matrix
        modularity_matrix = similarity_matrix - expected_weights  # Shape: [N, N]

        # Create a mask where communities[i] == communities[j]
        community_mask = communities.unsqueeze(1) == communities.unsqueeze(
            0
        )  # Shape: [N, N]

        # Compute modularity
        Q = modularity_matrix[community_mask].sum() / total_edge_weight  # Scalar

        return Q

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

    def store_event(self, event_tokens: torch.Tensor, timestamp: int):
        """Store event with improved checks and normalization."""
        # event_tokens: [batch_size, seq_length_event, embed_dim]
        batch_size = event_tokens.size(0)

        for b in range(batch_size):
            single_event_tokens = event_tokens[
                b
            ]  # Shape: [seq_length_event, embed_dim]

            # Minimum event size check
            if single_event_tokens.size(0) < 2:
                continue

            # Select representative token
            representative_token = single_event_tokens[0, :]  # Shape: [embed_dim]

            # Normalize event representation
            # event_key = F.layer_norm(representative_token, representative_token.shape)
            event_key = representative_token
            padded_event, actual_length = self.pad_sequence(
                single_event_tokens.unsqueeze(0)
            )

            idx = self.memory_index % self.num_total_memories  # Circular buffer index
            self.memory_keys[idx] = event_key.detach()
            self.memory_values[idx] = padded_event.squeeze(0).detach()
            self.memory_lengths[idx] = actual_length
            self.memory_timestamps[idx] = timestamp
            self.memory_index += 1  # Increment memory index

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
        return {similarity_buffer, contiguity_buffer}

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
