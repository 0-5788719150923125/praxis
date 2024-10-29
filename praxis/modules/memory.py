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
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.max_seq_length = config.context_length
        self.vocab_size = config.vocab_size
        self.surprise_threshold = 0.5
        self.max_memory_size = 1000
        self.similarity_buffer_size = 32
        self.contiguity_buffer_size = 16
        self.window_size = 100
        self.gamma = 2.0

        # Normalization and residual components
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)

        # Memory integration
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # Initialize memory stores
        self.memory_keys = []
        self.memory_values = []
        self.memory_lengths = []
        self.memory_timestamps = []
        self.current_timestamp = 0

        # Networks for surprise computation and similarity projection
        self.surprise_network = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.similarity_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

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

        residual = query
        x = self.layer_norm1(query)

        # Generate logits for the next token predictions
        logits = self.lm_head(x)

        # Compute surprise scores
        if target_tokens is not None:
            # Training: Use target_tokens to compute negative log-likelihood
            surprise_scores = self.compute_surprise_with_targets(logits, target_tokens)
        else:
            # Inference: Use entropy-based surprise
            surprise_scores = self.compute_surprise(logits)

        # Identify event boundaries using surprise_scores
        boundaries = self.identify_event_boundaries(surprise_scores)

        # Store events based on boundaries
        self.store_events(x, boundaries, timestamp)

        # Memory retrieval and integration
        sim_buffer, cont_buffer = self.retrieve_memories(x)

        if sim_buffer is not None and cont_buffer is not None:
            sim_buffer = self.memory_proj(sim_buffer)
            cont_buffer = self.memory_proj(cont_buffer)

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
        else:
            # No changes needed
            pass

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

    def refine_boundaries(
        self, boundaries: torch.Tensor, representations: torch.Tensor
    ) -> torch.Tensor:
        """Refine event boundaries using graph-theoretic metrics."""
        batch_size = representations.size(0)
        refined_boundaries = boundaries.clone()

        for b in range(batch_size):
            # Compute similarity matrix for this batch
            sim_matrix = torch.matmul(
                representations[b], representations[b].transpose(-2, -1)
            )

            # Normalize similarity matrix
            sim_matrix = F.normalize(sim_matrix, p=2, dim=-1)

            # Get initial boundary positions
            boundary_positions = boundaries[b].nonzero().squeeze(-1)

            # Try removing each boundary and keep if it improves modularity
            current_modularity = self.compute_modularity(sim_matrix, boundaries[b])
            current_conductance = self.compute_conductance(sim_matrix, boundaries[b])

            for pos in boundary_positions:
                temp_boundaries = boundaries[b].clone()
                temp_boundaries[pos] = 0
                new_modularity = self.compute_modularity(sim_matrix, temp_boundaries)
                new_conductance = self.compute_conductance(sim_matrix, temp_boundaries)

                # Keep boundary removal if it improves modularity or conductance
                if (
                    new_modularity > current_modularity
                    or new_conductance < current_conductance
                ):
                    refined_boundaries[b, pos] = 0
                    current_modularity = new_modularity
                    current_conductance = new_conductance

            # Ensure minimum event size
            event_sizes = torch.diff(
                torch.cat(
                    [torch.tensor([0]), refined_boundaries[b].nonzero().squeeze(-1)]
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
        total_edge_weight = similarity_matrix.sum()

        # Create community assignments based on boundaries
        communities = torch.cumsum(boundaries, dim=0)

        modularity = 0.0
        for i in range(N):
            for j in range(N):
                if communities[i] == communities[j]:
                    # Actual edge weight minus expected edge weight
                    modularity += (
                        similarity_matrix[i, j]
                        - similarity_matrix[i].sum()
                        * similarity_matrix[j].sum()
                        / total_edge_weight
                    )

        return modularity / total_edge_weight

    def compute_conductance(
        self, similarity_matrix: torch.Tensor, boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute conductance for given boundaries in similarity matrix."""
        N = similarity_matrix.size(0)
        communities = torch.cumsum(boundaries, dim=0)
        unique_communities = torch.unique(communities)

        total_conductance = 0.0
        for comm in unique_communities:
            comm_mask = communities == comm
            within_edges = similarity_matrix[comm_mask][:, comm_mask].sum()
            between_edges = similarity_matrix[comm_mask][:, ~comm_mask].sum()
            total_conductance += between_edges / (
                2 * within_edges + between_edges + 1e-10
            )

        return total_conductance / len(unique_communities)

    def identify_event_boundaries(
        self, surprise_scores: torch.Tensor, window_size: int = 10, gamma: float = 1.0
    ) -> torch.Tensor:
        """
        Identify event boundaries using dynamic thresholding.
        surprise_scores: (batch_size, seq_length)
        """
        batch_size, seq_length = surprise_scores.shape
        boundaries = torch.zeros_like(surprise_scores)

        for b in range(batch_size):
            for t in range(seq_length):
                start = max(0, t - window_size)
                end = t if t > 0 else 1  # Ensure at least one element
                window = surprise_scores[b, start:end]
                mu = window.mean()
                sigma = (
                    window.std(unbiased=False)
                    if window.size(0) > 1
                    else torch.tensor(0.0)
                )
                T = mu + gamma * sigma
                boundaries[b, t] = (surprise_scores[b, t] > T).float()

        return boundaries

    def pad_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, int]:
        batch_size, seq_len, embed_dim = sequence.shape
        actual_length = seq_len
        if seq_len < self.max_seq_length:
            padding = torch.zeros(
                batch_size,
                self.max_seq_length - seq_len,
                embed_dim,
                device=sequence.device,
            )
            sequence = torch.cat([sequence, padding], dim=1)
        elif seq_len > self.max_seq_length:
            sequence = sequence[:, : self.max_seq_length, :]
            actual_length = self.max_seq_length
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

            # Maintain memory size limit
            if len(self.memory_keys) >= self.max_memory_size:
                self.memory_keys.pop(0)
                self.memory_values.pop(0)
                self.memory_lengths.pop(0)
                self.memory_timestamps.pop(0)

            # Select representative token
            representative_token = single_event_tokens[0, :]  # Shape: [embed_dim]

            # Normalize event representation
            event_key = F.layer_norm(representative_token, representative_token.shape)
            padded_event, actual_length = self.pad_sequence(
                single_event_tokens.unsqueeze(0)
            )

            self.memory_keys.append(event_key)  # Shape: [embed_dim]
            self.memory_values.append(
                padded_event
            )  # Shape: [1, max_seq_length, embed_dim]
            self.memory_lengths.append(actual_length)
            self.memory_timestamps.append(timestamp)

    def retrieve_memories(
        self, query: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.memory_keys:
            return None, None

        # Convert lists to tensors
        memory_keys = torch.stack(self.memory_keys)  # Shape: [num_memories, embed_dim]
        memory_values = torch.cat(
            self.memory_values, dim=0
        )  # Shape: [num_memories, max_seq_length, embed_dim]

        # Normalize query and compute similarity
        query_mean = F.layer_norm(
            query.mean(dim=1), query.shape[-1:]
        )  # Shape: [batch_size, embed_dim]
        query_proj = self.similarity_proj(query_mean)  # Shape: [batch_size, embed_dim]
        keys_proj = self.similarity_proj(
            memory_keys
        )  # Shape: [num_memories, embed_dim]

        # Normalize projections
        query_proj_norm = F.normalize(
            query_proj, p=2, dim=-1
        )  # Shape: [batch_size, embed_dim]
        keys_proj_norm = F.normalize(
            keys_proj, p=2, dim=-1
        )  # Shape: [num_memories, embed_dim]

        # Compute cosine similarities
        similarity_scores = torch.matmul(
            query_proj_norm, keys_proj_norm.T
        )  # Shape: [batch_size, num_memories]

        # Average over the batch to get a single similarity score per memory
        similarity_scores = similarity_scores.mean(dim=0)  # Shape: [num_memories]

        # Get top-k similar memories
        k = min(self.similarity_buffer_size, len(self.memory_keys))
        top_k_sim, top_k_indices = torch.topk(similarity_scores, k)

        # Process similarity buffer
        selected_memories = memory_values[
            top_k_indices
        ]  # Shape: [k, max_seq_length, embed_dim]
        # Expand to batch dimension
        similarity_buffer = selected_memories.unsqueeze(0).expand(
            query.size(0), -1, -1, -1
        )
        similarity_buffer = similarity_buffer.reshape(
            query.size(0), -1, self.hidden_dim
        )
        # Now similarity_buffer has shape: [batch_size, k * max_seq_length, embed_dim]

        # Process temporal contiguity
        # Convert timestamps to float tensor and compute distances
        timestamps = torch.tensor(
            self.memory_timestamps, device=query.device, dtype=torch.float32
        )
        current_timestamp = (
            timestamps[-1]
            if len(timestamps) > 0
            else torch.tensor(0.0, device=query.device)
        )
        temporal_similarity = -torch.abs(
            timestamps - current_timestamp
        )  # Negative distance
        k_temporal = min(self.contiguity_buffer_size, len(self.memory_timestamps))
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

        return (
            F.layer_norm(similarity_buffer, similarity_buffer.shape[-1:]),
            F.layer_norm(contiguity_buffer, contiguity_buffer.shape[-1:]),
        )

    def extend_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = attention_mask.size(0)
        memory_len = key_len - seq_len

        # Initialize extended_attention_mask with zeros
        extended_attention_mask = torch.zeros(
            (batch_size, seq_len, key_len), device=device
        )

        # Copy the original attention mask into positions corresponding to the original sequence
        extended_attention_mask[:, :, memory_len:] = attention_mask

        # Create causal mask
        query_indices = torch.arange(seq_len, device=device).unsqueeze(
            1
        )  # [seq_len, 1]
        key_indices = torch.arange(key_len, device=device).unsqueeze(0)  # [1, key_len]

        # Allow attention to memory tokens and to previous tokens in the sequence
        causal_mask = (key_indices <= (memory_len + query_indices)).float()

        # Convert causal mask to additive mask
        causal_mask = (
            1.0 - causal_mask
        ) * -1e9  # zeros where allowed, -1e9 where masked

        # Expand causal_mask to batch size
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine extended_attention_mask and causal_mask
        extended_attention_mask = extended_attention_mask + causal_mask

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
