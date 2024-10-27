import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_modularity(
    similarity_matrix: torch.Tensor, boundaries: torch.Tensor
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
    similarity_matrix: torch.Tensor, boundaries: torch.Tensor
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
        total_conductance += between_edges / (2 * within_edges + between_edges + 1e-10)

    return total_conductance / len(unique_communities)


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1, **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        # Project and reshape
        q = (
            self.q_proj(query)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # Ensure mask matches the scores dimensions
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.embed_dim
        )
        return self.out_proj(attn_output)


class EpisodicMemory(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        surprise_threshold: float = 0.5,
        max_memory_size: int = 1000,
        similarity_buffer_size: int = 32,  # Reduced as per paper
        contiguity_buffer_size: int = 16,  # Reduced as per paper
        max_seq_length: int = 512,
        window_size: int = 100,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.surprise_threshold = surprise_threshold
        self.max_memory_size = max_memory_size
        self.similarity_buffer_size = similarity_buffer_size
        self.contiguity_buffer_size = contiguity_buffer_size
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        self.gamma = gamma

        # Memory stores
        self.memory_keys = []
        self.memory_values = []
        self.memory_lengths = []
        self.memory_timestamps = []

        # Networks
        self.surprise_network = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.similarity_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

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
            current_modularity = compute_modularity(sim_matrix, boundaries[b])

            for pos in boundary_positions:
                temp_boundaries = boundaries[b].clone()
                temp_boundaries[pos] = 0
                new_modularity = compute_modularity(sim_matrix, temp_boundaries)

                # Keep boundary removal if it improves modularity
                if new_modularity > current_modularity:
                    refined_boundaries[b, pos] = 0
                    current_modularity = new_modularity

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

    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Compute surprise scores with improved normalization."""
        normed_x = F.layer_norm(x, x.shape[-1:])
        return self.surprise_network(normed_x)

    def identify_event_boundaries(
        self, token_representations: torch.Tensor
    ) -> torch.Tensor:
        """Identify event boundaries using dynamic thresholding."""
        surprise_scores = self.compute_surprise(token_representations)
        dynamic_threshold = self.compute_surprise_threshold(surprise_scores)
        boundaries = (surprise_scores > dynamic_threshold).float()

        # Refine boundaries using graph-theoretic metrics
        refined_boundaries = self.refine_boundaries(boundaries, token_representations)

        return refined_boundaries

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        # Project and reshape
        q = (
            self.q_proj(query)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .reshape(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # Ensure mask matches the scores dimensions
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, -1, self.embed_dim
        )
        return self.out_proj(attn_output)

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
        # Minimum event size check
        if event_tokens.size(1) < 2:
            return

        if len(self.memory_keys) >= self.max_memory_size:
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
            self.memory_lengths.pop(0)
            self.memory_timestamps.pop(0)

        # Normalize event representation
        event_key = F.layer_norm(event_tokens.mean(dim=1), event_tokens.shape[-1:])
        padded_event, actual_length = self.pad_sequence(event_tokens)

        self.memory_keys.append(event_key)
        self.memory_values.append(padded_event)
        self.memory_lengths.append(actual_length)
        self.memory_timestamps.append(timestamp)

    def retrieve_memories(
        self, query: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve memories with improved similarity computation and normalization."""
        if not self.memory_keys:
            return None, None

        # Convert lists to tensors
        memory_keys = torch.stack(self.memory_keys)
        memory_values = torch.stack(self.memory_values)

        # Normalize query and compute similarity
        query_mean = F.layer_norm(query.mean(dim=1, keepdim=True), query.shape[-1:])
        query_proj = self.similarity_proj(query_mean)
        keys_proj = self.similarity_proj(memory_keys)

        # Compute similarity scores with improved normalization
        similarity_scores = torch.zeros(memory_keys.size(0), device=query.device)
        for b in range(query.size(0)):
            similarity_scores += F.cosine_similarity(
                F.normalize(query_proj[b], p=2, dim=-1),
                F.normalize(keys_proj[:, b], p=2, dim=-1),
                dim=-1,
            )
        similarity_scores = similarity_scores / query.size(0)

        # Get top-k similar memories
        k = min(self.similarity_buffer_size, len(self.memory_keys))
        top_k_sim, top_k_indices = torch.topk(similarity_scores, k)

        # Process similarity buffer
        selected_memories = memory_values[top_k_indices]
        similarity_buffer = selected_memories.transpose(0, 1).reshape(
            query.size(0), -1, self.embed_dim
        )

        # Process temporal contiguity
        # Convert timestamps to float tensor and compute distances
        timestamps = torch.tensor(
            self.memory_timestamps, device=query.device, dtype=torch.float32
        )
        temporal_dists = torch.abs(timestamps.unsqueeze(0) - timestamps.unsqueeze(1))

        # Normalize temporal distances
        if temporal_dists.numel() > 0:  # Check if tensor is non-empty
            temporal_dists = F.normalize(temporal_dists.float(), p=2, dim=-1)

        k_temporal = min(self.contiguity_buffer_size, len(self.memory_timestamps))
        _, temporal_indices = torch.topk(
            -temporal_dists[top_k_indices], k_temporal, dim=1
        )
        temporal_indices = torch.unique(temporal_indices.reshape(-1))

        temporal_memories = memory_values[temporal_indices]
        contiguity_buffer = temporal_memories.transpose(0, 1).reshape(
            query.size(0), -1, self.embed_dim
        )

        return (
            F.layer_norm(similarity_buffer, similarity_buffer.shape[-1:]),
            F.layer_norm(contiguity_buffer, contiguity_buffer.shape[-1:]),
        )

    def compute_surprise(self, x: torch.Tensor) -> torch.Tensor:
        return self.surprise_network(x)

    def identify_event_boundaries(
        self, token_representations: torch.Tensor
    ) -> torch.Tensor:
        surprise_scores = self.compute_surprise(token_representations)
        return (surprise_scores > self.surprise_threshold).float()


class EMLLMLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        # Components
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.episodic_memory = EpisodicMemory(
            embed_dim=embed_dim,
            max_seq_length=max_seq_length,
            similarity_buffer_size=32,  # As per paper
            contiguity_buffer_size=16,  # As per paper
        )

        # Normalization and residual components
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Memory integration
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, timestamp: int = 0):
        """Forward pass with improved memory integration."""
        residual = x
        x = self.layer_norm1(x)

        # Event segmentation
        boundaries = self.episodic_memory.identify_event_boundaries(x)

        # Store events
        batch_size, seq_length, _ = x.shape
        event_start = 0
        for i in range(seq_length):
            if boundaries[:, i].any() or i == seq_length - 1:
                if i > event_start:
                    event = x[:, event_start : i + 1, :]
                    self.episodic_memory.store_event(event, timestamp)
                event_start = i + 1

        # Memory retrieval and integration
        sim_buffer, cont_buffer = self.episodic_memory.retrieve_memories(x)

        if sim_buffer is not None and cont_buffer is not None:
            sim_buffer = self.memory_proj(sim_buffer)
            cont_buffer = self.memory_proj(cont_buffer)

            # Combine buffers with adaptive weighting
            context = torch.cat([x, sim_buffer, cont_buffer], dim=1)

            attn_mask = self.create_attention_mask(
                batch_size=batch_size,
                num_heads=self.attention.num_heads,
                query_len=seq_length,
                key_len=context.size(1),
                device=x.device,
            )
        else:
            context = x
            attn_mask = self.create_attention_mask(
                batch_size=batch_size,
                num_heads=self.attention.num_heads,
                query_len=seq_length,
                key_len=seq_length,
                device=x.device,
            )

        # Apply attention with improved normalization
        attn_output = self.attention(
            self.layer_norm2(x),
            self.layer_norm2(context),
            self.layer_norm2(context),
            attn_mask,
        )

        x = residual + self.dropout(attn_output)
        return x

    def create_attention_mask(
        self,
        batch_size: int,
        num_heads: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask for the integrated sequence"""
        # Create mask for main sequence (causal)
        main_mask = torch.triu(
            torch.ones(query_len, query_len, device=device), diagonal=1
        ).bool()
        main_mask = ~main_mask

        # If we have memory context, create the full mask
        if key_len > query_len:
            # Memory context mask (allow attention to all memory)
            memory_mask = torch.ones(
                query_len, key_len - query_len, device=device
            ).bool()
            # Combine masks
            mask = torch.cat([main_mask, memory_mask], dim=1)
        else:
            mask = main_mask

        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, num_heads, query_len, key_len)

        return mask


def test_em_llm_integrated():
    print("Initializing integrated test...")

    # Test parameters
    batch_size = 2
    seq_length = 10
    embed_dim = 256
    num_heads = 8

    # Create random input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Initialize model with proper parameters
    model = EMLLMLayer(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, max_seq_length=512
    )

    print("\nTesting forward pass...")

    # First forward pass
    output1 = model(x, timestamp=1)
    print(f"First forward pass output shape: {output1.shape}")

    # Store some events and do another forward pass
    print("\nTesting with memory...")

    # Create new input
    x2 = torch.randn(batch_size, seq_length, embed_dim)

    # Second forward pass
    output2 = model(x2, timestamp=2)
    print(f"Second forward pass output shape: {output2.shape}")

    # Test attention mask
    print("\nTesting attention mask...")
    query_len = 5
    key_len = 15  # query_len + memory_len
    test_batch_size = 2
    test_num_heads = 8
    mask = model.create_attention_mask(
        batch_size=test_batch_size,
        num_heads=test_num_heads,
        query_len=query_len,
        key_len=key_len,
        device=torch.device("cpu"),
    )
    print(f"Attention mask shape: {mask.shape}")
    print(f"Attention mask example (first head):\n{mask[0, 0]}")

    # Print memory statistics
    print("\nMemory statistics:")
    print(f"Number of stored events: {len(model.episodic_memory.memory_values)}")
    print(f"Stored sequence lengths: {model.episodic_memory.memory_lengths}")

    # Test if outputs are different (they should be due to memory)
    print("\nChecking if memory affects outputs:")
    output_diff = (output1 - output2).abs().mean().item()
    print(f"Average difference between outputs: {output_diff:.6f}")


def test_em_llm():
    print("Initializing test...")
    # Test parameters
    batch_size = 2
    seq_length = 10
    embed_dim = 256

    # Create random input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Initialize memory module
    memory = EpisodicMemory(embed_dim)

    print("\nTesting event storage...")
    # Test storing events of different lengths
    event1 = torch.randn(batch_size, 3, embed_dim)
    event2 = torch.randn(batch_size, 5, embed_dim)

    memory.store_event(event1, timestamp=1)
    memory.store_event(event2, timestamp=2)

    print(f"Number of stored events: {len(memory.memory_values)}")
    print(f"Stored sequence lengths: {memory.memory_lengths}")

    print("\nTesting memory retrieval...")
    sim_buffer, cont_buffer = memory.retrieve_memories(x)

    print("\nMemory retrieval results:")
    if sim_buffer is not None:
        print(f"Similarity buffer shape: {sim_buffer.shape}")
    if cont_buffer is not None:
        print(f"Contiguity buffer shape: {cont_buffer.shape}")

    print("\nTesting event boundary detection...")
    boundaries = memory.identify_event_boundaries(x)
    print(f"Boundary tensor shape: {boundaries.shape}")
    print(f"Number of events detected: {boundaries.sum().item()}")


if __name__ == "__main__":
    test_em_llm()
    test_em_llm_integrated()
