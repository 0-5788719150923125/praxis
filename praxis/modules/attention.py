import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules import MultiIdentity


class PraxisAttention(nn.Module):
    """
    We implement Differential Attention, to filter the noise from attention maps:
    https://arxiv.org/abs/2410.05258

    We implement ALiBi for length extrapolation, to keep parameter counts low:
    https://arxiv.org/abs/2108.12409
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.causal = config.causal
        self.differential = config.differential
        self.max_seq_len = config.context_length
        self.hidden_size = config.num_dims
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # self.memory = False
        # self.em = (
        #     EpisodicMemory(config, self.num_heads, self.head_dim)
        #     if self.memory
        #     else MultiIdentity()
        # )

        # Query and key projections for differential heads
        multiplier = 2 if self.differential else 1
        self.query = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * multiplier,
            bias=False,
        )
        self.key = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim * multiplier,
            bias=False,
        )
        self.value = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )

        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

        # Lambda vectors per differential head and per head
        if self.differential:
            self.lambda_init = 0.8  # A good default, per the paper
            self.lambdas = nn.ParameterDict(
                dict(
                    q1=nn.Parameter(torch.randn(self.head_dim)),
                    q2=nn.Parameter(torch.randn(self.head_dim)),
                    k1=nn.Parameter(torch.randn(self.head_dim)),
                    k2=nn.Parameter(torch.randn(self.head_dim)),
                )
            )
            self.norm = nn.GroupNorm(
                num_groups=self.num_heads, num_channels=self.num_heads * self.head_dim
            )

        self.output = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Pre-compute the ALiBi slopes
        slopes = 2 ** (-8 * torch.arange(1, self.num_heads + 1) / self.num_heads)
        self.register_buffer("slopes", slopes)
        self.register_buffer(
            "positions", torch.arange(self.max_seq_len, dtype=torch.float32)
        )

    def forward(
        self, inputs: Tensor, attention_mask: Tensor, token_indices: Optional[Tensor]
    ):
        batch_size, seq_len, _ = inputs.shape

        # Compute queries, keys, and values
        multiplier = 2 if self.differential else 1
        q = (
            self.query(inputs)
            .view(batch_size, seq_len, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(inputs)
            .view(batch_size, seq_len, self.num_heads, multiplier * self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(inputs)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        reciprocal = 1.0 / math.sqrt(self.head_dim)
        if self.differential:
            # Split queries and keys
            Q1, Q2 = q[..., : self.head_dim], q[..., self.head_dim :]
            K1, K2 = k[..., : self.head_dim], k[..., self.head_dim :]

            # Compute differntial attention scores
            scores = [
                torch.matmul(Q1, K1.transpose(-2, -1)) * reciprocal,
                torch.matmul(Q2, K2.transpose(-2, -1)) * reciprocal,
            ]
        else:
            # Compute attention scores
            scores = [torch.matmul(q, k.transpose(-2, -1)) * reciprocal]

        # Compute ALiBi biases
        if torch.is_tensor(token_indices):
            positions = self.positions[token_indices]
        else:
            positions = (
                self.positions[:seq_len].unsqueeze(0).expand(batch_size, seq_len)
            )

        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        biases = self.slopes.view(1, self.num_heads, 1, 1) * pos_diff.unsqueeze(1)
        scores = [s - biases for s in scores]

        # Apply masks
        if self.causal:
            causal_mask = (
                torch.triu(
                    torch.full((seq_len, seq_len), -1e9, device=inputs.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = [s + causal_mask for s in scores]

        attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        scores = [s + attention_mask for s in scores]

        # Compute attention weights
        weights = [self.dropout(F.softmax(s, dim=-1)) for s in scores]

        # Compute attention weights
        diff_weights = weights[0]
        if self.differential:
            # Compute scalar lambda
            lambda_scalar = (
                torch.exp(torch.dot(self.lambdas["q1"], self.lambdas["k1"]))
                - torch.exp(torch.dot(self.lambdas["q2"], self.lambdas["k2"]))
                + self.lambda_init
            )
            diff_weights = weights[0] - lambda_scalar * weights[1]

        # Compute attention output
        attention_scores = torch.matmul(
            diff_weights, v
        )  # Shape: (batch_size, num_heads, seq_len, head_dim)

        if self.differential:
            # Reshape for GroupNorm
            attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
            # Shape: (batch_size, seq_len, num_heads, head_dim)

            attention_scores = attention_scores.view(
                batch_size, seq_len, self.num_heads * self.head_dim
            )
            # Shape: (batch_size, seq_len, num_heads * head_dim)

            # Permute to (batch_size, num_channels, seq_len)
            attention_scores = attention_scores.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, num_heads * head_dim, seq_len)

            # Apply GroupNorm
            attention_scores = self.norm(attention_scores)

            # Permute back to (batch_size, seq_len, num_heads * head_dim)
            attention_scores = attention_scores.permute(0, 2, 1).contiguous()
            # Shape: (batch_size, seq_len, num_heads * head_dim)

            # Apply scaling factor
            attention_scores = attention_scores * (1 - self.lambda_init)
        else:
            attention_scores = attention_scores.transpose(1, 2).reshape(
                batch_size, seq_len, self.hidden_size
            )

        # Output projection
        return self.output(attention_scores)


class EpisodicMemory(nn.Module):
    """
    Implements Human-like Episodic Memory for Infinite Context LLMs:
    https://arxiv.org/abs/2407.09450
    """

    def __init__(self, config: PraxisConfig, num_heads: int, head_dim: int):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.memory_size = 1000  # Maximum number of events to store
        self.event_keys: List[Tensor] = (
            []
        )  # List to store representative keys of events
        self.event_values: List[Tensor] = (
            []
        )  # List to store event values (key-value pairs)
        self.event_positions: List[Tensor] = (
            []
        )  # Positions of events for temporal contiguity
        self.k = 10  # Number of events to retrieve

        # Parameters for surprise-based segmentation
        self.surprise_gamma = 1.0  # Scaling factor for threshold
        self.surprise_window = 100  # Window size for mean and std

        # Placeholder for past surprises to compute dynamic threshold
        self.past_surprises: List[float] = []

    def forward(
        self,
        inputs: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor,
        token_indices: Optional[Tensor],
        seq_len_q: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], int, Tensor]:
        """
        Args:
            inputs: [batch_size, seq_len_q, hidden_size]
            q: [batch_size, num_heads, seq_len_q, head_dim * multiplier]
            k: [batch_size, num_heads, seq_len_q, head_dim * multiplier]
            v: [batch_size, num_heads, seq_len_q, head_dim]
            attention_mask: [batch_size, seq_len_q]
            token_indices: Optional[Tensor], [batch_size, seq_len_q]
            seq_len_q: Original query sequence length

        Returns:
            Augmented inputs, q, k, v, attention_mask, token_indices, seq_len_k, positions_k
        """
        batch_size = inputs.size(0)

        # Compute positions_q
        if token_indices is not None:
            # Assuming token_indices are provided and correspond to absolute positions
            # Fetch positions from self.positions (registered buffer)
            positions_q = self.positions[token_indices]  # [batch_size, seq_len_q]
        else:
            # Generate sequential positions
            positions_q = (
                torch.arange(seq_len_q, device=inputs.device)
                .unsqueeze(0)
                .expand(batch_size, seq_len_q)
            )  # [batch_size, seq_len_q]

        # Initialize lists to collect augmented data
        augmented_q_list = []
        augmented_k_list = []
        augmented_v_list = []
        augmented_mask_list = []
        retrieved_positions_list = []

        # Iterate over each sample in the batch
        for b in range(batch_size):
            # Extract per-sample data
            q_b = q[b]  # [num_heads, seq_len_q, head_dim * multiplier]
            k_b = k[b]
            v_b = v[b]
            positions_q_b = positions_q[b]  # [seq_len_q]

            # Compute query_key as the concatenated key vector from all heads for the last token
            # Shape: [num_heads * head_dim * multiplier]
            query_key = k_b[:, -1, :].reshape(-1)

            # Update episodic memory with current keys and values
            # Combine heads and sequence length: [seq_len_q, num_heads * head_dim * multiplier]
            k_b_combined = k_b.permute(1, 0, 2).reshape(
                seq_len_q, -1
            )  # [seq_len_q, num_heads * head_dim * multiplier]
            v_b_combined = v_b.permute(1, 0, 2).reshape(
                seq_len_q, -1
            )  # [seq_len_q, num_heads * head_dim]

            self.update(k_b_combined, v_b_combined, positions_q_b)

            # Retrieve relevant events
            retrieved_keys, retrieved_values, retrieved_positions = (
                self.retrieve_events(query_key, positions_q_b[-1])
            )

            # Ensure a fixed number of retrieved events (pad if necessary)
            while len(retrieved_keys) < self.k:
                retrieved_keys.append(torch.zeros_like(self.event_keys[0]))
                retrieved_values.append(torch.zeros_like(self.event_values[0]))
                retrieved_positions.append(0.0)  # Assign a default position

            # Trim retrieved events to exactly self.k
            retrieved_keys = retrieved_keys[: self.k]
            retrieved_values = retrieved_values[: self.k]
            retrieved_positions = retrieved_positions[: self.k]

            # Collect retrieved_positions
            retrieved_positions_tensor = torch.tensor(
                retrieved_positions, device=inputs.device, dtype=torch.float32
            )
            retrieved_positions_list.append(retrieved_positions_tensor)  # [k]

            # Concatenate retrieved keys and current keys
            retrieved_k = torch.stack(
                retrieved_keys, dim=0
            )  # [k, num_heads * head_dim * multiplier]
            retrieved_v = torch.stack(
                retrieved_values, dim=0
            )  # [k, num_heads * head_dim]

            # Reshape to [num_heads, k, head_dim * multiplier]
            retrieved_k = retrieved_k.view(
                -1,
                self.num_heads,
                self.head_dim * (2 if self.config.differential else 1),
            )
            retrieved_v = retrieved_v.view(-1, self.num_heads, self.head_dim)

            # Permute to [num_heads, k, head_dim * multiplier]
            # Already in desired shape after view

            # Concatenate retrieved events with current keys and values along sequence dimension
            k_b_aug = torch.cat(
                [retrieved_k, k_b], dim=1
            )  # [num_heads, k + seq_len_q, head_dim * multiplier]
            v_b_aug = torch.cat(
                [retrieved_v, v_b], dim=1
            )  # [num_heads, k + seq_len_q, head_dim]

            # Update attention mask by prepending ones for retrieved events
            num_retrieved = retrieved_k.size(1)
            attn_mask_b_aug = F.pad(
                attention_mask[b], (num_retrieved, 0), value=1
            )  # [k + seq_len_q]

            # Append augmented tensors to lists
            augmented_q_list.append(
                q_b
            )  # [num_heads, seq_len_q, head_dim * multiplier]
            augmented_k_list.append(
                k_b_aug
            )  # [num_heads, k + seq_len_q, head_dim * multiplier]
            augmented_v_list.append(v_b_aug)  # [num_heads, k + seq_len_q, head_dim]
            augmented_mask_list.append(attn_mask_b_aug)  # [k + seq_len_q]

        # Stack augmented tensors across the batch
        q = torch.stack(
            augmented_q_list, dim=0
        )  # [batch_size, num_heads, seq_len_q, head_dim * multiplier]
        k = torch.stack(
            augmented_k_list, dim=0
        )  # [batch_size, num_heads, k + seq_len_q, head_dim * multiplier]
        v = torch.stack(
            augmented_v_list, dim=0
        )  # [batch_size, num_heads, k + seq_len_q, head_dim]
        attention_mask = torch.stack(
            augmented_mask_list, dim=0
        )  # [batch_size, k + seq_len_q]

        # Define updated sequence lengths
        seq_len_k = k.size(2)  # k + seq_len_q

        # Concatenate positions_q with retrieved_positions to form positions_k
        if token_indices is not None:
            # positions_q: [batch_size, seq_len_q]
            # retrieved_positions_list: List of [k] tensors
            retrieved_positions_tensor = torch.stack(
                retrieved_positions_list, dim=0
            )  # [batch_size, k]
            positions_k = torch.cat(
                [positions_q, retrieved_positions_tensor], dim=1
            )  # [batch_size, k + seq_len_q]
        else:
            # positions_q: [batch_size, seq_len_q]
            # retrieved_positions_list: List of [k] tensors
            retrieved_positions_tensor = torch.stack(
                retrieved_positions_list, dim=0
            )  # [batch_size, k]
            positions_k = torch.cat(
                [positions_q, retrieved_positions_tensor], dim=1
            )  # [batch_size, k + seq_len_q]

        return inputs, q, k, v, attention_mask, token_indices, seq_len_k, positions_k

    def compute_surprise(self, keys: Tensor) -> Tensor:
        """
        Computes surprise based on the norm of the difference between consecutive keys.

        Args:
            keys: [seq_len, hidden_size]

        Returns:
            surprise: [seq_len]
        """
        # Compute difference between consecutive keys
        diff = keys[1:] - keys[:-1]  # [seq_len - 1, hidden_size]
        # Compute L2 norm as surprise
        surprise = torch.norm(diff, dim=-1)  # [seq_len - 1]
        # Prepend zero for the first token
        surprise = torch.cat(
            [torch.zeros(1, device=keys.device), surprise], dim=0
        )  # [seq_len]
        return surprise

    def segment_events(self, surprises: Tensor) -> List[int]:
        """
        Segments events based on surprise thresholds.

        Args:
            surprises: [seq_len]

        Returns:
            boundaries: List of indices where events are segmented
        """
        # Dynamic threshold based on past surprises
        if len(self.past_surprises) >= self.surprise_window:
            window_surprises = torch.tensor(
                self.past_surprises[-self.surprise_window :], device=surprises.device
            )
        else:
            window_surprises = torch.tensor(
                self.past_surprises, device=surprises.device
            )

        if len(window_surprises) > 0:
            mean_surprise = window_surprises.mean()
            std_surprise = window_surprises.std()
        else:
            mean_surprise = surprises.mean()
            std_surprise = surprises.std()

        threshold = mean_surprise + self.surprise_gamma * std_surprise

        # Identify event boundaries where surprise exceeds threshold
        boundaries = (surprises > threshold).nonzero(as_tuple=True)[0].tolist()
        return boundaries

    def store_events(
        self, keys: Tensor, values: Tensor, positions: Tensor, boundaries: List[int]
    ):
        """
        Stores segmented events into memory.

        Args:
            keys: [seq_len, hidden_size]
            values: [seq_len, hidden_size]
            positions: [seq_len]
            boundaries: List of boundary indices
        """
        start = 0
        for boundary in boundaries:
            end = boundary + 1
            event_key = keys[start:end]  # [event_len, hidden_size]
            event_value = values[start:end]  # [event_len, hidden_size]
            event_position = positions[start:end]  # [event_len]

            # Store representative key as mean of keys in the event
            representative_key = event_key.mean(dim=0)  # [hidden_size]
            self.event_keys.append(representative_key)
            self.event_values.append(event_value)
            self.event_positions.append(event_position)
            start = end

            # Maintain memory size
            if len(self.event_keys) > self.memory_size:
                self.event_keys.pop(0)
                self.event_values.pop(0)
                self.event_positions.pop(0)

    def retrieve_events(
        self, query_key: Tensor, current_position: float
    ) -> Tuple[List[Tensor], List[Tensor], List[float]]:
        """
        Retrieves relevant events based on cosine similarity and temporal contiguity.

        Args:
            query_key: [hidden_size]
            current_position: float

        Returns:
            retrieved_keys: List of [hidden_size]
            retrieved_values: List of [event_len, hidden_size]
            retrieved_positions: List of positions (floats)
        """
        if len(self.event_keys) == 0:
            return [], [], []

        # Compute cosine similarities
        similarities = torch.stack(
            [torch.cosine_similarity(query_key, ek, dim=0) for ek in self.event_keys]
        )  # [num_events]

        # Get top-k similar events
        topk_similarities, topk_indices = torch.topk(
            similarities, k=min(self.k, similarities.size(0)), largest=True
        )

        # Retrieve corresponding events
        retrieved_keys = []
        retrieved_values = []
        retrieved_positions = []
        for idx in topk_indices:
            idx = idx.item()
            ev_key = self.event_keys[idx]
            ev_value = self.event_values[idx]
            ev_position = (
                self.event_positions[idx].to(torch.float32).mean().item()
            )  # Average position of the event

            retrieved_keys.append(ev_key)
            retrieved_values.append(ev_value)
            retrieved_positions.append(ev_position)

        # Add temporally contiguous events (previous and next)
        top_idx = topk_indices[0].item()
        if top_idx > 0 and len(retrieved_keys) < self.k:
            # Add previous event
            ev_key_prev = self.event_keys[top_idx - 1]
            ev_value_prev = self.event_values[top_idx - 1]
            ev_position_prev = self.event_positions[top_idx - 1].mean().item()

            retrieved_keys.append(ev_key_prev)
            retrieved_values.append(ev_value_prev)
            retrieved_positions.append(ev_position_prev)

        if top_idx < len(self.event_keys) - 1 and len(retrieved_keys) < self.k:
            # Add next event
            ev_key_next = self.event_keys[top_idx + 1]
            ev_value_next = self.event_values[top_idx + 1]
            ev_position_next = self.event_positions[top_idx + 1].mean().item()

            retrieved_keys.append(ev_key_next)
            retrieved_values.append(ev_value_next)
            retrieved_positions.append(ev_position_next)

        # Ensure exactly k events are retrieved by padding if necessary
        while len(retrieved_keys) < self.k:
            retrieved_keys.append(torch.zeros_like(self.event_keys[0]))
            retrieved_values.append(torch.zeros_like(self.event_values[0]))
            retrieved_positions.append(0.0)  # Default position

        # Trim to exactly k events
        retrieved_keys = retrieved_keys[: self.k]
        retrieved_values = retrieved_values[: self.k]
        retrieved_positions = retrieved_positions[: self.k]

        return retrieved_keys, retrieved_values, retrieved_positions

    def update(self, keys: Tensor, values: Tensor, positions: Tensor):
        """
        Updates the episodic memory with new keys, values, and positions.

        Args:
            keys: [seq_len_q, hidden_size]
            values: [seq_len_q, hidden_size]
            positions: [seq_len_q]
        """
        # Compute surprise
        surprise = self.compute_surprise(keys)  # [seq_len_q]
        self.past_surprises.extend(surprise.tolist())

        # Segment events
        boundaries = self.segment_events(surprise)  # List of boundary indices

        # Store events
        self.store_events(keys, values, positions, boundaries)
