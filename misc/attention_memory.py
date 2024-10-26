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

        self.memory = True
        self.em = (
            EpisodicMemory(config, self.num_heads, self.head_dim)
            if self.memory
            else False
        )

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

        # Project Q/K/V as before
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

        # Get augmented tensors and masks from memory module
        if self.memory:
            # Add shape logging
            print("Before memory - q shape:", q.shape)
            print("Before memory - k shape:", k.shape)
            print("Before memory - v shape:", v.shape)

            (
                inputs,
                q,
                k,
                v,
                attention_mask,
                token_indices,
                seq_len,
                positions,
                alibi_bias,
                causal_mask,
            ) = self.em(
                inputs=inputs,
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                token_indices=token_indices,
                seq_len_q=seq_len,
                slopes=self.slopes,
                positions=self.positions,
                causal=self.causal,
            )

            # Add shape logging
            print("After memory - q shape:", q.shape)
            print("After memory - k shape:", k.shape)
            print("After memory - v shape:", v.shape)

        # Rest of attention computation with prepared masks
        reciprocal = 1.0 / math.sqrt(self.head_dim)
        if self.differential:
            Q1, Q2 = q[..., : self.head_dim], q[..., self.head_dim :]
            K1, K2 = k[..., : self.head_dim], k[..., self.head_dim :]
            scores = [
                torch.matmul(Q1, K1.transpose(-2, -1)) * reciprocal,
                torch.matmul(Q2, K2.transpose(-2, -1)) * reciprocal,
            ]
        else:
            scores = [torch.matmul(q, k.transpose(-2, -1)) * reciprocal]

        # Apply masks (guaranteed to have correct shapes)
        if alibi_bias is not None:
            scores = [s - alibi_bias for s in scores]

        if self.causal:
            if self.memory and causal_mask is not None:
                scores = [s + causal_mask for s in scores]
            else:
                # Create standard causal mask when not using memory
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

        # Add logging for attention computation
        print("Scores shape:", scores[0].shape)
        print("Diff weights shape:", diff_weights.shape)
        print("V shape for matmul:", v.shape)
        print("Attention scores shape:", attention_scores.shape)

        num_memory_tokens = 10
        if self.differential:
            # First handle differential dimension by taking sum - reduces to 4D
            attention_scores = (
                attention_scores[:, 0] - self.lambda_init * attention_scores[:, 1]
            )

            # Now reshape for GroupNorm as before
            attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
            # Shape: (batch_size, seq_len, num_heads, head_dim)

            attention_scores = attention_scores.view(
                batch_size, seq_len - num_memory_tokens, self.num_heads * self.head_dim
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
                batch_size, seq_len - num_memory_tokens, self.hidden_size
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
        self.multiplier = 2 if config.differential else 1

        # Memory configuration
        self.memory_size = (
            config.memory_size if hasattr(config, "memory_size") else 1000
        )
        self.k_similar = config.k_similar if hasattr(config, "k_similar") else 8
        self.k_contiguous = (
            config.k_contiguous if hasattr(config, "k_contiguous") else 2
        )

        # Surprise-based segmentation parameters
        self.surprise_threshold = (
            config.surprise_threshold if hasattr(config, "surprise_threshold") else 0.5
        )
        self.surprise_window = (
            config.surprise_window if hasattr(config, "surprise_window") else 100
        )

        # Memory storage
        self.register_buffer(
            "events", torch.zeros(0, num_heads * head_dim * self.multiplier)
        )
        self.register_buffer("positions", torch.zeros(0))
        self.register_buffer("event_lengths", torch.zeros(0, dtype=torch.long))

    def forward(
        self,
        inputs: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor,  # [batch_size, seq_len]
        token_indices: Optional[Tensor],
        seq_len_q: int,
        slopes: Tensor,  # ALiBi slopes
        positions: Tensor,  # Position encodings
        causal: bool = False,  # Whether to use causal masking
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        int,
        Tensor,
        Tensor,
        Optional[Tensor],
    ]:
        """Memory augmented attention forward pass"""
        batch_size = inputs.size(0)
        device = inputs.device
        orig_seq_len = seq_len_q

        # Process each sequence in batch
        augmented_results = [
            self._process_sequence(
                q[i],
                k[i],
                v[i],
                attention_mask[i],
                token_indices[i] if token_indices is not None else None,
                seq_len_q,
                slopes,
                positions[i] if token_indices is not None else positions[:seq_len_q],
                causal,
            )
            for i in range(batch_size)
        ]

        # Unpack and stack results
        aug_q, aug_k, aug_v, aug_mask, aug_positions, aug_alibi, aug_causal = zip(
            *augmented_results
        )

        # Stack consistently sized tensors
        stacked_k = torch.stack(aug_k)  # [batch_size, num_heads, seq_len+mem, head_dim]
        stacked_v = torch.stack(aug_v)  # [batch_size, num_heads, seq_len+mem, head_dim]
        stacked_mask = torch.stack(aug_mask)  # [batch_size, seq_len+mem]
        stacked_positions = torch.stack(aug_positions)  # [batch_size, seq_len+mem]

        # Stack masks
        stacked_alibi = torch.stack(aug_alibi)
        stacked_causal = torch.stack(aug_causal) if causal else None

        return (
            inputs,
            torch.stack(aug_q),
            stacked_k,
            stacked_v,
            stacked_mask,
            token_indices,
            stacked_k.size(2),
            stacked_positions,
            stacked_alibi,
            stacked_causal,
        )

    def _process_sequence(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,  # [seq_len]
        indices: Optional[Tensor],
        seq_len: int,
        slopes: Tensor,
        positions: Tensor,
        causal: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Process single sequence"""
        orig_seq_len = seq_len

        # Get memory events
        retrieved_k, retrieved_v, retrieved_pos = self._retrieve_events(
            k[:, -1], indices[-1] if indices is not None else seq_len - 1
        )

        # Handle padding
        target_mem_size = self.k_similar + self.k_contiguous
        actual_mem_size = retrieved_k.size(1)

        if actual_mem_size < target_mem_size:
            pad_size = target_mem_size - actual_mem_size
            retrieved_k = F.pad(retrieved_k, (0, 0, 0, pad_size))
            retrieved_v = F.pad(retrieved_v, (0, 0, 0, pad_size))
            retrieved_pos = F.pad(retrieved_pos, (0, pad_size), value=float("-inf"))

        num_mem_tokens = retrieved_k.size(1)
        total_seq_len = orig_seq_len + num_mem_tokens

        # Create augmented sequences
        aug_k = torch.cat([retrieved_k, k], dim=1)
        aug_v = torch.cat([retrieved_v, v], dim=1)

        # Create attention mask with memory tokens
        mem_mask = torch.ones(num_mem_tokens, device=mask.device)
        aug_mask = torch.cat([mem_mask, mask])

        # Handle positions
        aug_positions = torch.cat([retrieved_pos, positions])

        # Compute ALiBi biases for original+memory sequence
        pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        orig_biases = slopes.view(-1, 1, 1) * pos_diff  # [num_heads, seq_len, seq_len]

        # Create padded ALiBi tensor
        alibi = torch.zeros(
            self.num_heads,
            orig_seq_len,  # Query dimension
            total_seq_len,  # Key dimension including memory
            device=k.device,
        )

        # Memory tokens get zero bias
        alibi[..., :num_mem_tokens] = 0

        # Place original biases after memory tokens
        alibi[..., num_mem_tokens:] = orig_biases

        # Create causal mask if needed
        causal_mask = None
        if causal:
            causal_mask = torch.full(
                (1, 1, orig_seq_len, total_seq_len),  # Match original mask shape
                -1e9,
                device=k.device,
            )

            # Memory tokens always visible
            causal_mask[..., :num_mem_tokens] = 0

            # Regular causal masking for original sequence
            causal_mask[..., num_mem_tokens:].triu_(diagonal=1)

        return q, aug_k, aug_v, aug_mask, aug_positions, alibi, causal_mask

    def _compute_surprise(self, keys: Tensor) -> Tensor:
        """Compute token-wise surprise scores"""
        # Flatten heads for surprise computation
        flat_keys = keys.transpose(0, 1).reshape(keys.size(1), -1)

        # Compute differences between consecutive tokens
        diffs = flat_keys[1:] - flat_keys[:-1]
        norms = torch.norm(diffs, dim=-1)

        # Pad first position
        surprise = F.pad(norms, (1, 0), value=0.0)

        return surprise

    def _segment_events(self, surprise: Tensor) -> List[int]:
        """Find event boundaries using surprise scores"""
        # Compute dynamic threshold
        mean = surprise.mean()
        std = surprise.std()
        threshold = mean + self.surprise_threshold * std

        # Find boundary positions
        boundaries = (surprise > threshold).nonzero(as_tuple=True)[0].tolist()

        # Always include sequence end
        if len(boundaries) == 0 or boundaries[-1] != len(surprise) - 1:
            boundaries.append(len(surprise) - 1)

        return boundaries

    def _update_memory(self, keys: Tensor, values: Tensor, boundaries: List[int]):
        """Update memory with new events"""
        start = 0
        new_events = []
        new_lengths = []

        # Extract events using boundaries
        for end in boundaries:
            event_k = keys[:, start : end + 1]  # [num_heads, event_len, head_dim]
            event_v = values[:, start : end + 1]

            # Compute event representation
            event_repr = event_k.mean(dim=1)  # [num_heads, head_dim]
            new_events.append(event_repr.reshape(-1))
            new_lengths.append(end - start + 1)

            start = end + 1

        # Update memory buffers
        if new_events:
            new_events_tensor = torch.stack(new_events)
            self.events = torch.cat([self.events, new_events_tensor])[
                -self.memory_size :
            ]
            self.event_lengths = torch.cat(
                [
                    self.event_lengths,
                    torch.tensor(new_lengths, device=self.events.device),
                ]
            )[-self.memory_size :]

    def _retrieve_events(
        self, query: Tensor, current_pos: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve relevant events using similarity and contiguity"""
        if len(self.events) == 0:
            return (
                query.new_zeros(
                    self.num_heads, 0, self.head_dim * self.multiplier
                ),  # Update dimension
                query.new_zeros(
                    self.num_heads, 0, self.head_dim
                ),  # v keeps original dim
                query.new_zeros(0),
            )

        # Compute similarities using only first half of vectors if differential
        flat_query = (
            query[..., : self.head_dim].reshape(-1)
            if self.config.differential
            else query.reshape(-1)
        )
        events_for_similarity = (
            (
                self.events.view(-1, self.num_heads, self.head_dim * self.multiplier)[
                    ..., : self.head_dim
                ].reshape(len(self.events), -1)
            )
            if self.config.differential
            else self.events
        )

        similarities = F.cosine_similarity(
            flat_query.unsqueeze(0), events_for_similarity
        )

        # Get top-k similar events
        k_total = self.k_similar + self.k_contiguous
        topk_sim, topk_idx = torch.topk(similarities, k=min(k_total, len(similarities)))

        # Reconstruct events
        retrieved_k = (
            self.events[topk_idx]
            .view(len(topk_idx), self.num_heads, self.head_dim * self.multiplier)
            .transpose(0, 1)
        )
        # For values, we create a simpler representation (could be improved)
        retrieved_v = (
            retrieved_k[..., : self.head_dim]
            if self.config.differential
            else retrieved_k
        )

        # Generate positions
        positions = torch.linspace(
            current_pos - k_total, current_pos - 1, len(topk_idx), device=query.device
        )

        return retrieved_k, retrieved_v, positions
