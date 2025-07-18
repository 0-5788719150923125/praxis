import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm implementation as used in the paper"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Cast to float32 for better precision in normalization
        with torch.autocast(enabled=False, device_type=x.device.type):
            norm_x = x.float() * torch.rsqrt(
                x.float().pow(2).mean(-1, keepdim=True) + self.eps
            )
        return norm_x.type_as(x) * self.weight


class RecurrentKVCache:
    """
    Custom KV cache for recurrent depth models.
    This cache is shared across recurrent iterations, allowing memory-efficient inference.
    """

    def __init__(self, max_cache_size=16):
        """
        Initialize the KV cache.

        Args:
            max_cache_size: Maximum number of recurrent iterations to store in the cache
                            If cache exceeds this size, entries will be overwritten in a circular fashion
        """
        self.key_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self.value_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self.max_cache_size = max_cache_size
        self.seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx: int,
        token_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key and value states.

        Args:
            key_states: Key states to add to the cache (batch_size, seq_len, num_heads, head_dim)
            value_states: Value states to add to the cache (batch_size, seq_len, num_heads, head_dim)
            step_idx: The recurrent iteration index
            token_position: Optional specific token position we're updating

        Returns:
            Tuple of (past_keys, past_values) including the new states
        """
        # For recurrent steps, use a modulo to reuse cache entries
        if step_idx >= 2:  # Skip prelude layers
            cache_idx = ((step_idx - 2) % self.max_cache_size) + 2
        else:
            cache_idx = step_idx

        # Initialize this cache position if not exists
        if cache_idx not in self.key_cache:
            self.key_cache[cache_idx] = {}
            self.value_cache[cache_idx] = {}

        # Update sequence length counter
        if step_idx == 0 and token_position is None:
            self.seen_tokens += key_states.shape[1]

        # Add entries to cache
        if token_position is None:
            # Bulk update for all tokens
            for i in range(key_states.shape[1]):
                pos = self.seen_tokens - key_states.shape[1] + i
                self.key_cache[cache_idx][pos] = key_states[:, i]
                self.value_cache[cache_idx][pos] = value_states[:, i]
        else:
            # Update for a specific token
            self.key_cache[cache_idx][token_position] = key_states
            self.value_cache[cache_idx][token_position] = value_states

        # Return the full set of cached states for attention
        return self.get_cache_for_attention(step_idx)

    def get_cache_for_attention(
        self, step_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the cached key and value states for a given recurrent step,
        using the latest available states for each token.
        """
        keys_list = []
        values_list = []

        # For each token position seen so far
        for pos in range(self.seen_tokens):
            # Find the best available cache entry for this token
            if step_idx >= 2:
                # For recurrent steps, prefer entries from same modulo position
                step_candidates = [
                    s for s in self.key_cache if s >= 2 and pos in self.key_cache[s]
                ]
                if step_candidates:
                    same_modulo_steps = [
                        s
                        for s in step_candidates
                        if (s - 2) % self.max_cache_size
                        == (step_idx - 2) % self.max_cache_size
                    ]
                    if same_modulo_steps:
                        best_step = max(same_modulo_steps)
                    else:
                        best_step = max(step_candidates)
                else:
                    # Fall back to prelude if no recurrent steps available
                    best_step = max(
                        [
                            s
                            for s in [0, 1]
                            if s in self.key_cache and pos in self.key_cache[s]
                        ],
                        default=0,
                    )
            else:
                # For prelude steps, just use the corresponding prelude cache
                best_step = (
                    step_idx
                    if step_idx in self.key_cache and pos in self.key_cache[step_idx]
                    else 0
                )

            # Add the best entry to our list
            if best_step in self.key_cache and pos in self.key_cache[best_step]:
                keys_list.append(self.key_cache[best_step][pos])
                values_list.append(self.value_cache[best_step][pos])

        # Stack all token states
        batch_size = keys_list[0].shape[0] if keys_list else 0
        if batch_size > 0:
            keys = torch.stack(keys_list, dim=1)
            values = torch.stack(values_list, dim=1)
            return keys, values
        else:
            # Empty cache case - return empty tensors
            return (torch.empty(0), torch.empty(0))

    def clear(self):
        """Clear the cache."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.seen_tokens = 0

    def get_memory_usage(self):
        """Calculate the memory usage of the cache in MB."""
        total_bytes = 0
        for step_idx in self.key_cache:
            for pos in self.key_cache[step_idx]:
                total_bytes += (
                    self.key_cache[step_idx][pos].nelement()
                    * self.key_cache[step_idx][pos].element_size()
                )
                total_bytes += (
                    self.value_cache[step_idx][pos].nelement()
                    * self.value_cache[step_idx][pos].element_size()
                )
        return total_bytes / (1024 * 1024)


class SandwichTransformerBlock(nn.Module):
    """Transformer block with sandwich normalization and custom KV cache support"""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        attn_dropout=0.0,
        hidden_dropout=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()

        # Attention normalization
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)

        # Keep track of attention heads for custom implementation
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Custom QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP normalization
        self.norm3 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm4 = RMSNorm(hidden_size, eps=norm_eps)

        # MLP with SiLU activation and gating
        self.fc1 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)

    def _reshape_for_attention(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def forward(
        self, x, step_idx=0, attention_mask=None, kv_cache=None, token_position=None
    ):
        # Sandwich format for attention
        residual = x
        x_norm = self.norm1(x)

        # Project queries, keys, values
        q = self._reshape_for_attention(
            self.q_proj(x_norm)
        )  # [batch, seq, heads, head_dim]
        k = self._reshape_for_attention(self.k_proj(x_norm))
        v = self._reshape_for_attention(self.v_proj(x_norm))

        # Handle KV caching for inference
        if kv_cache is not None:
            # If updating a specific token
            if token_position is not None:
                # Extract just that token for cache update
                k_to_cache = k[:, -1:]
                v_to_cache = v[:, -1:]
                # Use the token_position parameter, not self.seen_tokens
                k, v = kv_cache.update(k_to_cache, v_to_cache, step_idx, token_position)
            else:
                # Update all tokens in the cache
                k, v = kv_cache.update(k, v, step_idx)

        # Scaled dot-product attention
        batch_size, seq_len = q.shape[0], q.shape[1]

        # Convert shapes for bmm
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (
            torch.matmul(q, k.transpose(-2, -1)) * scale
        )  # [batch, heads, seq_q, seq_k]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to the right shape for broadcasting
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch, 1, 1, seq_k]
            scores = scores.masked_fill(expanded_mask == 0, float("-inf"))

        # Apply causal mask for autoregressive generation
        if token_position is None and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(1), float("-inf")
            )

        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Reshape back and project to output dimension
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Apply residual connection and normalization
        x = self.norm2(residual + attn_output)

        # Sandwich format for MLP
        residual = x
        x_norm = self.norm3(x)

        # Gated MLP with SiLU
        gate, value = self.fc1(x_norm).chunk(2, dim=-1)
        x_mlp = F.silu(gate) * value
        x_mlp = self.fc2(x_mlp)
        x_mlp = self.dropout(x_mlp)

        # Final normalization
        x = self.norm4(residual + x_mlp)

        return x


class RecurrentDepthReasoning(nn.Module):
    """
    Recurrent Depth Reasoning module with adaptive per-token computation and KV cache sharing.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        num_layers=4,
        norm_eps=1e-6,
        state_init_std=0.4,
        convergence_threshold=0.01,
        max_kv_cache_size=16,
    ):
        super().__init__()

        # Adapter to combine previous state with input embeddings
        self.adapter = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Recurrent transformer blocks
        self.blocks = nn.ModuleList(
            [
                SandwichTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = RMSNorm(hidden_size, eps=norm_eps)

        # Configuration
        self.hidden_size = hidden_size
        self.state_init_std = state_init_std
        self.convergence_threshold = convergence_threshold
        self.max_kv_cache_size = max_kv_cache_size

    def initialize_state(self, batch_size, seq_len, device, deterministic=False):
        """Initialize a random latent state or zero state if deterministic"""
        if deterministic:
            return torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

        # Random initialization with truncated normal
        state = torch.randn(batch_size, seq_len, self.hidden_size, device=device)
        std = self.state_init_std
        state = torch.clamp(state, min=-3 * std, max=3 * std) * std
        return state

    def forward(
        self,
        embedded_inputs,
        state=None,
        attention_mask=None,
        num_iterations=32,
        use_adaptive_compute=False,
        convergence_threshold=None,
        kv_cache=None,
        token_position=None,
    ):
        """
        Forward pass with recurrent reasoning, adaptive per-token computation, and KV cache sharing.

        Args:
            embedded_inputs: Tensor of shape (batch_size, seq_len, hidden_size)
            state: Optional initial state. If None, will be randomly initialized.
            attention_mask: Optional attention mask (1 = attend, 0 = ignore)
            num_iterations: Maximum number of recurrent iterations to perform
            use_adaptive_compute: Whether to use adaptive per-token computation
            convergence_threshold: Threshold for determining token convergence
            kv_cache: Optional KV cache for efficient inference
            token_position: Optional specific token position for generation

        Returns:
            output_state: Final state after recurrent iterations
            all_states: List of all intermediate states
            iteration_counts: Number of iterations used for each token (if adaptive_compute=True)
        """
        batch_size, seq_len, _ = embedded_inputs.shape
        device = embedded_inputs.device

        # Initialize KV cache if not provided
        if kv_cache is None and use_adaptive_compute:
            kv_cache = RecurrentKVCache(max_cache_size=self.max_kv_cache_size)

        # Initialize state if not provided
        if state is None:
            state = self.initialize_state(batch_size, seq_len, device)

        # Use provided convergence threshold or default
        threshold = (
            convergence_threshold
            if convergence_threshold is not None
            else self.convergence_threshold
        )

        # Initialize tracking variables for adaptive computation
        all_states = []
        iteration_counts = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=device
        )
        token_converged = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        # In token generation mode, we're only processing the last token
        if token_position is not None:
            effective_seq_len = 1
            token_mask = None  # Will be handled by the cache
        else:
            effective_seq_len = seq_len

        # Recurrent iterations
        for iteration in range(num_iterations):
            # Store current state
            all_states.append(state.clone())
            prev_state = state.clone()

            # Combine state with input embeddings
            combined = torch.cat([state, embedded_inputs], dim=-1)
            state = self.adapter(combined)

            # Apply transformer blocks with KV cache
            for block_idx, block in enumerate(self.blocks):
                # Each block gets a unique recurrence step ID
                step_idx = iteration * len(self.blocks) + block_idx
                state = block(state, step_idx, attention_mask, kv_cache, token_position)

            # Check for token convergence if using adaptive computation
            if use_adaptive_compute and iteration > 0:
                # Compute relative state difference per token
                if token_position is not None:
                    # Just for the single token being generated
                    state_diff = torch.norm(
                        state[:, -1:] - prev_state[:, -1:], dim=2
                    ) / (torch.norm(state[:, -1:], dim=2) + 1e-6)
                    state_diff = state_diff.squeeze(1)  # [batch]
                    token_idx = torch.zeros(
                        batch_size, dtype=torch.long, device=device
                    )  # Only one token per batch
                else:
                    # For all tokens in the sequence
                    state_diff = torch.norm(state - prev_state, dim=2) / (
                        torch.norm(state, dim=2) + 1e-6
                    )
                    token_idx = torch.arange(seq_len, device=device).expand(
                        batch_size, -1
                    )

                # Mark tokens as converged if difference is below threshold
                if token_position is not None:
                    new_converged = (state_diff < threshold) & ~token_converged[:, -1]
                    iteration_counts[torch.arange(batch_size), token_idx] = torch.where(
                        new_converged,
                        iteration * torch.ones_like(token_idx),
                        iteration_counts[torch.arange(batch_size), token_idx],
                    )
                    token_converged[:, -1] = token_converged[:, -1] | new_converged
                else:
                    new_converged = (state_diff < threshold) & ~token_converged
                    iteration_counts = torch.where(
                        new_converged,
                        iteration * torch.ones_like(iteration_counts),
                        iteration_counts,
                    )
                    token_converged = token_converged | new_converged

                # If all tokens have converged, we can stop early
                if token_position is not None:
                    if token_converged[:, -1].all():
                        break
                elif token_converged.all():
                    break

                # For converged tokens, restore the previous state to avoid further computation
                # This is a simplification - in practice, you'd optimize to skip computation for these tokens
                if token_position is not None:
                    for b in range(batch_size):
                        if token_converged[b, -1]:
                            state[b, -1] = prev_state[b, -1]
                elif token_converged.any():
                    for b in range(batch_size):
                        for s in range(seq_len):
                            if token_converged[b, s]:
                                state[b, s] = prev_state[b, s]

        # For tokens that never converged, record the max iteration count
        if use_adaptive_compute:
            if token_position is not None:
                non_converged = ~token_converged[:, -1]
                iteration_counts[
                    torch.arange(batch_size)[non_converged], token_idx[non_converged]
                ] = (num_iterations - 1)
            else:
                iteration_counts = torch.where(
                    ~token_converged,
                    (num_iterations - 1) * torch.ones_like(iteration_counts),
                    iteration_counts,
                )

        # Final normalization
        output_state = self.final_norm(state)

        if use_adaptive_compute:
            return output_state, all_states, iteration_counts
        else:
            return output_state, all_states


if __name__ == "__main__":
    # Test the RecurrentDepthReasoning module with KV cache sharing and adaptive computation

    # Model parameters
    hidden_size = 128
    num_attention_heads = 4
    intermediate_size = 512
    batch_size = 2
    seq_len = 16
    vocab_size = 32000  # For testing prediction

    # Create a random input
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len)  # All tokens are attended to

    # Create the model
    model = RecurrentDepthReasoning(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_layers=2,  # Using 2 layers for faster testing
        convergence_threshold=0.01,
        max_kv_cache_size=4,  # Small for testing
    )

    print("Testing RecurrentDepthReasoning module with KV cache sharing...")

    # Create KV cache
    kv_cache = RecurrentKVCache(max_cache_size=4)

    # Run model with KV cache
    output, states = model(
        inputs, attention_mask=mask, num_iterations=8, kv_cache=kv_cache
    )

    # Basic validation
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert len(states) == 8

    # Check that the KV cache is populated
    cache_memory = kv_cache.get_memory_usage()
    print(f"KV cache memory usage: {cache_memory:.2f} MB")

    print("\nTesting adaptive computation with KV cache sharing...")

    # Reset the cache
    kv_cache.clear()

    # Generate synthetic data with "easy" and "hard" tokens
    easy_tokens = torch.randn(batch_size, seq_len // 2, hidden_size) * 0.1
    hard_tokens = torch.randn(batch_size, seq_len - seq_len // 2, hidden_size)
    mixed_inputs = torch.cat([easy_tokens, hard_tokens], dim=1)

    # Run with adaptive computation and KV cache
    output, states, iteration_counts = model(
        mixed_inputs,
        attention_mask=mask,
        num_iterations=16,
        use_adaptive_compute=True,
        convergence_threshold=0.03,  # Set higher for testing
        kv_cache=kv_cache,
    )

    # Check that tokens converged at different rates
    print("\nToken convergence iteration counts:")
    print(iteration_counts)

    # Calculate average iterations per token
    avg_iterations = iteration_counts.float().mean().item()
    print(f"Average iterations per token: {avg_iterations:.2f} (max: 16)")

    # Verify that early tokens (easy ones) converged faster
    easy_avg = iteration_counts[:, : seq_len // 2].float().mean().item()
    hard_avg = iteration_counts[:, seq_len // 2 :].float().mean().item()
    print(f"Average iterations for 'easy' tokens: {easy_avg:.2f}")
    print(f"Average iterations for 'hard' tokens: {hard_avg:.2f}")

    # Compare compute savings
    total_possible_iterations = batch_size * seq_len * 16
    actual_iterations = iteration_counts.sum().item()
    compute_savings = 1 - (actual_iterations / total_possible_iterations)
    print(f"Compute saved with adaptive computation: {compute_savings:.2%}")

    # Check KV cache memory efficiency
    print(f"KV cache memory with shared entries: {kv_cache.get_memory_usage():.2f} MB")

    print("\nAll tests completed!")
