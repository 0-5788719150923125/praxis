import math

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


class SandwichTransformerBlock(nn.Module):
    """Transformer block with sandwich normalization as described in the paper"""

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

        # Self-attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=attn_dropout, batch_first=True
        )

        # MLP normalization
        self.norm3 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm4 = RMSNorm(hidden_size, eps=norm_eps)

        # MLP with SiLU activation and gating
        self.fc1 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x, attention_mask=None):
        # Sandwich format for attention
        residual = x
        x_norm = self.norm1(x)

        # Convert mask if needed
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (
                attention_mask == 0
            )  # Convert 1s (attend) to False, 0s (ignore) to True

        # Self-attention
        attn_output, _ = self.attention(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

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
    Recurrent Depth Reasoning module with adaptive per-token computation.
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
    ):
        """
        Forward pass with recurrent reasoning and optional adaptive per-token computation.

        Args:
            embedded_inputs: Tensor of shape (batch_size, seq_len, hidden_size)
            state: Optional initial state. If None, will be randomly initialized.
            attention_mask: Optional attention mask (1 = attend, 0 = ignore)
            num_iterations: Maximum number of recurrent iterations to perform
            use_adaptive_compute: Whether to use adaptive per-token computation
            convergence_threshold: Threshold for determining token convergence

        Returns:
            output_state: Final state after recurrent iterations
            all_states: List of all intermediate states
            iteration_counts: Number of iterations used for each token (if adaptive_compute=True)
        """
        batch_size, seq_len, _ = embedded_inputs.shape
        device = embedded_inputs.device

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

        # Recurrent iterations
        for iteration in range(num_iterations):
            # Store current state
            all_states.append(state.clone())
            prev_state = state.clone()

            # Combine state with input embeddings
            combined = torch.cat([state, embedded_inputs], dim=-1)
            state = self.adapter(combined)

            # Apply transformer blocks
            for block in self.blocks:
                state = block(state, attention_mask)

            # Check for token convergence if using adaptive computation
            if use_adaptive_compute and iteration > 0:
                # Compute relative state difference per token
                state_diff = torch.norm(state - prev_state, dim=2) / (
                    torch.norm(state, dim=2) + 1e-6
                )

                # Mark tokens as converged if difference is below threshold
                new_converged = (state_diff < threshold) & ~token_converged
                iteration_counts = torch.where(
                    new_converged,
                    iteration * torch.ones_like(iteration_counts),
                    iteration_counts,
                )
                token_converged = token_converged | new_converged

                # If all tokens have converged, we can stop early
                if token_converged.all():
                    break

                # For converged tokens, restore the previous state to avoid further computation
                # This is a simplification - in practice, you'd want to optimize to avoid computing these tokens
                if token_converged.any():
                    for b in range(batch_size):
                        for s in range(seq_len):
                            if token_converged[b, s]:
                                state[b, s] = prev_state[b, s]

        # For tokens that never converged, record the max iteration count
        if use_adaptive_compute:
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


def predict_from_state(state, lm_head):
    """Helper function to get predicted probabilities from a state"""
    logits = lm_head(state)
    probs = F.softmax(logits, dim=-1)
    return probs


if __name__ == "__main__":
    # Test the RecurrentDepthReasoning module with adaptive computation

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

    # Create a mock "language model head" for testing convergence
    lm_head = nn.Linear(hidden_size, vocab_size)

    # Create the model
    model = RecurrentDepthReasoning(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_layers=2,  # Using 2 layers for faster testing
        convergence_threshold=0.01,
    )

    print("Testing RecurrentDepthReasoning module with standard fixed iterations...")

    # Test basic functionality first
    output, states = model(inputs, attention_mask=mask, num_iterations=16)

    # Basic validation
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert len(states) == 16

    print("Testing adaptive per-token computation...")

    # Generate synthetic data with "easy" and "hard" tokens
    # Some tokens converge quickly, others take more iterations
    easy_tokens = torch.randn(batch_size, seq_len // 2, hidden_size) * 0.1
    hard_tokens = torch.randn(batch_size, seq_len - seq_len // 2, hidden_size)
    mixed_inputs = torch.cat([easy_tokens, hard_tokens], dim=1)

    # Run with adaptive computation
    output, states, iteration_counts = model(
        mixed_inputs,
        attention_mask=mask,
        num_iterations=32,
        use_adaptive_compute=True,
        convergence_threshold=0.03,  # Set higher for testing
    )

    # Check that tokens converged at different rates
    print("\nToken convergence iteration counts:")
    print(iteration_counts)

    # Calculate average iterations per token
    avg_iterations = iteration_counts.float().mean().item()
    print(f"Average iterations per token: {avg_iterations:.2f} (max: 32)")

    # Verify that early tokens (easy ones) converged faster
    easy_avg = iteration_counts[:, : seq_len // 2].float().mean().item()
    hard_avg = iteration_counts[:, seq_len // 2 :].float().mean().item()
    print(f"Average iterations for 'easy' tokens: {easy_avg:.2f}")
    print(f"Average iterations for 'hard' tokens: {hard_avg:.2f}")

    # Compare compute savings
    total_possible_iterations = batch_size * seq_len * 32
    actual_iterations = iteration_counts.sum().item()
    compute_savings = 1 - (actual_iterations / total_possible_iterations)
    print(f"Compute saved with adaptive computation: {compute_savings:.2%}")

    print("\nAll tests completed!")
