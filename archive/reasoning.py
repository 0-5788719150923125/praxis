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

        # Self-attention (this assumes you have your own attention implementation)
        # You can replace this with your own attention module
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
            # For nn.MultiheadAttention, key_padding_mask should be:
            # - True for positions to be masked (ignored)
            # - False for positions to be attended to
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
    Recurrent Depth Reasoning module as described in the paper.

    This module implements the core recurrent reasoning mechanism where a transformer block
    is applied multiple times to allow for "thinking" in latent space.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        num_layers=4,
        norm_eps=1e-6,
        state_init_std=0.4,
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
        self, embedded_inputs, state=None, attention_mask=None, num_iterations=32
    ):
        """
        Forward pass with recurrent reasoning.

        Args:
            embedded_inputs: Tensor of shape (batch_size, seq_len, hidden_size)
            state: Optional initial state. If None, will be randomly initialized.
            attention_mask: Optional attention mask (1 = attend, 0 = ignore)
            num_iterations: Number of recurrent iterations to perform

        Returns:
            output_state: Final state after recurrent iterations
            all_states: List of all intermediate states if return_all_states is True
        """
        batch_size, seq_len, _ = embedded_inputs.shape
        device = embedded_inputs.device

        # Initialize state if not provided
        if state is None:
            state = self.initialize_state(batch_size, seq_len, device)

        # Store all states if requested
        all_states = []

        # Recurrent iterations
        for _ in range(num_iterations):
            # Store current state
            all_states.append(state)

            # Combine state with input embeddings
            combined = torch.cat([state, embedded_inputs], dim=-1)
            state = self.adapter(combined)

            # Apply transformer blocks
            for block in self.blocks:
                state = block(state, attention_mask)

        # Final normalization
        output_state = self.final_norm(state)

        return output_state, all_states


if __name__ == "__main__":
    # Test the RecurrentDepthReasoning module

    # Model parameters
    hidden_size = 128
    num_attention_heads = 4
    intermediate_size = 512
    batch_size = 2
    seq_len = 16

    # Create a random input
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len)  # All tokens are attended to

    # Create the model
    model = RecurrentDepthReasoning(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        num_layers=2,  # Using 2 layers for faster testing
    )

    # Test iteration counts
    iteration_counts = [1, 4, 8, 16]

    print("Testing RecurrentDepthReasoning module...")

    for iterations in iteration_counts:
        # Run forward pass
        output, states = model(inputs, attention_mask=mask, num_iterations=iterations)

        # Basic validation
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert len(states) == iterations

        # Check that output changes with more iterations
        if iterations > 1:
            state_diff = torch.abs(states[-1] - states[0]).mean().item()
            print(f"Iterations: {iterations}, Average state change: {state_diff:.6f}")

    # Test path independence (same final state regardless of initialization)
    print("\nTesting path independence...")
    iterations = 16

    # Two different initializations
    state1 = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    state2 = torch.randn(batch_size, seq_len, hidden_size) * 0.5

    # Run with different initializations
    output1, _ = model(
        inputs, state=state1, attention_mask=mask, num_iterations=iterations
    )
    output2, _ = model(
        inputs, state=state2, attention_mask=mask, num_iterations=iterations
    )

    # Check how close the final states are
    state_diff = torch.abs(output1 - output2).mean().item()
    print(f"Path independence test - Difference between final states: {state_diff:.6f}")

    # Test recurrent computation improvement
    print("\nTesting improvement with more iterations on random data...")

    # Simple "loss" - just the MSE to a target tensor to see if more iterations help convergence
    target = torch.randn(batch_size, seq_len, hidden_size)

    for iterations in iteration_counts:
        output, _ = model(inputs, attention_mask=mask, num_iterations=iterations)
        loss = F.mse_loss(output, target)
        print(f"Iterations: {iterations}, Loss: {loss.item():.6f}")

    print("\nAll tests completed!")
