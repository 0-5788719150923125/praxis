import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PraxisRecurrent(nn.Module):
    """
    A recurrent block using LSTM with learnable probabilistic initial states
    using Gumbel Softmax for differentiable sampling.
    """

    def __init__(
        self,
        config: "AutoConfig",
        num_latent_states: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        hidden_dim = config.num_dims
        self.temperature = temperature
        self.num_latent_states = num_latent_states

        self.norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Learnable logits for initial state distributions over latent spaces
        self.h0_logits = nn.Parameter(torch.zeros(hidden_dim, num_latent_states))
        self.c0_logits = nn.Parameter(torch.zeros(hidden_dim, num_latent_states))

        # Learnable state embeddings - one per latent state
        self.h0_embeddings = nn.Parameter(
            torch.randn(num_latent_states) / math.sqrt(num_latent_states)
        )
        self.c0_embeddings = nn.Parameter(
            torch.randn(num_latent_states) / math.sqrt(num_latent_states)
        )

        # Initialize LSTM parameters
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor = None,
        router_weights: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        """Forward pass with sampled initial states."""
        batch_size = x.size(0)

        # Allow temperature adjustment during inference
        temperature = 0.1
        if not self.training:
            self.temperature = temperature

        # Sample initial states
        h0, c0 = self.sample_initial_state(batch_size)
        initial_state = (h0, c0)

        # Process input
        normed_input = self.norm(x)
        lstm_out, _ = self.lstm(normed_input, initial_state)
        lstm_out = self.dropout(lstm_out)

        return lstm_out + x

    def sample_initial_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Sample initial states using Gumbel Softmax."""
        # Sample from Gumbel Softmax distribution for each dimension
        h0_dist = F.gumbel_softmax(
            self.h0_logits, tau=self.temperature, hard=False, dim=1
        )  # Shape: [hidden_dim, num_latent_states]
        c0_dist = F.gumbel_softmax(
            self.c0_logits, tau=self.temperature, hard=False, dim=1
        )  # Shape: [hidden_dim, num_latent_states]

        # Mix embeddings for each dimension
        h0 = torch.matmul(h0_dist, self.h0_embeddings)  # Shape: [hidden_dim]
        c0 = torch.matmul(c0_dist, self.c0_embeddings)  # Shape: [hidden_dim]

        # Expand to batch size and add lstm expected dim
        h0 = h0.unsqueeze(0).unsqueeze(0).expand(1, batch_size, -1).contiguous()
        c0 = c0.unsqueeze(0).unsqueeze(0).expand(1, batch_size, -1).contiguous()

        return h0, c0

    @property
    def state_entropy(self) -> tuple[Tensor, Tensor]:
        """Calculate entropy of the initial state distributions."""
        h0_probs = F.softmax(
            self.h0_logits, dim=1
        )  # Shape: [hidden_dim, num_latent_states]
        c0_probs = F.softmax(self.c0_logits, dim=1)

        h0_entropy = -(h0_probs * torch.log(h0_probs + 1e-10)).sum(
            1
        )  # Shape: [hidden_dim]
        c0_entropy = -(c0_probs * torch.log(c0_probs + 1e-10)).sum(1)

        return h0_entropy, c0_entropy


if __name__ == "__main__":
    import random
    import time
    from dataclasses import dataclass

    # Mock AutoConfig class
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        dropout: float = 0.1
        causal: bool = True

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()

    def run_memory_test(model, x):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
            max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        else:
            max_memory = 0

        end_time = time.time()
        return output, end_time - start_time, max_memory

    print("Running tests for PraxisRecurrent...")

    # Create model
    model = PraxisRecurrent(config).to(device)

    # Test 1: Basic Functionality (Short Sequence)
    print("\nTest 1: Short Sequence Test")
    batch_size, seq_len = 2, 32
    x_short = torch.randn(batch_size, seq_len, config.num_dims).to(device)

    try:
        output_short = model(x_short)
        print(f"✓ Short sequence output shape: {output_short.shape}")
        assert output_short.shape == x_short.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: Sequential Processing
    print("\nTest 2: Sequential Processing Test")
    try:
        # Process two sequences consecutively
        output1 = model(x_short)
        output2 = model(x_short)
        assert (
            output1.shape == output2.shape == x_short.shape
        ), "Sequential processing shape mismatch"
        print("✓ Sequential processing test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 3: Memory and Speed Test
    print("\nTest 3: Memory and Speed Test")
    batch_size, seq_len = 4, 512
    x_long = torch.randn(batch_size, seq_len, config.num_dims).to(device)

    try:
        output, elapsed_time, max_memory = run_memory_test(model, x_long)
        print(f"✓ Processing time: {elapsed_time:.4f} seconds")
        if device.type == "cuda":
            print(f"✓ Peak memory usage: {max_memory:.2f} MB")
        print("✓ Memory and speed test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 4: Gradient Flow Test
    print("\nTest 4: Gradient Flow Test")
    try:
        model.zero_grad()
        output = model(x_short)
        loss = output.mean()
        loss.backward()

        # Check if gradients are flowing
        has_grad = all(p.grad is not None for p in model.parameters())
        assert has_grad, "Some parameters have no gradients"

        # Check for exploding gradients
        max_grad = max(p.grad.abs().max() for p in model.parameters())
        assert max_grad < 1000, f"Potential exploding gradient detected: {max_grad}"

        print(f"✓ Maximum gradient magnitude: {max_grad:.4f}")
        print("✓ Gradient flow test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
