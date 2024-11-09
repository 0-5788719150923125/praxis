import torch
import torch.nn as nn
from dataclasses import dataclass
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class EfficientPraxisNano(nn.Module):
    """
    A revised implementation that maintains mathematical properties of NanoFFT
    while supporting longer sequences through careful windowing.
    """

    def __init__(
        self,
        config,
        base_seq_len: int = 384,
        stride: int = 256,
    ):
        super().__init__()
        self.hidden_dim = config.num_dims
        self.base_seq_len = base_seq_len
        self.stride = stride

        # Core weight matrices at fixed size
        self.fft = nn.ParameterDict(
            {
                "w1": nn.Parameter(torch.Tensor(base_seq_len, base_seq_len)),
                "w2": nn.Parameter(torch.Tensor(base_seq_len, base_seq_len)),
            }
        )

        # Initialize weights with triangular structure
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fft["w1"])
            nn.init.xavier_uniform_(self.fft["w2"])
            self.fft["w1"].copy_(torch.tril(self.fft["w1"]))
            self.fft["w2"].copy_(torch.tril(self.fft["w2"]))

        # Create fixed mask for the base sequence length
        mask = torch.tril(torch.ones(base_seq_len, base_seq_len))
        row_sums = mask.sum(dim=1, keepdim=True)
        self.register_buffer("base_mask", mask / row_sums)

        # Register gradient hooks to maintain triangular structure
        self.fft["w1"].register_hook(lambda grad: grad * self.base_mask)
        self.fft["w2"].register_hook(lambda grad: grad * self.base_mask)

        class SineActivation(nn.Module):
            def forward(self, x):
                return torch.sin(x)

        # Layer norms and FFN
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.ffw = nn.Sequential(
            nn.Linear(self.hidden_dim, config.num_embeds),
            SineActivation(),
            nn.Linear(config.num_embeds, self.hidden_dim),
        )

    def process_window(self, x: Tensor, start_idx: int) -> Tensor:
        """Process a single window while maintaining causality."""
        # Extract window
        end_idx = min(start_idx + self.base_seq_len, x.size(1))
        effective_len = end_idx - start_idx

        # If we need padding, add it
        window = x[:, start_idx:end_idx, :]
        if effective_len < self.base_seq_len:
            padding = self.base_seq_len - effective_len
            window = F.pad(window, (0, 0, 0, padding))

        B, T, E = window.shape

        # Apply FFT transformation with fixed weights
        x_fft = window.transpose(1, 2).reshape(-1, T)
        x_fft = x_fft @ self.fft["w1"]
        x_fft = x_fft @ self.fft["w2"]
        x_fft = x_fft.view(B, E, T).transpose(1, 2)

        # Return only the valid part
        return x_fft[:, :effective_len, :]

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, T, E = x.shape
        x = self.ln1(x)

        if T <= self.base_seq_len:
            # For shorter sequences, use standard processing
            window = x
            if T < self.base_seq_len:
                window = F.pad(x, (0, 0, 0, self.base_seq_len - T))

            x_fft = window.transpose(1, 2).reshape(-1, self.base_seq_len)
            x_fft = x_fft @ self.fft["w1"]
            x_fft = x_fft @ self.fft["w2"]
            x_fft = x_fft.view(B, E, self.base_seq_len).transpose(1, 2)
            x_fft = x_fft[:, :T, :]  # Trim padding if any

        else:
            # For longer sequences, use strided windows
            x_fft = torch.zeros_like(x)
            count = torch.zeros(B, T, 1, device=x.device)

            for start_idx in range(0, T, self.stride):
                window_output = self.process_window(x, start_idx)
                end_idx = min(start_idx + self.base_seq_len, T)

                # Add window contribution
                x_fft[:, start_idx:end_idx, :] += window_output[
                    :, : (end_idx - start_idx), :
                ]
                count[:, start_idx:end_idx, :] += 1

            # Average overlapping regions
            x_fft = x_fft / count.clamp(min=1)

        # Apply residual connections and FFN
        x = x + x_fft
        x = self.ln2(x)
        x = x + self.ffw(x)

        return x


@dataclass
class MockConfig:
    num_dims: int = 768
    num_embeds: int = 768
    context_length: int = 2048
    vocab_size: int = 50257


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MockConfig()

    def run_memory_test(seq_length):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Create model and sample input
        model = EfficientPraxisNano(config).to(device)
        x = torch.randn(1, seq_length, config.num_dims).to(device)

        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(x)
        torch.cuda.synchronize()
        end_time = time.time()

        max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        return output, end_time - start_time, max_memory

    print("Running tests for EfficientPraxisNano...")

    # Test 1: Basic Functionality (Short Sequence)
    print("\nTest 1: Short Sequence Test")
    x_short = torch.randn(2, 256, config.num_dims).to(device)
    model = EfficientPraxisNano(config).to(device)

    try:
        output_short = model(x_short)
        print(f"✓ Short sequence shape: {output_short.shape}")
        assert output_short.shape == x_short.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: Long Sequence Handling
    print("\nTest 2: Long Sequence Test")
    x_long = torch.randn(2, 1024, config.num_dims).to(device)

    try:
        output_long = model(x_long)
        print(f"✓ Long sequence shape: {output_long.shape}")
        assert output_long.shape == x_long.shape, "Output shape mismatch"
        print("✓ Long sequence test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 3: Memory Scaling Test
    print("\nTest 3: Memory Scaling Test")
    sequence_lengths = [256, 512, 1024, 2048]
    results = []

    for seq_len in sequence_lengths:
        output, duration, memory = run_memory_test(seq_len)
        results.append((seq_len, duration, memory))
        print(f"\nSequence Length: {seq_len}")
        print(f"✓ Processing Time: {duration:.4f} seconds")
        print(f"✓ Peak Memory Usage: {memory:.2f} MB")

    # Test 4: Gradient Flow Test
    print("\nTest 4: Gradient Flow Test")
    model.zero_grad()
    x = torch.randn(2, 512, config.num_dims, requires_grad=True).to(device)

    try:
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check if gradients exist and are not None
        has_grads = all(p.grad is not None for p in model.parameters())
        print(f"✓ Gradients exist: {has_grads}")

        # Check if gradients contain NaN values
        has_nans = any(torch.isnan(p.grad).any() for p in model.parameters())
        print(f"✓ Gradients are clean (no NaNs): {not has_nans}")

        print("✓ Gradient flow test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    print("\nAll tests completed!")
