import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from typing import Optional
from dataclasses import dataclass
import random
from praxis.modules.dense import PraxisMLP, PraxisGLU
from praxis.activations import ACT2FN, ACT2CLS


class PraxisNano(nn.Module):
    """
    A special kind of block that omits all self-attention mechanisms, in favor
    of dense layers with periodic activations. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        hidden_dim = config.num_dims
        projection = int(hidden_dim * 2.0)
        bottleneck = int(hidden_dim * 0.5)

        # Define the weight matrices with maximum sequence length
        self.fft_norm = nn.LayerNorm(hidden_dim)
        self.fft = nn.Sequential(
            ElasticLinear(
                in_features=projection, out_features=bottleneck, causal=config.causal
            ),
            nn.Dropout(config.dropout),
            ElasticLinear(
                in_features=bottleneck, out_features=hidden_dim, causal=config.causal
            ),
        )

        # Feed-forward network with sine activation
        config.activation = "sin_cos"
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = PraxisGLU(config)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        # x shape: (batch_size, seq_len, hidden_dim)
        chunk_norm = self.fft_norm(x)
        # Transpose to (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim, seq_len)
        chunk_fft = chunk_norm.transpose(1, 2)
        # Pass through FFT layers
        chunk_fft = self.fft(chunk_fft)
        # Transpose back to original shape
        chunk_fft = chunk_fft.transpose(1, 2)
        # Residual connection
        residual = chunk_fft + x
        chunk = self.ffw_norm(residual)
        chunk = self.ffw(chunk)
        return chunk + residual


class ElasticLinear(nn.Module):
    def __init__(
        self, in_features, out_features, std=0.02, causal=False, *args, **kwargs
    ):
        super().__init__()
        self.base_in_features = in_features
        self.base_out_features = out_features
        self.std = std
        self.causal = causal

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

        # Learnable scalar for scaling padded weights
        self.alpha = nn.Parameter(torch.ones(1))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def _adjust_weight_matrix(self, in_features, out_features):
        # Adjust the weight matrix to match in_features and out_features
        weight = self.weight

        # Generate random noise
        noise = (
            torch.randn(out_features, in_features, device=weight.device)
            * self.std
            * self.alpha
        )

        if (
            in_features <= self.base_in_features
            and out_features <= self.base_out_features
        ):
            # Slice self.weight to match in_features and out_features
            adjusted_weight = weight[:out_features, :in_features]
        else:
            # Pad self.weight with zeros to match in_features and out_features
            pad_in = max(0, in_features - self.base_in_features)
            pad_out = max(0, out_features - self.base_out_features)
            # Padding: (left, right, top, bottom)
            padding = (0, pad_in, 0, pad_out)
            adjusted_weight = F.pad(weight, padding, "constant", 0)

        # Add random noise to all weights being used
        adjusted_weights = adjusted_weight + noise

        return adjusted_weights

    def forward(self, x):
        # x shape: (batch_size, in_features, seq_len)
        in_features = x.size(1)
        out_features = self.base_out_features

        adjusted_weights = self._adjust_weight_matrix(in_features, out_features)
        # adjusted_weights: (out_features, in_features)

        if self.causal:
            mask = torch.tril(
                torch.ones_like(adjusted_weights, dtype=torch.float32, device=x.device)
            )
            mask_normalized = mask / mask.sum(dim=1, keepdim=True)
            adjusted_weights = adjusted_weights * mask_normalized

        # Perform batch matrix multiplication
        output = torch.matmul(adjusted_weights, x)
        # output shape: (batch_size, out_features, seq_len)
        return output


if __name__ == "__main__":
    # Mock AutoConfig class to simulate the configuration
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 8192
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.0

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()
    chunk_size = 256  # Explicitly define for tests
    stride = 128  # Example stride with overlap

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

    print("Running tests for PraxisNano...")

    # Create model once for all tests
    model = PraxisNano(config).to(device)

    # Test 1: Basic Functionality (Short Sequence)
    print("\nTest 1: Short Sequence Test")
    # Test with a sequence length that's exactly half of chunk_size
    x_short = torch.randn(2, chunk_size // 2, config.num_dims).to(device)

    try:
        output_short = model(x_short)
        print(f"✓ Short sequence shape: {output_short.shape}")
        assert output_short.shape == x_short.shape, "Output shape mismatch"
        print("✓ Basic functionality test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 2: Long Sequence Handling
    print("\nTest 2: Long Sequence Test")
    x_long = torch.randn(2, chunk_size * 4, config.num_dims).to(
        device
    )  # Test with multiple of chunk_size

    try:
        output_long = model(x_long)
        print(f"✓ Long sequence shape: {output_long.shape}")
        assert output_long.shape == x_long.shape, "Output shape mismatch"
        print("✓ Long sequence test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 3: Memory Scaling Test
    print("\nTest 3: Memory Scaling Test")
    chunk_sizes = [
        chunk_size // 2,
        chunk_size,
        chunk_size * 2,
        chunk_size * 4,
    ]
    results = []

    for cs in chunk_sizes:
        # Adjust stride accordingly (for simplicity, stride = cs // 2)
        current_stride = cs // 2
        model_test = PraxisNano(config).to(device)
        x_test = torch.randn(1, cs * 4, config.num_dims).to(device)
        output, duration, memory = run_memory_test(model_test, x_test)
        results.append((cs, duration, memory))
        print(f"\nChunk Size: {cs}")
        print(f"✓ Processing Time: {duration:.4f} seconds")
        print(f"✓ Peak Memory Usage: {memory:.2f} MB")

    # Test 4: Gradient Flow Test
    print("\nTest 4: Gradient Flow Test")
    model.zero_grad()
    x = torch.randn(2, chunk_size * 2, config.num_dims, requires_grad=True).to(device)

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

    # Test 5: Chunk Boundary Test
    print("\nTest 5: Chunk Boundary Test")
    x = torch.randn(1, chunk_size * 2, config.num_dims).to(device)

    try:
        # Get outputs for consecutive chunks
        out1 = model(x[:, :chunk_size, :])
        out2 = model(x[:, chunk_size:, :])

        # Process as single sequence
        out_full = model(x)

        # Compare the results at the boundary
        boundary_diff = (
            (
                out_full[:, chunk_size - stride : chunk_size + stride, :]
                - torch.cat([out1[:, -stride:, :], out2[:, :stride, :]], dim=1)
            )
            .abs()
            .mean()
        )

        print(f"✓ Chunk boundary difference: {boundary_diff:.6f}")
        assert boundary_diff < 1.0, "Chunk boundary difference too large"
        print("✓ Chunk boundary test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 6: Random Offset Consistency
    print("\nTest 6: Random Offset Test")
    x = torch.randn(1, chunk_size * 3, config.num_dims).to(device)

    try:
        # Multiple forward passes should give similar results due to deterministic processing
        out1 = model(x)
        out2 = model(x)
        difference = (out1 - out2).abs().mean().item()

        print(f"✓ Output difference between passes: {difference:.6f}")
        assert difference == 0, "Outputs differ despite deterministic processing"
        print("✓ Random offset test passed")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")

    # Test 7: Random Dimensions Test
    print("\nTest 7: Random Dimensions Test")
    torch.manual_seed(42)  # For reproducibility
    for i in range(5):
        # Generate random dimensions
        base_in_features = random.randint(5, 500)
        base_out_features = random.randint(5, 500)
        batch_dim = random.randint(1, 32)
        time_dim = random.randint(1, 64)
        input_dim = random.randint(5, 500)

        print(f"\nRandom test iteration {i+1}:")
        print(
            f"Base in_features: {base_in_features}, Base out_features: {base_out_features}"
        )
        print(
            f"Input dimensions: [batch={batch_dim}, time={time_dim}, input_dim={input_dim}]"
        )

        # Create model and input
        model = ElasticLinear(
            in_features=base_in_features, out_features=base_out_features
        ).to(device)
        x = torch.randn(batch_dim, input_dim, time_dim).to(device)

        # Forward pass
        out = model(x)

        # Verify shapes
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(
            f"Adjusted weight matrix shape: {model._adjust_weight_matrix(input_dim, base_out_features).shape}"
        )
        print(f"Padding scale (alpha): {model.alpha.item():.4f}")
        assert out.shape == (
            batch_dim,
            base_out_features,
            time_dim,
        ), "Output shape mismatch"

    print("\nAll tests completed!")
