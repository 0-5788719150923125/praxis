import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from transformers import AutoConfig
from typing import Optional
from praxis.modules.dense import PraxisGLU, PraxisMLP
from praxis.activations import ACT2FN
from dataclasses import dataclass


class PraxisNano(nn.Module):
    """
    A special kind of block that omits the self-attention mechanism, in favor
    of dense layers with periodic activations, sequence chunking and state
    preservation. Inspired by NanoFFT:
    https://github.com/timurgepard/nanoFFT
    """

    def __init__(self, config: "AutoConfig", chunk_size: int = 64, *args, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.stride = chunk_size // 2

        embed_dim = config.num_embeds
        hidden_dim = config.num_dims

        self.fft_norm = nn.LayerNorm(hidden_dim)
        self.fft = nn.Sequential(
            TriLinear(chunk_size, int(chunk_size * 0.75)),
            TriLinear(int(chunk_size * 0.75), chunk_size),
        )

        config.activation = "sin"
        self.ffw_norm = nn.LayerNorm(hidden_dim)
        self.ffw = PraxisGLU(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, E = x.shape
        chunk_size = self.chunk_size
        stride = self.stride
        device = x.device

        # Initialize output tensor
        output = torch.zeros(B, T, E, device=device)

        # Initialize state tensor for previous chunk's overlapping region
        prev_state = None

        for start in range(0, T, stride):
            end = min(start + chunk_size, T)
            current_size = end - start

            # Extract and maybe pad current chunk
            if current_size < chunk_size:
                chunk = F.pad(
                    x[:, start:end, :],
                    (0, 0, 0, chunk_size - current_size),
                    "constant",
                    0,
                )
            else:
                chunk = x[:, start:end, :]

            # Blend with previous state if exists
            if prev_state is not None:
                overlap_size = min(stride, current_size)
                # Create blended version of overlap region
                alpha = torch.linspace(0, 1, overlap_size, device=device).view(1, -1, 1)
                blended = (
                    prev_state[:, -overlap_size:, :] * (1 - alpha)
                    + chunk[:, :overlap_size, :] * alpha
                )
                # Create new chunk with blended region
                chunk = torch.cat([blended, chunk[:, overlap_size:, :]], dim=1)

            # Process the chunk
            processed_chunk = self.process_chunk(chunk)

            # Store state for next iteration if needed
            if start + stride < T:
                prev_state = processed_chunk

            # Remove padding if necessary and store in output
            if current_size < chunk_size:
                processed_chunk = processed_chunk[:, :current_size, :]
            output[:, start:end, :] = processed_chunk[:, :current_size, :]

        return output

    def process_chunk(self, chunk: Tensor) -> Tensor:
        residual = chunk
        chunk_norm = self.fft_norm(chunk)
        chunk_fft = chunk_norm.transpose(1, 2)
        chunk_fft = self.fft(chunk_fft)
        chunk_fft = chunk_fft.transpose(1, 2)
        chunk = chunk_fft + residual
        residual = chunk
        chunk = self.ffw_norm(chunk)
        chunk = self.ffw(chunk)
        return chunk + residual


class TriLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

        # Create a lower triangular mask
        causal_mask = torch.tril(
            torch.ones((out_features, in_features), dtype=torch.float32)
        )

        # Apply the mask to the weights: keep lower triangle as initialized, zero upper triangle
        with torch.no_grad():
            self.weight.copy_(self.weight * causal_mask)

        # Compute the normalized mask and register it as a buffer
        mask_normalized = causal_mask / causal_mask.sum(dim=1, keepdim=True)
        self.register_buffer("mask_normalized", mask_normalized)

        # Register the hook to zero gradients outside the mask
        self.weight.register_hook(lambda grad: grad * self.mask_normalized)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


# class PraxisNano(nn.Module):
#     """
#     A special kind of block that omits the self-attention mechanism, in favor
#     of dense layers with sine activations. Inspired by NanoFFT:
#     https://github.com/timurgepard/nanoFFT
#     """

#     def __init__(self, config: "AutoConfig", chunk_size: int = 64, *args, **kwargs):
#         super().__init__()
#         self.chunk_size = chunk_size
#         self.stride = chunk_size // 2

#         embed_dim = config.num_embeds
#         hidden_dim = config.num_dims

#         # Define the weight matrices with chunk_size
#         self.fft_norm = nn.LayerNorm(hidden_dim)
#         self.fft = nn.Sequential(
#             TriLinear(chunk_size, int(chunk_size * 0.75)),
#             TriLinear(int(chunk_size * 0.75), chunk_size),
#         )

#         # Feed-forward network with sine activation
#         config.activation = "sin"
#         self.ffw_norm = nn.LayerNorm(hidden_dim)
#         self.ffw = PraxisGLU(config)

#     def forward(
#         self,
#         x: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         router_weights: Optional[Tensor] = None,
#         token_indices: Optional[Tensor] = None,
#     ):
#         B, T, E = x.shape
#         chunk_size, stride = self.chunk_size, self.stride
#         output = torch.zeros_like(x)
#         previous_overlap = None

#         for start in range(0, T, stride):
#             end = start + chunk_size
#             chunk = x[:, start:end, :]  # [B, chunk_size, E]

#             # Handle the last chunk which might be smaller than chunk_size
#             if chunk.size(1) < chunk_size:
#                 padding = chunk_size - chunk.size(1)
#                 chunk = F.pad(chunk, (0, 0, 0, padding), "constant", 0)

#             if previous_overlap is not None:
#                 # Replace the first 'stride' tokens with the previous processed overlap
#                 chunk = torch.cat([previous_overlap, chunk[:, stride:, :]], dim=1)

#             # Apply first layer normalization
#             chunk_norm = self.fft_norm(chunk)  # [B, chunk_size, E]

#             # Reshape chunk for FFT layers
#             chunk_fft = chunk_norm.transpose(1, 2)  # [B, E, T_chunk]

#             # Apply FFT layers
#             chunk_fft = self.fft(chunk_fft)  # [B, E, T_chunk]

#             # Reshape back to original dimensions
#             chunk_fft = chunk_fft.transpose(1, 2)  # [B, T_chunk, E]

#             # First residual connection
#             scale = 0.5
#             residual = (chunk_fft + chunk) * scale  # [B, chunk_size, E]

#             # Apply second layer normalization
#             chunk_norm_ffw = self.ffw_norm(residual)  # [B, chunk_size, E]

#             # Apply feed-forward network
#             chunk_ffw = self.ffw(chunk_norm_ffw)  # [B, chunk_size, E]

#             # Second residual connection
#             chunk = (chunk_ffw + residual) * scale  # [B, chunk_size, E]

#             # If the chunk was padded, remove the padding
#             if end > T:
#                 chunk = chunk[:, : T - start, :]  # [B, actual_seq_len, E]

#             # Accumulate the output
#             seq_len = chunk.size(1)
#             output[:, start : start + seq_len, :] += chunk

#             # Update previous_overlap with the current chunk's overlapping region
#             if stride > 0 and chunk.size(1) >= stride:
#                 previous_overlap = chunk[:, -stride:, :]  # [B, stride, E]
#             else:
#                 previous_overlap = None

#         return output


# class PraxisNano(nn.Module):
#     """
#     A special kind of block that omits the self-attention mechanism, in favor
#     of dense layers with sine activations. Inspired by NanoFFT:
#     https://github.com/timurgepard/nanoFFT
#     """

#     def __init__(
#         self,
#         config: "AutoConfig",
#         chunk_size: int = 64,
#         stride: int = 32,
#     ):
#         super().__init__()
#         self.chunk_size = chunk_size
#         self.stride = stride

#         embed_dim = config.num_embeds
#         hidden_dim = config.num_dims

#         # Define the weight matrices with chunk_size
#         self.ln1 = nn.LayerNorm(hidden_dim)
#         self.fft = nn.Sequential(
#             TriLinear(chunk_size, int(chunk_size * 0.75)),
#             TriLinear(int(chunk_size * 0.75), chunk_size),
#         )

#         # Feed-forward network with sine activation
#         self.ln2 = nn.LayerNorm(hidden_dim)
#         config.activation = "sin"
#         self.ffw = PraxisGLU(config)

#     def forward(
#         self,
#         x: Tensor,
#         attention_mask: Optional[Tensor] = None,
#     ):
#         B, T, E = x.shape
#         chunk_size = self.chunk_size
#         stride = self.stride

#         # Calculate number of chunks, ensuring at least one chunk
#         num_chunks = (T + stride - 1) // stride
#         num_chunks = max(num_chunks, 1)

#         # Calculate the padded length
#         T_padded = stride * (num_chunks - 1) + chunk_size
#         pad_amount = T_padded - T

#         if pad_amount > 0:
#             # Pad the sequence at the end with zeros
#             x_padded = F.pad(x, (0, 0, 0, pad_amount), "constant", 0)
#         else:
#             x_padded = x

#         # Permute to [B, E, T_padded]
#         x_permuted = x_padded.permute(0, 2, 1)  # [B, E, T_padded]

#         # Use unfold to extract sliding chunks
#         # Each chunk will have size `chunk_size` and step `stride`
#         chunks = x_permuted.unfold(
#             dimension=2, size=chunk_size, step=stride
#         )  # [B, E, num_chunks, chunk_size]

#         # Reshape to [B * num_chunks, E, chunk_size]
#         B_num = B * num_chunks
#         chunks = chunks.contiguous().view(B_num, E, chunk_size)  # [B*num_chunks, E, C]

#         # Permute to [B*num_chunks, C, E] for LayerNorm
#         chunks = chunks.permute(0, 2, 1)  # [B*num_chunks, C, E]

#         # Create the residual tensor
#         residual = chunks

#         # Apply LayerNorm
#         chunks_norm = self.ln1(chunks)  # [B*num_chunks, C, E]

#         # Permute to [B*num_chunks, E, C] for TriLinear
#         chunks_fft_input = chunks_norm.permute(0, 2, 1)  # [B*num_chunks, E, C]

#         # Apply fft (TriLinear layers)
#         chunks_fft = self.fft(chunks_fft_input)  # [B*num_chunks, E, C]

#         # Permute back to [B*num_chunks, C, E]
#         chunks_fft = chunks_fft.permute(0, 2, 1)  # [B*num_chunks, C, E]

#         # Residual connections
#         residual = chunks_fft + residual  # [B*num_chunks, C, E]

#         # Apply second LayerNorm
#         chunks_ln2 = self.ln2(residual)  # [B*num_chunks, C, E]

#         # Apply feed-forward network
#         chunks_ffw = self.ffw(chunks_ln2)  # [B*num_chunks, C, E]

#         # Another residual connection
#         chunks_final = chunks_ffw + residual  # [B*num_chunks, C, E]

#         # Reshape back to [B, num_chunks, C, E]
#         chunks_final = chunks_final.view(
#             B, num_chunks, chunk_size, E
#         )  # [B, num_chunks, C, E]

#         # Zero out padded positions in the last chunk
#         if pad_amount > 0:
#             valid_length = T - stride * (num_chunks - 1)
#             if valid_length > 0:
#                 chunks_final[:, -1, valid_length:, :] = 0  # Zero out padded positions

#         # Reshape to [B, num_chunks * C, E]
#         chunks_final = chunks_final.view(
#             B, num_chunks * chunk_size, E
#         )  # [B, num_chunks*C, E]

#         # Create a mask to identify valid positions (1) and padded positions (0)
#         mask = torch.ones_like(chunks_final, dtype=x.dtype, device=x.device)
#         if pad_amount > 0:
#             valid_length = T - stride * (num_chunks - 1)
#             if valid_length > 0:
#                 mask[:, -pad_amount:, :] = 0  # Zero out padded positions

#         # Generate position indices for each element
#         # Generate a range [0, chunk_size) and add stride * chunk_idx
#         chunk_range = (
#             torch.arange(chunk_size, device=x.device).unsqueeze(0).unsqueeze(0)
#         )  # [1,1,C]
#         chunk_indices = (
#             torch.arange(num_chunks, device=x.device).unsqueeze(0).unsqueeze(2)
#         )  # [1,num_chunks,1]
#         positions = chunk_range + (chunk_indices * stride)  # [1, num_chunks, C]
#         positions = positions.expand(B, -1, -1)  # [B, num_chunks, C]
#         positions = positions.contiguous().view(B, -1)  # [B, num_chunks*C]

#         # Clamp positions to [0, T-1]
#         positions = positions.clamp(max=T - 1).long()  # [B, num_chunks*C]

#         # Expand positions to match E
#         positions = positions.unsqueeze(-1).expand(-1, -1, E)  # [B, num_chunks*C, E]

#         # Initialize output and overlap counts
#         output = torch.zeros(B, T, E, device=x.device, dtype=x.dtype)
#         overlap_counts = torch.zeros(B, T, E, device=x.device, dtype=x.dtype)

#         # Scatter add
#         output.scatter_add_(1, positions, chunks_final * mask)
#         overlap_counts.scatter_add_(1, positions, mask)

#         # Avoid division by zero
#         overlap_counts = torch.clamp(overlap_counts, min=1.0)

#         # Average the overlapping regions
#         output = output / overlap_counts

#         return output


# class PraxisNano(nn.Module):
#     """
#     A special kind of block that omits the self-attention mechanism, in favor
#     of dense layers with sine activations. Inspired by NanoFFT:
#     https://github.com/timurgepard/nanoFFT
#     """

#     def __init__(self, config: "AutoConfig", chunk_size: int = 64):
#         super().__init__()
#         self.chunk_size = chunk_size
#         self.stride = chunk_size // 2

#         embed_dim = config.num_embeds
#         hidden_dim = config.num_dims

#         # Define the weight matrices with chunk_size
#         self.ln1 = nn.LayerNorm(hidden_dim)
#         self.fft = nn.Sequential(
#             TriLinear(chunk_size, int(chunk_size * 0.75)),
#             TriLinear(int(chunk_size * 0.75), chunk_size),
#         )

#         # Feed-forward network with sine activation
#         config.activation = "sin"
#         self.ln2 = nn.LayerNorm(hidden_dim)
#         self.ffw = PraxisGLU(config)

#     def forward(
#         self,
#         x: Tensor,
#         attention_mask: Optional[Tensor] = None,
#         router_weights: Optional[Tensor] = None,
#         token_indices: Optional[Tensor] = None,
#     ):
#         B, T, E = x.shape
#         chunk_size = self.chunk_size
#         stride = self.stride

#         # Initialize the output tensor and a tensor to keep track of overlap counts
#         device = x.device
#         output = torch.zeros_like(x)
#         overlap_counts = torch.zeros(B, T, E, device=device)

#         # Iterate over the sequence in chunks
#         for start in range(0, T, stride):
#             end = start + chunk_size
#             chunk = x[:, start:end, :]

#             # Handle the last chunk which might be smaller than chunk_size
#             if chunk.size(1) < chunk_size:
#                 padding = chunk_size - chunk.size(1)
#                 chunk = nn.functional.pad(chunk, (0, 0, 0, padding), "constant", 0)

#             # save residual and apply layer norm
#             residual = chunk
#             chunk_norm = self.ln1(chunk)

#             # Reshape chunk for matrix multiplication
#             chunk_fft = chunk_norm.transpose(1, 2)  # [B, embed_dim, T_chunk]

#             # Apply the masked and normalized weight matrices
#             chunk_fft = self.fft(chunk_fft)

#             # Reshape back to original dimensions
#             chunk_fft = chunk_fft.transpose(1, 2)  # [B, T_chunk, E]

#             # Residual connection
#             chunk = chunk_fft + residual

#             # Apply second layer norm and feed-forward network
#             residual = chunk
#             chunk = self.ln2(chunk)
#             chunk = self.ffw(chunk) + residual

#             # If the chunk was padded, remove the padding
#             if end > T:
#                 chunk = chunk[:, : T - start, :]

#             # Accumulate the output and overlap counts
#             seq_len = chunk.size(1)
#             output[:, start : start + seq_len, :] += chunk
#             overlap_counts[:, start : start + seq_len, :] += 1

#         # Avoid division by zero
#         overlap_counts = torch.clamp(overlap_counts, min=1.0)

#         # Average the overlapping regions
#         output = output / overlap_counts

#         return output


if __name__ == "__main__":
    # Mock AutoConfig class to simulate the configuration
    @dataclass
    class AutoConfig:
        num_dims: int = 768
        num_embeds: int = 768
        context_length: int = 2048
        vocab_size: int = 50257
        causal: bool = True
        dropout: float = 0.0

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig()
    chunk_size = 256  # Explicitly define for tests
    stride = 128  # Example stride with overlap

    def run_memory_test(model, x):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Forward pass
        start_time = time.time()
        with torch.cuda.amp.autocast():
            output = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        max_memory = (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        )  # Convert to MB
        return output, end_time - start_time, max_memory

    print("Running tests for PraxisNano...")

    # Create model once for all tests
    model = PraxisNano(config, chunk_size=chunk_size, stride=stride).to(device)

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
        model_test = PraxisNano(config, chunk_size=cs, stride=current_stride).to(device)
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

    print("\nAll tests completed!")
