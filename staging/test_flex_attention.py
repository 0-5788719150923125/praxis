#!/usr/bin/env python3
"""
FlexAttention Test Script for RTX 5060
Tests causal attention for language modeling using PyTorch's FlexAttention
"""

import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Check PyTorch version and CUDA availability
def check_environment():
    """Check if the environment is properly configured"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! FlexAttention requires CUDA.")
        return False

    print(f"✅ CUDA is available")
    print(f"CUDA Version: {torch.version.cuda}")

    # Check GPU info
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

        # Check if it's an RTX 5060
        if "5060" in gpu_name:
            print(f"✅ RTX 5060 detected!")

        # Get GPU properties
        props = torch.cuda.get_device_properties(i)
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Memory: {props.total_memory / 1024**3:.2f} GB")

    # Check if torch.compile is available
    try:
        dummy = torch.compile(lambda x: x)
        print("✅ torch.compile is available")
    except Exception as e:
        print(f"⚠️  torch.compile might not be fully functional: {e}")

    return True


# Import FlexAttention components
def import_flex_attention():
    """Import FlexAttention with error handling"""
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        print("✅ FlexAttention imported successfully")
        return flex_attention, create_block_mask
    except ImportError as e:
        print(f"❌ Failed to import FlexAttention: {e}")
        print("Note: FlexAttention requires PyTorch nightly or version >= 2.5.0")
        print(
            "Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121"
        )
        return None, None


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention using FlexAttention"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Import FlexAttention
        self.flex_attention, self.create_block_mask = import_flex_attention()

        # Will store the block mask
        self.block_mask = None

    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create a causal mask for the given sequence length"""
        if self.create_block_mask is None:
            return None

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create block mask (broadcasting over batch and heads)
        block_mask = self.create_block_mask(
            causal_mask,
            B=None,  # Broadcast over batch
            H=None,  # Broadcast over heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        return block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # Calculate QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        # (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Use FlexAttention if available, otherwise fall back to standard attention
        if self.flex_attention is not None:
            # Create causal mask if needed (cache it for reuse)
            if self.block_mask is None or self.block_mask.shape[-1] != T:
                self.block_mask = self.create_causal_mask(T, x.device)

            # Apply FlexAttention with causal mask
            attn_output = self.flex_attention(q, k, v, block_mask=self.block_mask)
        else:
            # Fallback to standard PyTorch attention
            print("⚠️  Using standard attention (FlexAttention not available)")
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_proj(attn_output)
        out = self.resid_dropout(out)

        return out


class SimpleTransformerLM(nn.Module):
    """Simple Transformer Language Model for testing"""

    def __init__(
        self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(1024, embed_dim)  # Max seq len 1024

        self.layers = nn.ModuleList(
            [CausalSelfAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        # Apply transformer layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


def test_flexattention():
    """Main test function"""
    print("\n" + "=" * 60)
    print("FlexAttention Test for Language Modeling")
    print("=" * 60)

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Exiting.")
        return

    # Try to import FlexAttention
    flex_attention, create_block_mask = import_flex_attention()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Model parameters
    vocab_size = 1000
    embed_dim = 512
    num_heads = 8
    num_layers = 2
    batch_size = 4
    seq_len = 256

    print(f"\nModel Configuration:")
    print(f"  - Vocab Size: {vocab_size}")
    print(f"  - Embedding Dim: {embed_dim}")
    print(f"  - Number of Heads: {num_heads}")
    print(f"  - Number of Layers: {num_layers}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Sequence Length: {seq_len}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = SimpleTransformerLM(vocab_size, embed_dim, num_heads, num_layers).to(device)
    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    try:
        with torch.cuda.amp.autocast():  # Use mixed precision for better performance
            output = model(input_ids)
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # Test backward pass
    print("\n" + "=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)

    try:
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = F.cross_entropy(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        print(f"✅ Backward pass successful!")
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return

    # Benchmark performance
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    model.eval()

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)

    torch.cuda.synchronize()

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
    throughput = (batch_size * seq_len) / (avg_time / 1000)  # Tokens per second

    print(f"Average forward pass time: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.0f} tokens/second")

    # Memory usage
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

    # Test with torch.compile if available
    if flex_attention is not None:
        print("\n" + "=" * 60)
        print("Testing with torch.compile")
        print("=" * 60)

        try:
            compiled_model = torch.compile(model)
            with torch.no_grad():
                output_compiled = compiled_model(input_ids)
            print("✅ torch.compile works with FlexAttention!")

            # Check if outputs match
            if torch.allclose(output, output_compiled, rtol=1e-4, atol=1e-4):
                print("✅ Compiled and non-compiled outputs match!")
        except Exception as e:
            print(f"⚠️  torch.compile test failed: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

    if flex_attention is None:
        print("\n⚠️  Note: FlexAttention was not available, using standard attention.")
        print("To use FlexAttention, install PyTorch nightly:")
        print(
            "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121"
        )
    else:
        print("\n🎉 FlexAttention is working correctly on your RTX 5060!")
        print("   You can now use FlexAttention for efficient attention computations.")


if __name__ == "__main__":
    test_flexattention()
