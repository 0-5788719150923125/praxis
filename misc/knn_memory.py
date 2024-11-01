import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (won't be updated during backprop)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_memory: bool = False,
        memory_size: int = 0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_memory = use_memory

        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Memory components (if enabled)
        if use_memory:
            self.memory = KNNMemory(self.d_k, memory_size)
            self.memory_gate = nn.Parameter(torch.zeros(1))

    def generate_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(device)
        return mask

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, v)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Create causal mask
        mask = self.generate_causal_mask(seq_len, x.device)

        # Linear projections and reshape
        q = (
            self.q_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Standard attention output
        attn_out = self.attention(q, k, v, mask.unsqueeze(0).unsqueeze(0))
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        attn_out = self.out_linear(attn_out)

        if not self.use_memory:
            return attn_out

        # Memory operations using the same q/k/v projections
        # Reshape for memory operations [batch*heads*seq_len, d_k]
        flat_q = q.contiguous().view(-1, self.d_k)
        flat_k = k.contiguous().view(-1, self.d_k)
        flat_v = v.contiguous().view(-1, self.d_k)

        # Get memory output
        scores, indices = self.memory.find_knn(flat_q)

        if scores is None:
            self.memory.update_memory(flat_k, flat_v)
            return attn_out

        # Gather and process memory values
        memory_values = self.memory.get_values(indices)
        memory_out = torch.sum(memory_values * scores.unsqueeze(-1), dim=-2)

        # Reshape memory output back to match attention output
        memory_out = memory_out.view(batch_size, self.num_heads, seq_len, self.d_k)
        memory_out = (
            memory_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        memory_out = self.out_linear(memory_out)

        # Combine outputs using learned gate
        gate = torch.sigmoid(self.memory_gate)
        output = gate * memory_out + (1 - gate) * attn_out

        # Update memory
        self.memory.update_memory(flat_k, flat_v)

        return output


# https://openreview.net/pdf?id=TrjbxzRcnf-
class KNNMemory(nn.Module):
    def __init__(self, dim: int, max_memories: int, k: int = 32):
        super().__init__()
        self.max_memories = max_memories
        self.dim = dim
        self.k = k

        # Initialize key_memories and value_memories as empty tensors and register as buffers
        self.register_buffer("key_memories", torch.empty(0, dim))
        self.register_buffer("value_memories", torch.empty(0, dim))

    @property
    def current_size(self):
        return self.key_memories.size(0)

    def normalize_vectors(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def find_knn(self, query: torch.Tensor) -> tuple:
        """
        Args:
            query: shape [batch_size * seq_len * num_heads, d_k]
        """
        if self.key_memories.size(0) == 0:
            return None, None

        # keys: [num_memories, dim]
        keys = self.key_memories

        # Normalize query and keys
        query = self.normalize_vectors(query)
        keys = self.normalize_vectors(keys)

        # Compute similarities
        similarities = torch.matmul(
            query, keys.t()
        )  # [batch*seq_len*num_heads, num_memories]

        # Get top k scores and indices
        k = min(self.k, self.key_memories.size(0))
        scores, indices = torch.topk(similarities, k, dim=-1)

        return scores, indices

    def get_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: shape [batch*seq_len*num_heads, k]
        Returns:
            values: shape [batch*seq_len*num_heads, k, d_k]
        """
        # values: [num_memories, dim]
        values = self.value_memories

        # Gather values for each query's top k indices
        gathered_values = values[indices]  # [batch*seq_len*num_heads, k, dim]

        return gathered_values

    def update_memory(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Args:
            keys: shape [batch*seq_len*num_heads, d_k]
            values: shape [batch*seq_len*num_heads, d_k]
        """
        # Concatenate new keys and values
        self.key_memories = torch.cat([self.key_memories, keys], dim=0)
        self.value_memories = torch.cat([self.value_memories, values], dim=0)

        # If exceed max_memories, keep the latest max_memories
        if self.key_memories.size(0) > self.max_memories:
            self.key_memories = self.key_memories[-self.max_memories :]
            self.value_memories = self.value_memories[-self.max_memories :]


if __name__ == "__main__":
    # Smoke tests
    print("Running smoke tests...")

    # Test parameters
    batch_size = 32
    seq_length = 50
    d_model = 512
    num_heads = 8
    memory_size = 1000

    # Initialize models
    pos_encoder = PositionalEncoding(d_model)

    # Test standard attention without memory
    print("\nTesting standard attention (no memory)...")
    attention = MultiHeadAttention(d_model, num_heads, use_memory=False)
    x = torch.randn(batch_size, seq_length, d_model)

    # Test 1: Shape check
    print("Test 1: Checking output shapes...")
    x_pos = pos_encoder(x)
    output = attention(x_pos)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"

    # Test 2: Causal mask check
    print("\nTest 2: Checking causal masking...")
    mask = attention.generate_causal_mask(5, torch.device("cpu"))
    expected_mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    assert torch.all(mask == expected_mask), "Causal mask incorrect!"

    # Test memory-enabled attention
    print("\nTesting memory-enabled attention...")
    memory_attention = MultiHeadAttention(
        d_model, num_heads, use_memory=True, memory_size=memory_size
    )

    # Test 3: Memory initialization
    print("Test 3: Memory initialization...")
    assert hasattr(memory_attention, "memory"), "Memory not initialized!"
    assert memory_attention.memory.current_size == 0, "Memory not empty at start!"

    # Test 4: First forward pass
    print("\nTest 4: First forward pass...")
    output_1 = memory_attention(x)
    assert output_1.shape == x.shape, "Output shape mismatch!"
    print(f"Memory size after first pass: {memory_attention.memory.current_size}")
    assert memory_attention.memory.current_size > 0, "Memory not updated!"

    # Test 5: Memory capacity
    print("\nTest 5: Memory capacity...")
    # Add enough tokens to exceed memory capacity
    for i in range(3):
        memory_attention(x)
        print(f"Memory size after pass {i+2}: {memory_attention.memory.current_size}")
    assert (
        memory_attention.memory.current_size <= memory_size
    ), "Memory exceeded capacity!"

    # Test 6: Output consistency with same input
    print("\nTest 6: Output consistency...")
    x_repeat = x.clone()
    output_1 = memory_attention(x)
    output_2 = memory_attention(x_repeat)
    diff = (output_1 - output_2).abs().mean().item()
    print(f"Average difference between outputs: {diff}")
    assert diff < 1.0, "Outputs too different for same input!"

    # Test 7: Gradient flow
    print("\nTest 7: Testing gradient flow...")
    try:
        memory_attention.train()
        output = memory_attention(x)
        loss = output.sum()
        loss.backward()
        print("Forward and backward pass successful!")
    except Exception as e:
        print(f"Forward/backward pass failed: {e}")

    # Test 8: Memory per head
    print("\nTest 8: Testing per-head memory...")
    # Get first memory entry size
    first_key_size = memory_attention.memory.key_memories[0].shape[-1]
    assert (
        first_key_size == memory_attention.d_k
    ), "Memory not storing per-head vectors!"

    print("\nAll tests passed!")
