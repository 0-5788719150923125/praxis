import math

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
        return x + self.pe[: x.size(1), :]


class KNNMemory(nn.Module):
    def __init__(self, num_heads: int, dim: int, max_memories: int, k: int = 32):
        """
        Initializes the KNNMemory module.

        Args:
            num_heads (int): Number of attention heads.
            dim (int): Dimension of each head.
            max_memories (int): Maximum number of memories to store per head.
            k (int): Number of nearest neighbors to retrieve.
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.k = k
        self.max_memories = max_memories

        # Initialize key_memories and value_memories for each head
        # Shape: [num_heads, max_memories, dim]
        self.register_buffer("key_memories", torch.empty(num_heads, 0, dim))
        self.register_buffer("value_memories", torch.empty(num_heads, 0, dim))

    @property
    def current_size(self):
        """
        Returns the current number of memories stored per head.

        Returns:
            int: Number of memories per head.
        """
        return self.key_memories.size(1)

    def normalize_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes vectors to unit length.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim].

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

    def find_knn(self, queries: torch.Tensor) -> tuple:
        """
        Finds the k-nearest neighbors for each query across all heads.

        Args:
            queries (torch.Tensor): Queries of shape [num_heads, Q, dim].

        Returns:
            tuple: (scores, indices)
                scores (torch.Tensor): Similarity scores of shape [num_heads, Q, k].
                indices (torch.Tensor): Indices of nearest neighbors of shape [num_heads, Q, k].
        """
        if self.key_memories.size(1) == 0:
            return None, None

        # Normalize queries and keys
        queries_norm = self.normalize_vectors(queries)  # [num_heads, Q, dim]
        keys_norm = self.normalize_vectors(self.key_memories)  # [num_heads, K, dim]

        # Compute cosine similarity: [num_heads, Q, K]
        similarities = torch.bmm(queries_norm, keys_norm.transpose(1, 2)) / math.sqrt(
            self.dim
        )

        # Get top-k similarities and their indices
        k = min(self.k, self.key_memories.size(1))
        scores, indices = similarities.topk(k, dim=-1)  # [num_heads, Q, k]

        return scores, indices

    def get_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the values corresponding to the nearest neighbors.

        Args:
            indices (torch.Tensor): Indices of nearest neighbors of shape [num_heads, Q, k].

        Returns:
            torch.Tensor: Retrieved values of shape [num_heads, Q, k, dim].
        """
        # Gather values for each head
        # Using gather requires indices to have the same number of dimensions as value_memories
        # Expand dimensions to gather across the memory dimension (dim=1)
        # First, ensure indices are of type long
        indices = indices.long()

        # Expand indices to [num_heads, Q, k, dim]
        indices_expanded = indices.unsqueeze(-1).expand(
            -1, -1, -1, self.dim
        )  # [num_heads, Q, k, dim]

        # Gather values
        gathered_values = torch.gather(
            self.value_memories.unsqueeze(1).expand(
                -1, indices.size(1), -1, -1
            ),  # [num_heads, Q, K, dim]
            2,
            indices_expanded,  # [num_heads, Q, k, dim]
        )  # [num_heads, Q, k, dim]
        return gathered_values

    def update_memory(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Updates the memory with new keys and values.

        Args:
            keys (torch.Tensor): New keys of shape [num_heads, Q, dim].
            values (torch.Tensor): New values of shape [num_heads, Q, dim].
        """
        # Concatenate new keys and values
        self.key_memories = torch.cat(
            [self.key_memories, keys], dim=1
        )  # [num_heads, new_total_K, dim]
        self.value_memories = torch.cat(
            [self.value_memories, values], dim=1
        )  # [num_heads, new_total_K, dim]

        # Trim memory if exceeding max_memories
        if self.key_memories.size(1) > self.max_memories:
            excess = self.key_memories.size(1) - self.max_memories
            self.key_memories = self.key_memories[:, excess:, :]
            self.value_memories = self.value_memories[:, excess:, :]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_memory: bool = False,
        memory_size: int = 0,
        k: int = 32,
    ):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): Total dimension of the model.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            use_memory (bool): Whether to use external memory.
            memory_size (int): Maximum memory size per head.
            k (int): Number of nearest neighbors to retrieve.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_memory = use_memory
        self.k = k

        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Memory components (if enabled)
        if use_memory:
            self.memory = KNNMemory(num_heads, self.d_k, memory_size, k)
            self.memory_gate = nn.Parameter(torch.zeros(num_heads))  # One gate per head

    def generate_causal_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """
        Generates a causal mask for the attention.

        Args:
            batch_size (int): Number of sequences in the batch.
            seq_len (int): Sequence length.
            device (torch.device): Device to place the mask on.

        Returns:
            torch.Tensor: Mask of shape [num_heads * batch_size, seq_len, seq_len]
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool()  # [seq_len, seq_len]
        mask = mask.unsqueeze(0).repeat(
            self.num_heads * batch_size, 1, 1
        )  # [num_heads * batch_size, seq_len, seq_len]
        mask = mask.to(device)
        return mask

    def attention(self, q, k, v, mask=None):
        """
        Computes scaled dot-product attention.

        Args:
            q (torch.Tensor): Queries of shape [batch_heads, Q, d_k].
            k (torch.Tensor): Keys of shape [batch_heads, K, d_k].
            v (torch.Tensor): Values of shape [batch_heads, K, d_k].
            mask (torch.Tensor, optional): Mask of shape [batch_heads, Q, K].

        Returns:
            torch.Tensor: Attention output of shape [batch_heads, Q, d_k].
        """
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(
            self.d_k
        )  # [batch_heads, Q, K]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [batch_heads, Q, K]
        attn = self.dropout(attn)

        attn_out = torch.bmm(attn, v)  # [batch_heads, Q, d_k]
        return attn_out

    def forward(self, x):
        """
        Forward pass for MultiHeadAttention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, d_model = x.size()

        # Create causal mask
        mask = self.generate_causal_mask(
            batch_size, seq_len, x.device
        )  # [num_heads * batch_size, seq_len, seq_len]

        # Linear projections
        q = self.q_linear(x)  # [batch_size, seq_len, d_model]
        k = self.k_linear(x)  # [batch_size, seq_len, d_model]
        v = self.v_linear(x)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head: [batch_size, num_heads, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len, d_k]
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Reshape to [num_heads * batch_size, seq_len, d_k]
        q = q.contiguous().view(self.num_heads * batch_size, seq_len, self.d_k)
        k = k.contiguous().view(self.num_heads * batch_size, seq_len, self.d_k)
        v = v.contiguous().view(self.num_heads * batch_size, seq_len, self.d_k)

        # Compute standard attention
        attn_out = self.attention(
            q, k, v, mask
        )  # [num_heads * batch_size, seq_len, d_k]

        # Reshape back to [batch_size, num_heads, seq_len, d_k]
        attn_out = (
            attn_out.view(self.num_heads, batch_size, seq_len, self.d_k)
            .permute(1, 0, 2, 3)
            .contiguous()
        )  # [batch_size, num_heads, seq_len, d_k]

        if not self.use_memory:
            # Combine heads: [batch_size, seq_len, num_heads * d_k]
            attn_out_combined = (
                attn_out.contiguous()
                .view(batch_size, self.num_heads * self.d_k, seq_len)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, seq_len, d_model]
            attn_out_combined = self.out_linear(
                attn_out_combined
            )  # [batch_size, seq_len, d_model]
            return attn_out_combined

        # Memory operations
        # Reshape q, k, v to [num_heads, batch_size * seq_len, d_k]
        q_mem = (
            q.view(self.num_heads, batch_size, seq_len, self.d_k)
            .transpose(1, 0)
            .contiguous()
            .view(self.num_heads, batch_size * seq_len, self.d_k)
        )  # [num_heads, Q, d_k]
        k_mem = (
            k.view(self.num_heads, batch_size, seq_len, self.d_k)
            .transpose(1, 0)
            .contiguous()
            .view(self.num_heads, batch_size * seq_len, self.d_k)
        )  # [num_heads, Q, d_k]
        v_mem = (
            v.view(self.num_heads, batch_size, seq_len, self.d_k)
            .transpose(1, 0)
            .contiguous()
            .view(self.num_heads, batch_size * seq_len, self.d_k)
        )  # [num_heads, Q, d_k]

        # Find kNN
        scores_mem, indices_mem = self.memory.find_knn(
            q_mem
        )  # [num_heads, Q, k], [num_heads, Q, k]

        if scores_mem is not None:
            # Retrieve memory values
            memory_values = self.memory.get_values(
                indices_mem
            )  # [num_heads, Q, k, d_k]

            # Compute weighted sum: [num_heads, Q, d_k]
            weighted_memory = torch.sum(
                memory_values * scores_mem.unsqueeze(-1), dim=2
            )  # [num_heads, Q, d_k]

            # Reshape to [num_heads, batch_size, seq_len, d_k]
            weighted_memory = weighted_memory.view(
                self.num_heads, batch_size, seq_len, self.d_k
            )

            # Permute to [batch_size, num_heads, seq_len, d_k] to align with attn_out
            weighted_memory = weighted_memory.permute(
                1, 0, 2, 3
            ).contiguous()  # [batch_size, num_heads, seq_len, d_k]

            # Apply per-head gating
            gate = (
                torch.sigmoid(self.memory_gate)
                .view(1, self.num_heads, 1, 1)
                .to(x.device)
            )  # [1, num_heads, 1, 1]

            # Combine attention and memory outputs using the gate
            output = (
                gate * weighted_memory + (1 - gate) * attn_out
            )  # [batch_size, num_heads, seq_len, d_k]

            # Combine heads: [batch_size, seq_len, num_heads * d_k]
            output = (
                output.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size, seq_len, self.num_heads * self.d_k)
            )  # [batch_size, seq_len, d_model]
            output = self.out_linear(output)  # [batch_size, seq_len, d_model]
        else:
            # If no memory found, use standard attention output
            output = attn_out  # [batch_size, num_heads, seq_len, d_k]
            # Combine heads: [batch_size, seq_len, num_heads * d_k]
            output = (
                output.contiguous()
                .view(batch_size, self.num_heads * self.d_k, seq_len)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, seq_len, d_model]
            output = self.out_linear(output)  # [batch_size, seq_len, d_model]

        # Update memory with current keys and values
        self.memory.update_memory(k_mem, v_mem)

        return output


if __name__ == "__main__":
    # Smoke tests
    print("Running smoke tests...")

    # Test parameters
    batch_size = 2  # Reduced for quicker testing
    seq_length = 5
    d_model = 16
    num_heads = 4
    memory_size = 10
    k = 2  # Reduced k for testing

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
    print("Test 1 passed.")

    # Test 2: Causal mask check
    print("\nTest 2: Checking causal masking...")
    mask = attention.generate_causal_mask(batch_size, seq_length, torch.device("cpu"))
    expected_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    expected_mask = expected_mask.unsqueeze(0).repeat(num_heads * batch_size, 1, 1)
    assert torch.all(mask == expected_mask), "Causal mask incorrect!"
    print("Causal mask correct.")
    print("Test 2 passed.")

    # Test memory-enabled attention
    print("\nTesting memory-enabled attention...")
    memory_attention = MultiHeadAttention(
        d_model, num_heads, use_memory=True, memory_size=memory_size, k=k
    )

    # Test 3: Memory initialization
    print("Test 3: Memory initialization...")
    assert hasattr(memory_attention, "memory"), "Memory not initialized!"
    assert memory_attention.memory.current_size == 0, "Memory not empty at start!"
    print("Memory initialized correctly.")
    print("Test 3 passed.")

    # Test 4: First forward pass
    print("\nTest 4: First forward pass...")
    output_1 = memory_attention(x)
    assert output_1.shape == x.shape, "Output shape mismatch!"
    current_size = memory_attention.memory.current_size
    print(f"Output shape: {output_1.shape}")
    print(f"Memory size after first pass: {current_size}")
    assert (
        current_size == batch_size * seq_length
    ), "Memory not updated correctly after first pass!"
    print("Memory updated correctly after first pass.")
    print("Test 4 passed.")

    # Test 5: Memory capacity
    print("\nTest 5: Memory capacity...")
    # Add enough tokens to exceed memory capacity
    for i in range(3):
        output_i = memory_attention(x)
        current_size = memory_attention.memory.current_size
        print(f"Memory size after pass {i+2}: {current_size}")
        assert current_size <= memory_size, "Memory exceeded capacity!"
    print("Memory capacity enforced correctly.")
    print("Test 5 passed.")

    # Test 6: Output consistency with same input
    print("\nTest 6: Output consistency...")
    # Do NOT reset memory for this test

    # Forward pass 1
    output_1 = memory_attention(x)

    # Forward pass 2 with same input
    output_2 = memory_attention(x.clone())

    # Compare outputs
    diff = (output_1 - output_2).abs().mean().item()
    print(f"Average difference between outputs: {diff}")
    # Since memory is updated after first pass, the second pass should have different outputs
    assert diff > 0.0, "Outputs should differ after memory update!"
    print("Output consistency test passed.")
    print("Test 6 passed.")

    # Test 7: Gradient flow
    print("\nTest 7: Testing gradient flow...")
    memory_attention.train()
    output = memory_attention(x)
    loss = output.sum()
    try:
        loss.backward()
        print("Forward and backward pass successful!")
    except Exception as e:
        print(f"Forward/backward pass failed: {e}")
        assert False, "Gradient flow test failed!"
    print("Test 7 passed.")

    # Test 8: Memory per head
    print("\nTest 8: Testing per-head memory...")
    first_key_size = memory_attention.memory.key_memories[0].shape[-1]
    assert (
        first_key_size == memory_attention.d_k
    ), "Memory not storing per-head vectors!"
    print("Memory per head test passed.")
    print("Test 8 passed.")

    print("\nAll tests passed successfully!")
