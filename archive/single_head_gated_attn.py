import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadGatedAttention(nn.Module):
    """
    According to MEGA, "Single-head gated attention has been empirically
    shown [to be] as performant as vanilla multi-head attention."
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, embed_dim, gate_hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Gate generator G(X)
        if gate_hidden_dim is None:
            gate_hidden_dim = embed_dim * 4

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, query_input, key_input, value_input, mask=None):
        """
        Args:
            query_input: Tensor of shape (batch_size, seq_len, embed_dim)
            key_input: Tensor of shape (num_layers, embed_dim)
            value_input: Tensor of shape (num_layers, embed_dim)
            mask: Optional attention mask
        """
        B, S, E = query_input.shape
        num_layers = key_input.shape[0]

        # Project inputs
        q = self.q_proj(query_input)  # [B, S, E]
        k = self.k_proj(key_input)  # [num_layers, E]
        v = self.v_proj(value_input)  # [num_layers, E]

        # Expand k and v to match batch size
        k = k.unsqueeze(0).expand(B, -1, -1)  # [B, num_layers, E]
        v = v.unsqueeze(0).expand(B, -1, -1)  # [B, num_layers, E]

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (
            self.embed_dim**0.5
        )  # [B, S, num_layers]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B, S, num_layers]

        # Apply attention to values
        out = torch.bmm(attn, v)  # [B, S, E]

        # Generate and apply gates
        gates = self.gate_net(query_input)  # [B, S, E]
        out = out * gates

        return out


if __name__ == "__main__":
    # Test settings
    batch_size = 4
    seq_length = 10
    num_layers = 6
    embed_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on {device}")

    # Initialize model
    model = SingleHeadGatedAttention(embed_dim).to(device)

    # Create random inputs
    query_input = torch.randn(batch_size, seq_length, embed_dim).to(device)
    key_input = torch.randn(num_layers, embed_dim).to(device)
    value_input = torch.randn(num_layers, embed_dim).to(device)

    try:
        # Test forward pass
        out = model(query_input, key_input, value_input)
        expected_shape = (batch_size, seq_length, embed_dim)
        assert (
            out.shape == expected_shape
        ), f"Output shape mismatch: got {out.shape}, expected {expected_shape}"
        print("✓ Basic forward pass")

        # Test with attention mask
        mask = torch.ones(batch_size, seq_length, num_layers).to(device)
        mask[:, :, num_layers // 2 :] = 0  # Mask out second half of layers
        out_masked = model(query_input, key_input, value_input, mask)
        assert out_masked.shape == expected_shape, "Masked output shape mismatch"
        print("✓ Forward pass with mask")

        # Test that masked and unmasked outputs are different
        assert not torch.allclose(
            out, out_masked
        ), "Masked and unmasked outputs should differ"
        print("✓ Mask affects output")

        # Test gradient flow
        loss = out.sum()
        loss.backward()
        print("✓ Gradient computation")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise
