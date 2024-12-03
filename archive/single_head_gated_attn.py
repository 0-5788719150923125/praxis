import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadGatedAttention(nn.Module):
    """
    According to MEGA, "Single-head gated attention has been empirically
    shown [to be] as performant as vanilla multi-head attention."
    https://arxiv.org/abs/2209.10655
    """

    def __init__(self, embed_dim, output_dim=None, gate_hidden_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Q, K, V projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if output_dim is None:
            output_dim = embed_dim
        self.output = nn.Linear(embed_dim, output_dim)

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
        scores, v = self.compute_scores(query_input, key_input, value_input)
        out = self.compute_weights(scores, v)

        # Generate and apply gates
        gates = self.gate_net(query_input)  # [B, S, E]
        out = out * gates

        return self.output(out)

    def compute_scores(self, query_input, key_input, value_input):
        B, S, E = query_input.shape

        # Project inputs
        q = self.query(query_input)  # [B, S, E]
        k = self.key(key_input)  # [num_layers, E]
        v = self.value(value_input)  # [num_layers, E]

        # Expand k and v to match batch size
        k = k.unsqueeze(0).expand(B, -1, E)  # [B, num_layers, E]
        v = v.unsqueeze(0).expand(B, -1, E)  # [B, num_layers, E]

        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (
            math.sqrt(embed_dim)
        )  # [B, S, num_layers]

        return scores, v

    def compute_weights(self, scores, v):
        attn = F.softmax(scores, dim=-1)  # [B, S, num_layers]
        attn = self.dropout(attn)
        # Apply attention to values
        out = torch.bmm(attn, v)  # [B, S, E]
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

        # Test gradient flow
        loss = out.sum()
        loss.backward()
        print("✓ Gradient computation")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise
