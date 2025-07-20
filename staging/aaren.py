import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_scan_attention(scores, values, mask=None):
    """
    Numerically stable, vectorized implementation of attention using parallel scan.
    Based on paper's formulation of prefix computation.
    """
    batch_size, seq_len, num_heads, dim = values.shape

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)

    # Compute running maximum (paper's m_A)
    # Note: cummax returns (values, indices)
    m_A = torch.cummax(scores, dim=1)[0]

    # Compute exponential terms once, normalized by current max (stable)
    exp_scores = torch.exp(scores - m_A)

    # Compute u_A (denominator)
    u_A = exp_scores.cumsum(dim=1)

    # Compute w_A (numerator)
    w_A = (exp_scores.unsqueeze(-1) * values).cumsum(dim=1)

    # Final output is w_A / u_A
    output = w_A / u_A.unsqueeze(-1)

    return output


class AAREN(nn.Module):
    """
    From "Attention as a Recurrent Neural Network":
    https://arxiv.org/abs/2405.13956v2
    https://github.com/claCase/Attention-as-RNN/blob/main/src/layers.py
    """

    def __init__(
        self,
        heads: int,
        dim: int,
        dropout: float = 0.1,
        activation: str = "silu",
        concat_heads: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.concat_heads = concat_heads
        self.dropout = nn.Dropout(p=dropout)

        # Linear projections
        self.kv_proj = nn.Linear(dim, heads * dim * 2, bias=False)
        self.q_proj = nn.Parameter(torch.randn(heads, dim))

        # Initialize parameters
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.q_proj)

        # Activation function
        self.activation = F.silu if activation == "silu" else F.relu

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project input to key-value pairs
        kv = self.kv_proj(x)
        kv = kv.view(batch_size, seq_len, self.heads, self.dim, 2)
        kv = self.activation(kv)
        kv = self.dropout(kv)

        # Split into keys and values
        k, v = kv[..., 0], kv[..., 1]

        # Compute attention scores
        s = torch.einsum("hd,bthd->bth", self.q_proj, k)

        # Apply stable scan
        output = stable_scan_attention(s, v, mask)

        if self.concat_heads:
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, -1)
        else:
            output = output.mean(dim=2)

        return output


if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility

    # Test settings
    batch_size = 2
    seq_len = 4
    input_dim = 64
    num_heads = 4

    # Create random input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Create mask (optional)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -1] = False  # mask out last token

    # Initialize model
    model = AAREN(
        heads=num_heads,
        dim=input_dim,
        dropout=0.1,
        activation="silu",
        concat_heads=False,
    )

    # Forward pass
    output = model(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
