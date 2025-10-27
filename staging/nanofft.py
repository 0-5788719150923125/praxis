import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # Create a lower-triangular mask and normalize per row
        mask = torch.tril(torch.ones(out_features, in_features))
        # Avoid division by zero
        row_sums = mask.sum(dim=1, keepdim=True)
        epsilon = 1e-8
        row_sums = row_sums + epsilon  # Add epsilon to zero row sums
        mask = mask / row_sums
        self.register_buffer("mask", mask)

    def forward(self, input):
        # Apply the masked and normalized weights
        masked_weight = self.weight * self.mask
        return F.linear(input, masked_weight, self.bias)


class PraxisNano(nn.Module):
    def __init__(self, seq_len, vocab_embed_dim, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Define the fft module with masked and normalized Linear layers
        self.fft = nn.Sequential(
            MaskedLinear(seq_len, seq_len, bias=False),
            MaskedLinear(seq_len, seq_len, bias=False),
        )

        # Feed-forward network with sine activation
        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, vocab_embed_dim),
            SineActivation(),
            nn.Linear(vocab_embed_dim, embed_dim),
        )

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        B, T, E = x.shape
        x = self.ln1(x)

        # Reshape x for fft module
        x_fft = x.transpose(1, 2)  # [B, E, T]
        x_fft = self.fft(x_fft)  # Apply masked and normalized Linear layers
        x_fft = x_fft.transpose(1, 2)  # [B, T, E]

        x = x + x_fft  # Residual connection
        x = self.ln2(x)
        x = x + self.ffw(x)  # Residual connection
        return x


if __name__ == "__main__":
    # Smoke tests
    batch_size = 4
    seq_len = 16
    vocab_embed_dim = 128
    embed_dim = 64

    block = PraxisNano(
        seq_len=seq_len, vocab_embed_dim=vocab_embed_dim, embed_dim=embed_dim
    )

    # Random input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    output = block(x)

    # Check output shape
    assert output.shape == x.shape, "Output shape does not match input shape."
    print("Test passed: Output shape matches input shape.")
