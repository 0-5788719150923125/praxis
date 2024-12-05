import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class MLPMixerForSequences(nn.Module):
    def __init__(
        self, max_seq_length, channel_dim, token_dim=128, channel_dim_multiplier=4
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.channel_dim = channel_dim

        # Token mixing
        self.token_norm = nn.LayerNorm(channel_dim)
        self.token_mix = MLPBlock(max_seq_length, token_dim)

        # Channel mixing
        self.channel_norm = nn.LayerNorm(channel_dim)
        self.channel_mix = MLPBlock(channel_dim, channel_dim * channel_dim_multiplier)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channel_dim)
            attention_mask: Boolean mask of shape (batch_size, seq_length)
        """
        # Handle padding for sequences shorter than max_length
        batch_size, seq_length, _ = x.shape

        if seq_length < self.max_seq_length:
            # Create padding
            padding = torch.zeros(
                (batch_size, self.max_seq_length - seq_length, self.channel_dim),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)

            # Update attention mask if none provided
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=x.device)
            padding_mask = torch.zeros(
                (batch_size, self.max_seq_length - seq_length), device=x.device
            )
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)

        elif seq_length > self.max_seq_length:
            # Truncate sequence
            x = x[:, : self.max_seq_length, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_seq_length]

        # Token mixing with mask
        y = self.token_norm(x)
        y = y.transpose(1, 2)  # (batch_size, channel_dim, seq_length)
        y = self.token_mix(y)
        y = y.transpose(1, 2)  # (batch_size, seq_length, channel_dim)

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_length, 1)
            y = y * mask

        x = x + y  # Skip connection

        # Channel mixing
        y = self.channel_norm(x)
        y = self.channel_mix(y)

        # Apply mask again
        if attention_mask is not None:
            y = y * mask

        x = x + y  # Skip connection

        return x, attention_mask


if __name__ == "__main__":
    # Test configurations
    batch_size = 4
    max_seq_length = 16
    channel_dim = 64

    # Initialize model
    mixer = MLPMixerForSequences(max_seq_length=max_seq_length, channel_dim=channel_dim)

    # Test 1: Sequence shorter than max_length
    x_short = torch.randn(batch_size, 10, channel_dim)
    out, mask = mixer(x_short)
    assert out.shape == (batch_size, max_seq_length, channel_dim)
    assert mask.shape == (batch_size, max_seq_length)
    print("Test 1 (Short sequence handling) passed!")

    # Test 2: Sequence longer than max_length
    x_long = torch.randn(batch_size, 20, channel_dim)
    out, mask = mixer(x_long)
    assert out.shape == (batch_size, max_seq_length, channel_dim)
    print("Test 2 (Long sequence handling) passed!")

    # Test 3: Exact length sequence
    x_exact = torch.randn(batch_size, max_seq_length, channel_dim)
    out, mask = mixer(x_exact)
    assert out.shape == x_exact.shape
    print("Test 3 (Exact length handling) passed!")

    # Test 4: Masking effect
    x = torch.ones(batch_size, 10, channel_dim)
    attention_mask = torch.ones(batch_size, 10)
    attention_mask[:, 5:] = 0  # Mask out second half
    out, _ = mixer(x, attention_mask)
    # Check if masked positions have different values than unmasked
    assert torch.any(out[:, 0, :] != out[:, 9, :])
    print("Test 4 (Masking effect) passed!")

    print("All tests passed!")
