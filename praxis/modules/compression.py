import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig


class PraxisCompressor(nn.Module):
    """
    Compresses inputs along the sequence length.
    """

    def __init__(self, config: AutoConfig, target_len: int = 256):
        super().__init__()
        num_features = config.num_dims
        self.target_len = target_len
        self.hidden_size = num_features // 2

        self.recurrent = nn.GRU(
            input_size=num_features, hidden_size=self.hidden_size, batch_first=True
        )

        self.projection = nn.Linear(self.hidden_size, num_features)

    def forward(self, x: Tensor, attention_mask: Tensor):
        residual = x
        batch_size, seq_len, num_features = x.shape

        # Calculate adaptive window size using ceil to ensure we cover the sequence
        window_size = math.ceil(seq_len / self.target_len)

        # Apply mask before compression
        x = x * attention_mask.unsqueeze(-1)

        # Create new mask for compressed sequence
        attention_mask = torch.ones((batch_size, self.target_len), device=x.device)

        # Reshape into windows
        windows = x.unfold(dimension=1, size=window_size, step=window_size)

        # Process each window
        windows_reshaped = windows.reshape(-1, window_size, num_features)
        _, hidden = self.recurrent(windows_reshaped)

        # Reshape hidden state
        hidden = hidden.squeeze(0).view(batch_size, -1, self.hidden_size)

        # Handle sequence length adjustment with front padding
        if hidden.size(1) < self.target_len:
            pad_size = self.target_len - hidden.size(1)
            padding = torch.zeros(
                batch_size, pad_size, self.hidden_size, device=hidden.device
            )
            hidden = torch.cat([padding, hidden], dim=1)  # Pad at front
        else:
            # If we got more segments, take the last target_len ones
            hidden = hidden[:, -self.target_len :, :]

        # Main projection
        output = self.projection(hidden)

        # Simple average pooling for residual
        residual = F.adaptive_avg_pool1d(
            residual.transpose(1, 2), self.target_len
        ).transpose(1, 2)

        # Add residual
        output = output + residual

        return output, attention_mask


if __name__ == "__main__":
    # Create a config class with required attributes
    class DummyConfig:
        num_dims = 16

    config = DummyConfig()

    # Create manual input tensor with shape [1, 4, 16]
    # Each row will have increasing values from 0.1 to 0.4
    x = torch.tensor(
        [
            [  # Batch dimension (1)
                [0.1] * 16,  # First sequence position
                [0.2] * 16,  # Second sequence position
                [0.3] * 16,  # Third sequence position
                [0.4] * 16,  # Fourth sequence position
            ]
        ],
        dtype=torch.float32,
    )

    # Create attention mask
    attention_mask = torch.ones(1, 4)

    # Initialize compressor
    compressor = PraxisCompressor(config, target_len=2)

    # Forward pass
    output, output_mask = compressor(x, attention_mask)

    # Print shapes and values
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input values:\n{x[0]}")  # Remove batch dimension for readability
    print(f"Output values:\n{output[0]}")  # Remove batch dimension for readability
