import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PraxisCompressor(nn.Module):
    """
    Compresses inputs along the sequence length.
    """

    def __init__(self, num_features, target_len=256, hidden_size=256):
        super().__init__()
        self.target_len = target_len
        self.hidden_size = hidden_size

        self.recurrent = nn.GRU(
            input_size=num_features, hidden_size=hidden_size, batch_first=True
        )

        self.projection = nn.Linear(hidden_size, num_features)

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
