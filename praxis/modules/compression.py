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
        """
        Args:
            num_features: Number of features in input sequence
            target_len: Desired output sequence length
            hidden_size: Size of LSTM hidden state
        """
        super().__init__()
        self.target_len = target_len
        self.hidden_size = hidden_size

        self.recurrent = nn.LSTM(
            input_size=num_features, hidden_size=hidden_size, batch_first=True
        )

        self.projection = nn.Linear(hidden_size, num_features)

    def forward(self, x: Tensor, attention_mask: Tensor):
        batch_size, seq_len, num_features = x.shape

        # Calculate adaptive window size
        window_size = max(1, seq_len // self.target_len)

        # Apply mask before compression
        x = x * attention_mask.unsqueeze(-1)

        # Create new mask for compressed sequence
        attention_mask = torch.ones((batch_size, self.target_len), device=x.device)

        # Pad sequence if needed
        pad_len = (window_size * self.target_len) - seq_len
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, num_features, device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len = x.shape[1]

        # Reshape into windows
        windows = x.view(batch_size, self.target_len, window_size, num_features)

        # Process each window
        windows_reshaped = windows.reshape(-1, window_size, num_features)
        _, (hidden, _) = self.recurrent(windows_reshaped)

        # Reshape to target length
        hidden = hidden.view(batch_size, self.target_len, self.hidden_size)
        output = self.projection(hidden)

        return output, attention_mask
