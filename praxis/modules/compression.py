import torch
import torch.nn as nn


class PraxisCompressor(nn.Module):
    """
    Compresses inputs along the sequence length.
    """

    def __init__(
        self, num_features, compression_ratio=2, hidden_size=256, threshold=64
    ):
        """
        Args:
            num_features: Number of features in input sequence
            compression_ratio: How much to compress sequence (e.g., 2 means half length)
            hidden_size: Size of GRU hidden state
            threshold: Omit inputs under this sequence length.
        """
        super().__init__()
        self.compression_ratio = compression_ratio
        self.hidden_size = hidden_size
        self.threshold = threshold

        # GRU layer to process sequence windows
        self.recurrent = nn.GRU(
            input_size=num_features, hidden_size=hidden_size, batch_first=True
        )

        # Project back to original feature dimension
        self.projection = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        Returns:
            Compressed tensor of shape [batch_size, seq_len//compression_ratio, num_features]
        """
        batch_size, seq_len, num_features = x.shape
        window_size = self.compression_ratio

        if seq_len < self.threshold:
            return x

        # Reshape sequence into windows
        num_windows = seq_len // window_size
        windows = x[:, : (num_windows * window_size), :].view(
            batch_size, num_windows, window_size, num_features
        )

        # Reshape to (batch_size * num_windows, window_size, num_features)
        windows_reshaped = windows.reshape(-1, window_size, num_features)

        # Run GRU and take final hidden state for each window
        _, hidden = self.recurrent(
            windows_reshaped
        )  # hidden shape: [1, batch_size * num_windows, hidden_size]
        hidden = hidden.squeeze(0)  # shape: [batch_size * num_windows, hidden_size]

        # Reshape back to batch dimension
        hidden = hidden.view(batch_size, num_windows, self.hidden_size)

        # Project back to original feature dimension
        output = self.projection(
            hidden
        )  # shape: [batch_size, seq_len//compression_ratio, num_features]

        return output
