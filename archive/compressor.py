import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCompressor(nn.Module):
    def __init__(self, input_dim=64, compressed_seq_len=32, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_seq_len = compressed_seq_len
        self.hidden_dim = hidden_dim

        # Encoder components
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.compress = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder components
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encode and compress input sequence to fixed length
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            torch.Tensor: Compressed tensor of shape (batch_size, compressed_seq_len, hidden_dim)
        """
        # Encode sequence
        encoded, _ = self.encoder_lstm(x)

        # Compress to fixed length using adaptive average pooling
        compressed = F.adaptive_avg_pool1d(
            encoded.transpose(1, 2), self.compressed_seq_len
        ).transpose(1, 2)

        # Project to hidden dimension
        compressed = self.compress(compressed)

        return compressed

    def decode(self, compressed, target_length):
        """
        Decode compressed representation back to original sequence length
        Args:
            compressed (torch.Tensor): Compressed tensor of shape (batch_size, compressed_seq_len, hidden_dim)
            target_length (int): Desired output sequence length
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, target_length, input_dim)
        """
        # Decode compressed representation
        decoded, _ = self.decoder_lstm(compressed)

        # Upsample back to target sequence length
        decoded = F.interpolate(
            decoded.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        # Project to input dimension
        output = self.output_proj(decoded)

        return output

    def forward(self, x):
        """
        Convenience method for full encode-decode pipeline
        """
        compressed = self.encode(x)
        return self.decode(compressed, x.size(1))


class SequenceCompressorTester:
    def __init__(self):
        self.model = SequenceCompressor(
            input_dim=64, compressed_seq_len=32, hidden_dim=128
        )
        self.criterion = nn.MSELoss()

    def test_encoding(self, batch_size, seq_len):
        """Test the encoding process"""
        print(f"\nTesting encoding with sequence length {seq_len}")

        # Create input
        x = torch.randn(batch_size, seq_len, self.model.input_dim)

        # Encode
        compressed = self.model.encode(x)

        # Verify shapes
        expected_shape = (
            batch_size,
            self.model.compressed_seq_len,
            self.model.hidden_dim,
        )
        actual_shape = tuple(compressed.shape)

        print(f"Input shape: {tuple(x.shape)}")
        print(f"Compressed shape: {actual_shape}")
        print(f"Expected compressed shape: {expected_shape}")
        print(f"Compression ratio: {seq_len/self.model.compressed_seq_len:.2f}x")

        assert (
            actual_shape == expected_shape
        ), f"Encoding shape mismatch! Expected {expected_shape}, got {actual_shape}"

        return compressed

    def test_decoding(self, compressed, target_length):
        """Test the decoding process"""
        print(f"\nTesting decoding to length {target_length}")

        # Decode
        decoded = self.model.decode(compressed, target_length)

        # Verify shapes
        expected_shape = (compressed.size(0), target_length, self.model.input_dim)
        actual_shape = tuple(decoded.shape)

        print(f"Compressed shape: {tuple(compressed.shape)}")
        print(f"Decoded shape: {actual_shape}")
        print(f"Expected decoded shape: {expected_shape}")

        assert (
            actual_shape == expected_shape
        ), f"Decoding shape mismatch! Expected {expected_shape}, got {actual_shape}"

        return decoded

    def test_full_pipeline(self, batch_size, seq_len):
        """Test the complete encode-decode pipeline"""
        print(f"\nTesting full pipeline with sequence length {seq_len}")

        # Create input
        x = torch.randn(batch_size, seq_len, self.model.input_dim)

        # Full forward pass
        compressed = self.model.encode(x)
        output = self.model.decode(compressed, seq_len)

        # Compute reconstruction loss
        loss = self.criterion(output, x)

        print(f"Input shape: {tuple(x.shape)}")
        print(f"Compressed shape: {tuple(compressed.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Reconstruction loss: {loss.item():.4f}")

        # Verify shapes match
        assert (
            x.shape == output.shape
        ), f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"

    def run_all_tests(self):
        """Run complete test suite"""
        print("Running SequenceCompressor test suite...")

        batch_size = 8
        test_lengths = [50, 100, 200]

        for seq_len in test_lengths:
            # Test encoding
            compressed = self.test_encoding(batch_size, seq_len)

            # Test decoding
            self.test_decoding(compressed, seq_len)

            # Test full pipeline
            self.test_full_pipeline(batch_size, seq_len)

            print("-" * 80)


if __name__ == "__main__":
    tester = SequenceCompressorTester()
    tester.run_all_tests()
