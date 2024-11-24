import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisCompressor(nn.Module):
    def __init__(self, config, compressed_dim=128):
        super(PraxisCompressor, self).__init__()
        self.input_dim = config.num_dims
        self.compressed_dim = compressed_dim

        # Compression via causal convolution
        self.encoder_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.compressed_dim,
            kernel_size=4,
            stride=2,
            padding=0,  # Padding will be added manually for causality
        )

        # Decompression via causal transposed convolution
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=self.compressed_dim,
            out_channels=self.input_dim,
            kernel_size=4,
            stride=2,
            padding=0,  # Padding will be handled to align output size
        )

    def encode(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        # Apply causal padding on the left
        kernel_size = self.encoder_conv.kernel_size[0]
        pad = (kernel_size - 1, 0)  # Only pad the beginning
        x = F.pad(x, pad)

        # Pass through the convolutional encoder
        compressed = self.encoder_conv(
            x
        )  # [batch_size, compressed_dim, compressed_seq_len]
        compressed = compressed.transpose(
            1, 2
        )  # [batch_size, compressed_seq_len, compressed_dim]
        return compressed

    def decode(self, compressed, original_seq_len):
        # compressed: [batch_size, compressed_seq_len, compressed_dim]
        x = compressed.transpose(
            1, 2
        )  # [batch_size, compressed_dim, compressed_seq_len]

        # Pass through the transposed convolutional decoder
        decompressed = self.decoder_conv(x)  # [batch_size, input_dim, seq_len + extra]

        # Trim the output to match the original sequence length
        decompressed = decompressed[:, :, -(original_seq_len):]
        decompressed = decompressed.transpose(1, 2)  # [batch_size, seq_len, input_dim]
        return decompressed

    def forward(self, x):
        # For simplicity, forward runs the encode method
        return self.encode(x)


if __name__ == "__main__":

    class SimpleCompressorTester:
        def __init__(self):
            class DummyConfig:
                def __init__(self):
                    self.num_dims = 64  # Input feature dimension

            config = DummyConfig()
            self.model = PraxisCompressor(config, compressed_dim=128)
            self.criterion = nn.MSELoss()

        def test_encoding(self, batch_size, seq_len):
            """Test the encoding process"""
            print(f"\nTesting encoding with sequence length {seq_len}")

            # Create input tensor
            x = torch.randn(batch_size, seq_len, self.model.input_dim)

            # Encode
            compressed = self.model.encode(x)

            # Calculate expected compressed sequence length
            kernel_size = self.model.encoder_conv.kernel_size[0]
            stride = self.model.encoder_conv.stride[0]
            padding = kernel_size - 1  # Left padding
            L_in = seq_len
            L_out = (L_in + padding - (kernel_size - 1) - 1) // stride + 1
            expected_compressed_seq_len = L_out

            # Verify shapes
            expected_shape = (
                batch_size,
                expected_compressed_seq_len,
                self.model.compressed_dim,
            )

            actual_shape = compressed.shape

            print(f"Input shape: {x.shape}")
            print(f"Compressed shape: {compressed.shape}")
            print(f"Expected compressed shape: {expected_shape}")

            assert (
                actual_shape == expected_shape
            ), f"Encoding shape mismatch! Expected {expected_shape}, got {actual_shape}"

            return compressed

        def test_decoding(self, compressed, original_seq_len):
            """Test the decoding process"""
            print(f"\nTesting decoding to sequence length {original_seq_len}")

            # Decode
            decoded = self.model.decode(compressed, original_seq_len)

            # Verify shapes
            expected_shape = (
                compressed.size(0),
                original_seq_len,
                self.model.input_dim,
            )

            actual_shape = decoded.shape

            print(f"Compressed shape: {compressed.shape}")
            print(f"Decoded shape: {decoded.shape}")
            print(f"Expected decoded shape: {expected_shape}")

            assert (
                actual_shape == expected_shape
            ), f"Decoding shape mismatch! Expected {expected_shape}, got {actual_shape}"

            return decoded

        def test_full_pipeline(self, batch_size, seq_len):
            """Test the complete encode-decode pipeline"""
            print(f"\nTesting full pipeline with sequence length {seq_len}")

            # Create input tensor
            x = torch.randn(batch_size, seq_len, self.model.input_dim)

            # Full forward pass
            compressed = self.model.encode(x)
            output = self.model.decode(compressed, seq_len)

            # Compute reconstruction loss
            loss = self.criterion(output, x)

            print(f"Input shape: {x.shape}")
            print(f"Compressed shape: {compressed.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Reconstruction loss: {loss.item():.6f}")

            # Verify shapes match
            assert (
                x.shape == output.shape
            ), f"Shape mismatch! Input: {x.shape}, Output: {output.shape}"

        def run_all_tests(self):
            """Run the complete test suite"""
            print("Running SimpleCompressor test suite...")

            batch_size = 8
            test_lengths = [10, 50, 100, 200]

            for seq_len in test_lengths:
                # Test encoding
                compressed = self.test_encoding(batch_size, seq_len)

                # Test decoding
                self.test_decoding(compressed, seq_len)

                # Test full pipeline
                self.test_full_pipeline(batch_size, seq_len)

                print("-" * 80)

    tester = SimpleCompressorTester()
    tester.run_all_tests()
