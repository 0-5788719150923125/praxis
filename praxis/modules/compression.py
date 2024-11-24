import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisCompressor(nn.Module):
    def __init__(self, config: AutoConfig, compressed_seq_len=64, compressed_dim=128):
        super().__init__()
        assert config.causal, "`compression=True` cannot be used with `causal=False`"
        self.input_dim = config.num_dims
        self.compressed_seq_len = compressed_seq_len
        self.compressed_dim = compressed_dim

        # Encoder components
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=compressed_dim,
            batch_first=True,
            bidirectional=False,
            dropout=config.dropout,
        )
        self.downsample = LearnedResampling(compressed_dim)

        # Decoder components
        self.decoder = nn.LSTM(
            input_size=compressed_dim,
            hidden_size=compressed_dim,
            batch_first=True,
            dropout=config.dropout,
        )
        self.upsample = LearnedResampling(compressed_dim)
        self.project = nn.Linear(compressed_dim, self.input_dim)

    def encode(self, x, attention_mask=None):
        self.residual = x

        # Encode sequence
        encoded, _ = self.encoder(x)

        # Compress sequence
        compressed = self.downsample(encoded, self.compressed_seq_len)

        # Handle attention mask
        compressed_mask = None
        if attention_mask is not None:
            original_dtype = attention_mask.dtype
            mask_expanded = attention_mask.float().unsqueeze(1)
            compressed_mask = F.adaptive_avg_pool1d(
                mask_expanded, self.compressed_seq_len
            ).squeeze(1)
            compressed_mask = (compressed_mask > 0.5).to(original_dtype)

        return compressed, compressed_mask

    def decode(self, compressed):
        # Decode sequence
        decoded, _ = self.decoder(compressed)

        # Uncompress sequence
        decoded = self.upsample(decoded, self.residual.size(1))
        decoded = self.project(decoded)

        # Add residual connection with the original input
        return decoded + self.residual

    def forward(self, x, attention_mask=None):
        """Convenience method for encode only"""
        return self.encode(x, attention_mask)


class LearnedResampling(nn.Module):
    """
    A unified module for sequence length adjustment (both upsampling and downsampling)
    using learned position embeddings and attention mechanisms.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, target_length):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Create position embeddings for target sequence length
        positions = torch.linspace(0, 1, target_length, device=x.device)
        position_embeddings = self.mlp(positions.unsqueeze(-1))

        # Expand position embeddings for batch dimension
        position_embeddings = position_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Compute attention scores
        attention = torch.bmm(
            position_embeddings, x.transpose(1, 2)
        )  # [batch, target_len, seq_len]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(target_length, seq_len, device=x.device), diagonal=1
        ).bool()
        attention = attention.masked_fill(causal_mask, float("-inf"))

        # Scale and normalize attention weights
        attention = F.softmax(attention / math.sqrt(self.hidden_dim), dim=-1)

        # Apply attention to get resampled sequence
        return torch.bmm(attention, x)


if __name__ == "__main__":

    class SequenceCompressorTester:
        def __init__(self):
            class DummyConfig:
                def __init__(self):
                    self.causal = True
                    self.num_dims = 64
                    self.dropout = 0.0

            config = DummyConfig()
            self.model = PraxisCompressor(
                config, compressed_seq_len=32, compressed_dim=128
            )
            self.criterion = nn.MSELoss()

        def test_encoding(self, batch_size, seq_len):
            """Test the encoding process"""
            print(f"\nTesting encoding with sequence length {seq_len}")

            # Create input and mask
            x = torch.randn(batch_size, seq_len, self.model.input_dim)
            mask = torch.ones(batch_size, seq_len)

            # Randomly mask some positions
            mask[:, seq_len // 2 :] = 0.0

            # Encode
            compressed, compressed_mask = self.model.encode(x, mask)

            # Verify shapes
            expected_shape = (
                batch_size,
                self.model.compressed_seq_len,
                self.model.compressed_dim,
            )
            expected_mask_shape = (batch_size, self.model.compressed_seq_len)

            actual_shape = tuple(compressed.shape)
            actual_mask_shape = tuple(compressed_mask.shape)

            print(f"Input shape: {tuple(x.shape)}")
            print(f"Input mask shape: {tuple(mask.shape)}")
            print(f"Compressed shape: {actual_shape}")
            print(f"Compressed mask shape: {actual_mask_shape}")
            print(f"Compression ratio: {seq_len/self.model.compressed_seq_len:.2f}x")

            # Verify compression maintains masking proportions
            original_mask_ratio = mask.float().mean()
            compressed_mask_ratio = compressed_mask.float().mean()
            print(f"Original mask ratio: {original_mask_ratio:.3f}")
            print(f"Compressed mask ratio: {compressed_mask_ratio:.3f}")

            assert (
                actual_shape == expected_shape
            ), f"Encoding shape mismatch! Expected {expected_shape}, got {actual_shape}"
            assert (
                actual_mask_shape == expected_mask_shape
            ), f"Mask shape mismatch! Expected {expected_mask_shape}, got {actual_mask_shape}"

            return compressed, compressed_mask

        def test_decoding(self, compressed_tuple, target_length):
            """Test the decoding process"""
            print(f"\nTesting decoding to length {target_length}")

            compressed, compressed_mask = compressed_tuple

            # Decode
            decoded = self.model.decode(compressed)

            # Verify shapes
            expected_shape = (compressed.size(0), target_length, self.model.input_dim)
            expected_mask_shape = (compressed.size(0), target_length)

            actual_shape = tuple(decoded.shape)

            print(f"Compressed shape: {tuple(compressed.shape)}")
            print(f"Compressed mask shape: {tuple(compressed_mask.shape)}")
            print(f"Decoded shape: {actual_shape}")

            assert (
                actual_shape == expected_shape
            ), f"Decoding shape mismatch! Expected {expected_shape}, got {actual_shape}"

            return decoded

        def test_full_pipeline(self, batch_size, seq_len):
            """Test the complete encode-decode pipeline"""
            print(f"\nTesting full pipeline with sequence length {seq_len}")

            # Create input and mask
            x = torch.randn(batch_size, seq_len, self.model.input_dim)
            mask = torch.ones(batch_size, seq_len)
            mask[:, seq_len // 2 :] = 0.0

            # Full forward pass
            compressed, compressed_mask = self.model.encode(x, mask)
            output = self.model.decode(compressed)

            # Compute reconstruction loss
            loss = self.criterion(output, x)

            print(f"Input shape: {tuple(x.shape)}")
            print(f"Input mask shape: {tuple(mask.shape)}")
            print(f"Compressed shape: {tuple(compressed.shape)}")
            print(f"Compressed mask shape: {tuple(compressed_mask.shape)}")
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
            test_lengths = [10, 100, 1000, 10000]

            for seq_len in test_lengths:
                # Test encoding
                compressed_tuple = self.test_encoding(batch_size, seq_len)

                # Test decoding
                self.test_decoding(compressed_tuple, seq_len)

                # Test full pipeline
                self.test_full_pipeline(batch_size, seq_len)

                print("-" * 80)

    tester = SequenceCompressorTester()
    tester.run_all_tests()
