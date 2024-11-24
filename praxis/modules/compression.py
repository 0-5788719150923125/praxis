import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisCompressor(nn.Module):
    def __init__(self, config: AutoConfig, compressed_seq_len=64, compressed_dim=128):
        super().__init__()
        self.input_dim = config.num_dims
        self.compressed_seq_len = compressed_seq_len
        self.compressed_dim = compressed_dim

        # Initial encoding
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=compressed_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
        )

        # Multi-scale pooling
        self.compress = MultiScalePooling(compressed_dim, num_scales=8)

        # Decoder components remain the same
        self.decoder = nn.LSTM(
            input_size=compressed_dim,
            hidden_size=compressed_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
        )
        self.upsample = CausalUpsampling(compressed_dim)
        self.project = nn.Linear(compressed_dim, self.input_dim)

    def encode(self, x, attention_mask=None):
        self.residual = x

        # Encode sequence
        encoded, _ = self.encoder(x)

        # Multi-scale compression
        compressed = self.compress(encoded, self.compressed_seq_len)

        # Handle attention mask (simplified for multi-scale)
        compressed_mask = None
        if attention_mask is not None:
            original_dtype = attention_mask.dtype
            compressed_mask = (
                F.interpolate(
                    attention_mask.float().unsqueeze(1),
                    size=self.compressed_seq_len,
                    mode="linear",
                ).squeeze(1)
                > 0.5
            ).to(original_dtype)

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


class MultiScalePooling(nn.Module):
    def __init__(self, dim, num_scales=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        # Learn how to weight different scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x, target_length):
        batch_size, seq_len, _ = x.shape
        pools = []

        for scale in range(self.num_scales):
            # Exponentially increasing kernel sizes
            kernel_size = 2**scale
            stride = max(seq_len // target_length // self.num_scales, 1)

            # Causal pooling (pad on the right)
            padded = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
            pooled = F.avg_pool1d(
                padded, kernel_size=kernel_size, stride=stride, padding=0
            )

            # Interpolate to target length
            pooled = F.interpolate(pooled, size=target_length, mode="linear")
            pools.append(pooled)

        # Combine different scales with learned weights
        scale_weights = F.softmax(self.scale_weights, dim=0)
        pools = torch.stack(pools, dim=0)
        combined = (pools * scale_weights.view(-1, 1, 1, 1)).sum(dim=0)

        return combined.transpose(1, 2)


class CausalUpsampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Maybe add a small MLP to help with the projection
        self.pre_project = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x, target_length):
        # First, condition the representations
        x = self.pre_project(x)

        # Simple linear interpolation maintains causality
        # and preserves the transformed representations
        upsampled = F.interpolate(
            x.transpose(1, 2), size=target_length, mode="linear"
        ).transpose(1, 2)

        return upsampled


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
