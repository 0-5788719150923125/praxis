import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisCompressor(nn.Module):
    def __init__(
        self, config: AutoConfig, compressed_seq_len=64, compressed_dim=128, num_heads=8
    ):
        super(PraxisCompressor, self).__init__()
        self.input_dim = config.num_dims
        self.compressed_seq_len = compressed_seq_len
        self.compressed_dim = compressed_dim
        self.num_heads = num_heads

        # Project input to compressed_dim
        self.input_proj = nn.Linear(self.input_dim, self.compressed_dim)

        # Learnable queries for compression
        self.compressed_queries = nn.Parameter(
            torch.randn(compressed_seq_len, self.compressed_dim)
        )

        # Multihead Attention for compression
        self.attention = nn.MultiheadAttention(
            embed_dim=self.compressed_dim, num_heads=self.num_heads, batch_first=True
        )

        # Project back to input_dim for decoding
        self.output_proj = nn.Linear(self.compressed_dim, self.input_dim)

    def encode(self, x, attention_mask=None):
        """
        Compress the input sequence into a fixed-length sequence and compress the attention mask.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            attention_mask (Tensor, optional): Attention mask of shape [batch_size, seq_len] where 1 indicates valid tokens and 0 indicates padding.

        Returns:
            Tuple[Tensor, Tensor]:
                - Compressed embeddings of shape [batch_size, actual_compressed_seq_len, compressed_dim]
                - Compressed attention mask of shape [batch_size, actual_compressed_seq_len]
        """
        batch_size, seq_len, _ = x.size()

        # Adjust compressed_seq_len if necessary
        actual_compressed_seq_len = min(self.compressed_seq_len, seq_len)

        # Store target sequence length
        self.last_target_seq_len = torch.tensor(seq_len, device=x.device)

        # Project input embeddings to compressed_dim
        x_proj = self.input_proj(x)  # [batch_size, seq_len, compressed_dim]

        # Expand compressed queries for the batch
        # Shape: [batch_size, actual_compressed_seq_len, compressed_dim]
        queries = (
            self.compressed_queries[:actual_compressed_seq_len]
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Prepare key_padding_mask for MultiheadAttention
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # [batch_size, seq_len]
        else:
            key_padding_mask = None

        # Create a causal mask [actual_compressed_seq_len, seq_len]
        # Each compressed query i can attend to keys up to position i in the original sequence
        causal_mask = torch.tril(torch.ones(actual_compressed_seq_len, seq_len)).to(
            x.device
        )
        # Convert to float mask: 0 for allowed positions, -inf for masked positions
        causal_mask = causal_mask.masked_fill(
            causal_mask == 0, float("-inf")
        ).masked_fill(causal_mask == 1, float(0.0))

        # Apply Multihead Attention with the causal_mask
        compressed, _ = self.attention(
            query=queries,  # [batch_size, actual_compressed_seq_len, compressed_dim]
            key=x_proj,  # [batch_size, seq_len, compressed_dim]
            value=x_proj,  # [batch_size, seq_len, compressed_dim]
            key_padding_mask=key_padding_mask,  # [batch_size, seq_len]
            attn_mask=causal_mask,  # [actual_compressed_seq_len, seq_len]
        )

        # Compress the attention mask using adaptive average pooling
        if attention_mask is not None:
            # Pad the attention mask on the left to align with compression
            pad_size = actual_compressed_seq_len - 1
            if pad_size > 0:
                mask_padded = F.pad(
                    attention_mask.float(), (pad_size, 0)
                )  # [batch_size, seq_len + pad_size]
            else:
                mask_padded = attention_mask.float()

            # Reshape for pooling: [batch_size, 1, seq_len + pad_size]
            mask_padded = mask_padded.unsqueeze(1)

            # Apply adaptive average pooling to compress the mask
            mask_compressed = F.adaptive_avg_pool1d(
                mask_padded, actual_compressed_seq_len
            )

            # Threshold to obtain binary mask: 1 if average > 0.5, else 0
            compressed_mask = (
                (mask_compressed > 0.5).int().squeeze(1)
            )  # [batch_size, actual_compressed_seq_len]
        else:
            compressed_mask = None

        return (
            compressed,
            compressed_mask,
        )  # [batch_size, actual_compressed_seq_len, compressed_dim], [batch_size, actual_compressed_seq_len]

    def decode(self, compressed):
        """
        Decompress the fixed-length sequence back to the original sequence length.

        Args:
            compressed (Tensor): Compressed embeddings of shape [batch_size, actual_compressed_seq_len, compressed_dim]

        Returns:
            Tensor: Decoded embeddings of shape [batch_size, target_seq_len, input_dim]
        """
        # Retrieve target sequence length
        target_seq_len = self.last_target_seq_len.item()

        # Project back to input_dim
        decoded_proj = self.output_proj(
            compressed
        )  # [batch_size, actual_compressed_seq_len, input_dim]

        # Transpose for upsampling: [batch_size, input_dim, actual_compressed_seq_len]
        decoded_proj = decoded_proj.transpose(1, 2)

        # Use F.interpolate for upsampling to the exact target_seq_len
        decoded_upsampled = F.interpolate(
            decoded_proj, size=target_seq_len, mode="linear", align_corners=False
        )

        # Transpose back: [batch_size, target_seq_len, input_dim]
        decoded = decoded_upsampled.transpose(1, 2)

        return decoded  # [batch_size, target_seq_len, input_dim]

    def forward(self, x, attention_mask=None):
        return self.encode(x, attention_mask)


# Testing Code
if __name__ == "__main__":

    class PraxisCompressorTester:
        def __init__(self):
            class DummyConfig:
                def __init__(self):
                    self.num_dims = 64  # Input feature dimension
                    self.dropout = 0.1

            config = DummyConfig()
            self.model = PraxisCompressor(
                config, compressed_seq_len=64, compressed_dim=128, num_heads=8
            )
            self.criterion = nn.MSELoss()

        def test_encoding(self, batch_size, seq_len, device):
            """Test the encoding process"""
            print(f"\nTesting encoding with sequence length {seq_len} on {device}")

            # Create input tensor
            x = torch.randn(batch_size, seq_len, self.model.input_dim).to(device)

            # Create attention mask: 1 for valid tokens, 0 for padding
            mask = torch.ones(batch_size, seq_len).to(device)
            if seq_len > 10:
                mask[:, -seq_len // 2 :] = 0  # Simulate padding in the second half

            # Encode
            compressed, compressed_mask = self.model.encode(x, attention_mask=mask)

            # Determine actual_compressed_seq_len
            actual_compressed_seq_len = min(self.model.compressed_seq_len, seq_len)

            # Verify shapes
            expected_shape = (
                batch_size,
                actual_compressed_seq_len,
                self.model.compressed_dim,
            )
            expected_mask_shape = (
                batch_size,
                actual_compressed_seq_len,
            )

            actual_shape = compressed.shape
            actual_mask_shape = (
                compressed_mask.shape if compressed_mask is not None else None
            )

            print(f"Input shape: {x.shape}")
            print(f"Compressed shape: {compressed.shape}")
            print(f"Expected compressed shape: {expected_shape}")
            print(
                f"Compressed mask shape: {compressed_mask.shape if compressed_mask is not None else 'None'}"
            )
            print(f"Expected compressed mask shape: {expected_mask_shape}")

            assert (
                actual_shape == expected_shape
            ), f"Encoding shape mismatch! Expected {expected_shape}, got {actual_shape}"
            if compressed_mask is not None:
                assert (
                    actual_mask_shape == expected_mask_shape
                ), f"Mask shape mismatch! Expected {expected_mask_shape}, got {actual_mask_shape}"

            return compressed, compressed_mask

        def test_decoding(self, compressed_tuple, original_seq_len, device):
            """Test the decoding process"""
            print(
                f"\nTesting decoding to sequence length {original_seq_len} on {device}"
            )

            compressed, _ = compressed_tuple

            # Decode
            decoded = self.model.decode(compressed)

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

        def test_full_pipeline(self, batch_size, seq_len, device):
            """Test the complete encode-decode pipeline"""
            print(f"\nTesting full pipeline with sequence length {seq_len} on {device}")

            # Create input tensor
            x = torch.randn(batch_size, seq_len, self.model.input_dim).to(device)

            # Create attention mask: 1 for valid tokens, 0 for padding
            mask = torch.ones(batch_size, seq_len).to(device)
            if seq_len > 10:
                mask[:, -seq_len // 2 :] = 0  # Simulate padding in the second half

            # Full forward pass: encode and then decode
            compressed, compressed_mask = self.model.encode(x, attention_mask=mask)
            decoded = self.model.decode(compressed)

            # Compute reconstruction loss
            target_x = x  # Use the entire original input as target
            loss = self.criterion(decoded, target_x)

            print(f"Input shape: {x.shape}")
            print(f"Compressed shape: {compressed.shape}")
            print(
                f"Compressed mask shape: {compressed_mask.shape if compressed_mask is not None else 'None'}"
            )
            print(f"Decoded shape: {decoded.shape}")
            print(f"Reconstruction loss: {loss.item():.6f}")

            # Verify shapes match
            assert (
                decoded.shape == x.shape
            ), f"Shape mismatch! Input: {x.shape}, Output: {decoded.shape}"

        def run_all_tests(self):
            """Run the complete test suite"""
            print("Running PraxisCompressor test suite...")

            batch_size = 8
            test_lengths = [10, 50, 100, 200]

            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            for seq_len in test_lengths:
                # Test encoding
                compressed_tuple = self.test_encoding(batch_size, seq_len, device)

                # Test decoding
                self.test_decoding(compressed_tuple, seq_len, device)

                # Test full pipeline
                self.test_full_pipeline(batch_size, seq_len, device)

                print("-" * 80)

    tester = PraxisCompressorTester()
    tester.run_all_tests()
