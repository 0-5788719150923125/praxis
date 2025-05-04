from typing import List, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.compression.base import NoCompression
from praxis.utils import create_block_ids

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class SequenceInterpolation(NoCompression):
    """
    A module for interpolation-based sequence reduction and expansion in language models.

    This module provides methods to reduce a sequence of token embeddings
    to a shorter length using interpolation, and then expand it back
    to its original length. This can be used to reduce the computational
    complexity of processing long sequences while preserving the essential
    patterns in the data.

    Also provides methods to handle block IDs for packed sequences.

    Uses torch.nn.functional.interpolate for efficient vectorized operations.
    """

    def __init__(self, config: ConfigType, method="linear", factor: int = 0.9):
        """
        Initialize the SequenceInterpolation.

        Args:
            method: The method to use for interpolation.
                Options: 'linear', 'nearest'
            factor: Float between 0 and 1 representing the target length as a
                    fraction of the original length
        """
        super().__init__(config)

        self.factor = factor

        if not (0 < factor <= 1):
            raise ValueError(f"Factor must be between 0 and 1, got {factor}")

        # Supported interpolation methods for F.interpolate
        self.supported_modes = {"linear", "nearest"}

        if method not in self.supported_modes:
            raise ValueError(
                f"Interpolation method '{method}' not supported. "
                f"Supported methods: {self.supported_modes}"
            )

        self.method = method

    def reduce_sequence(self, sequence: Tensor):
        """
        Reduce sequence length by a multiplication factor (e.g., 0.9 for 90% length).

        Args:
            sequence: Tensor of shape (batch_size, seq_length, hidden_dim)

        Returns:
            Reduced sequence of shape (batch_size, target_length, hidden_dim)
        """

        # Calculate target length, ensuring at least 1 token
        batch_size, seq_length, hidden_dim = sequence.shape
        target_length = max(1, int(seq_length * self.factor))

        return self._reduce_sequence(sequence, target_length)

    def _reduce_sequence(self, sequence: Tensor, target_length: int):
        """
        Internal method to reduce sequence length through interpolation.

        Args:
            sequence: Tensor of shape (batch_size, seq_length, hidden_dim)
            target_length: The desired reduced sequence length

        Returns:
            Reduced sequence of shape (batch_size, target_length, hidden_dim)
        """
        batch_size, seq_length, hidden_dim = sequence.shape

        # If target length is already equal to sequence length, return the original sequence
        if target_length == seq_length:
            return sequence

        # If target length is greater than sequence length, that's not a reduction
        if target_length > seq_length:
            raise ValueError(
                f"Target length ({target_length}) must be less than or equal to sequence length ({seq_length})"
            )

        # Reshape for interpolate: [batch_size, hidden_dim, seq_length]
        # torch.nn.functional.interpolate expects [N, C, L...]
        x = sequence.transpose(1, 2)

        # Use F.interpolate to perform the reduction in a vectorized way
        mode = self.method
        x_reduced = F.interpolate(
            x,
            size=target_length,
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )

        # Reshape back to [batch_size, target_length, hidden_dim]
        reduced_sequence = x_reduced.transpose(1, 2)

        return reduced_sequence

    def expand_sequence(self, sequence: Tensor, original_length: int):
        """
        Expand a reduced sequence back to its original length through interpolation.

        Args:
            sequence: Tensor of shape (batch_size, reduced_length, hidden_dim)
            original_length: The original sequence length to expand to

        Returns:
            Expanded sequence of shape (batch_size, original_length, hidden_dim)
        """
        batch_size, reduced_length, hidden_dim = sequence.shape

        # If reduced length is already equal to original length, return the input sequence
        if reduced_length == original_length:
            return sequence

        # If reduced length is greater than original length, that's not an expansion
        if reduced_length > original_length:
            raise ValueError(
                f"Reduced length ({reduced_length}) must be less than or equal to original length ({original_length})"
            )

        # Reshape for interpolate: [batch_size, hidden_dim, reduced_length]
        x = sequence.transpose(1, 2)

        # Use F.interpolate to perform the expansion in a vectorized way
        mode = self.method
        x_expanded = F.interpolate(
            x,
            size=original_length,
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )

        # Reshape back to [batch_size, original_length, hidden_dim]
        expanded_sequence = x_expanded.transpose(1, 2)

        return expanded_sequence

    def reduce_block_ids(self, block_ids: Tensor):
        """
        Reduce block IDs by a multiplication factor (e.g., 0.9 for 90% length).

        Args:
            block_ids: Tensor of shape (batch_size, seq_length) containing integer block IDs

        Returns:
            Reduced block IDs of shape (batch_size, target_length)
        """

        # Calculate target length, ensuring at least 1 token
        batch_size, seq_length = block_ids.shape
        target_length = max(1, int(seq_length * self.factor))

        return self._reduce_block_ids(block_ids, target_length)

    def _reduce_block_ids(self, block_ids: Tensor, target_length: int):
        """
        Internal method to reduce block IDs to match a reduced token sequence length.

        Args:
            block_ids: Tensor of shape (batch_size, seq_length) containing integer block IDs
            target_length: The desired reduced length

        Returns:
            Reduced block IDs of shape (batch_size, target_length)
        """
        batch_size, seq_length = block_ids.shape

        # If target length is already equal to sequence length, return the original block IDs
        if target_length == seq_length:
            return block_ids

        # If target length is greater than sequence length, that's not a reduction
        if target_length > seq_length:
            raise ValueError(
                f"Target length ({target_length}) must be less than or equal to sequence length ({seq_length})"
            )

        # Get sampling indices using the same approach as in reduce_sequence
        indices = torch.linspace(
            0, seq_length - 1, target_length, device=block_ids.device
        ).long()

        # Sample the block IDs at these indices
        reduced_block_ids = torch.gather(
            block_ids, 1, indices.unsqueeze(0).expand(batch_size, -1)
        )

        # Handle edge case: ensure first token's block ID is preserved
        reduced_block_ids[:, 0] = block_ids[:, 0]

        # Handle edge case: preserve the last block's ID
        max_block_id_indices = torch.max(block_ids, dim=1)[1]
        for b in range(batch_size):
            # Find the max block ID in the original sequence
            max_block_id = block_ids[b, max_block_id_indices[b]]

            # Ensure the max block ID is preserved in the reduced sequence
            last_block_idx = (block_ids[b] == max_block_id).nonzero(as_tuple=True)[0][0]

            # Map to reduced sequence
            reduced_idx = torch.min(indices[indices >= last_block_idx])
            if reduced_idx < target_length:
                reduced_block_ids[b, reduced_idx:] = max_block_id

        return reduced_block_ids


if __name__ == "__main__":
    # Test the SequenceInterpolation
    class DummyConfig:
        debug = False

    config = DummyConfig()

    # Create a sample sequence
    batch_size = 2
    seq_length = 20
    hidden_dim = 4

    # Create a sample sequence with a clear pattern for visualization
    sequence = torch.zeros(batch_size, seq_length, hidden_dim)

    # First batch: embedding dim 0 has an increasing pattern
    sequence[0, :, 0] = torch.linspace(0, 1, seq_length)
    # First batch: embedding dim 1 has a sine wave pattern
    sequence[0, :, 1] = torch.sin(torch.linspace(0, 2 * torch.pi, seq_length))
    # Second batch: embedding dim 0 has a decreasing pattern
    sequence[1, :, 0] = torch.linspace(1, 0, seq_length)
    # Second batch: embedding dim 1 has a cosine wave pattern
    sequence[1, :, 1] = torch.cos(torch.linspace(0, 2 * torch.pi, seq_length))

    print(
        "Testing SequenceInterpolation with interpolation-based sequence reduction..."
    )

    # Test with different interpolation methods
    for method in ["linear", "nearest"]:
        print(f"\n=== Testing with {method} interpolation ===")

        # Test with different reduction factors
        for factor in [0.9, 0.7, 0.5]:
            # Initialize the SequenceInterpolation
            merger = SequenceInterpolation(config, method=method, factor=factor)

            print(f"\nReduction to {factor*100}% of original length")

            # Reduce the sequence using the factor-based method
            reduced = merger.reduce_sequence(sequence)

            # Expand the sequence back
            expanded = merger.expand_sequence(reduced, seq_length)

            # Calculate reconstruction error
            mse = torch.mean((sequence - expanded) ** 2).item()
            print(f"  Mean Squared Error: {mse:.6f}")

            # Verify shapes
            print(f"  Original shape: {sequence.shape}")
            print(f"  Reduced shape: {reduced.shape}")
            print(f"  Expanded shape: {expanded.shape}")

    # Test block ID reduction
    print("\n=== Testing Block ID Reduction ===")

    # Create sample input IDs with some special tokens
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))

    # Place some special tokens to create blocks
    special_token_id = 1

    # First sequence: special tokens at positions 0, 8, 16, 24
    input_ids[0, 0] = special_token_id
    input_ids[0, 8] = special_token_id
    input_ids[0, 16] = special_token_id
    input_ids[0, 24] = special_token_id

    # Second sequence: special tokens at positions 0, 10, 20
    input_ids[1, 0] = special_token_id
    input_ids[1, 10] = special_token_id
    input_ids[1, 20] = special_token_id

    # Create the block IDs
    block_ids = create_block_ids(input_ids, [special_token_id])

    print("Original input IDs with special tokens (showing first sequence):")
    for i in range(seq_length):
        if input_ids[0, i].item() == special_token_id:
            print(f"Position {i}: SPECIAL TOKEN")

    print("\nOriginal block IDs:")
    print(block_ids)

    # Test reducing to different percentages
    for factor in [0.75, 0.5]:
        merger = SequenceInterpolation(config, method="linear", factor=factor)

        print(f"\nReducing block IDs to {factor*100}% of original length")

        # Reduce the block IDs with the factor-based method
        reduced_block_ids = merger.reduce_block_ids(block_ids)

        print(f"Reduced block IDs:")
        print(reduced_block_ids)

        # Check if the block structure is preserved
        print("\nBlock ID transitions (first sequence):")
        print("Original:")
        transitions = (block_ids[0, 1:] != block_ids[0, :-1]).nonzero(as_tuple=True)[0]
        print(f"  Transitions at positions: {[t.item() + 1 for t in transitions]}")

        print("Reduced:")
        reduced_transitions = (
            reduced_block_ids[0, 1:] != reduced_block_ids[0, :-1]
        ).nonzero(as_tuple=True)[0]
        print(
            f"  Transitions at positions: {[t.item() + 1 for t in reduced_transitions]}"
        )

        # Analyze block counts
        original_blocks = block_ids.max().item()
        reduced_blocks = reduced_block_ids.max().item()
        print(f"\nOriginal number of blocks: {original_blocks}")
        print(f"Reduced number of blocks: {reduced_blocks}")

    # Performance comparison
    import time

    # Create a larger sequence for performance testing
    batch_size = 8
    seq_length = 1024
    hidden_dim = 768

    large_sequence = torch.randn(batch_size, seq_length, hidden_dim)

    # Create large block IDs (simulating document boundaries every 128 tokens)
    large_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    for b in range(batch_size):
        for i in range(0, seq_length, 128):
            large_input_ids[b, i] = special_token_id
    large_block_ids = create_block_ids(large_input_ids, [special_token_id])

    print("\n=== Performance Testing ===")

    # Test reduction factor
    factor = 0.5

    merger = SequenceInterpolation(config, method="linear", factor=factor)

    # Measure reduction time
    start_time = time.time()
    reduced = merger.reduce_sequence(large_sequence)
    reduction_time = time.time() - start_time

    # Measure expansion time
    start_time = time.time()
    expanded = merger.expand_sequence(reduced, seq_length)
    expansion_time = time.time() - start_time

    # Measure block ID reduction time
    start_time = time.time()
    reduced_block_ids = merger.reduce_block_ids(large_block_ids)
    block_reduction_time = time.time() - start_time

    print(f"Large sequence shape: {large_sequence.shape}")
    print(f"Sequence reduction time ({factor*100}%): {reduction_time:.4f} seconds")
    print(f"Sequence expansion time (back to 100%): {expansion_time:.4f} seconds")
    print(
        f"Block ID reduction time ({factor*100}%): {block_reduction_time:.4f} seconds"
    )

    # Test reduction+expansion in a language model context
    print("\n=== Language Model Context Example with Block IDs ===")

    # Create mock language model components
    embedding_dim = 512
    embedding_layer = nn.Embedding(10000, embedding_dim)

    # Create sample input IDs
    seq_length = 128
    input_ids = torch.randint(0, 10000, (batch_size, seq_length))

    # Create block IDs (simulating 4 documents of 32 tokens each)
    for b in range(batch_size):
        for i in range(0, seq_length, 32):
            input_ids[b, i] = special_token_id
    block_ids = create_block_ids(input_ids, [special_token_id])

    # Create sample language model pipeline
    # Get embeddings
    embeddings = embedding_layer(input_ids)

    # Reduce sequence length using factor
    factor = 0.75
    merger = SequenceInterpolation(config, method="linear", factor=factor)
    reduced_embeddings = merger.reduce_sequence(embeddings)
    reduced_block_ids = merger.reduce_block_ids(block_ids)

    print(
        f"Original shapes: embeddings={embeddings.shape}, block_ids={block_ids.shape}"
    )
    print(
        f"Reduced shapes: embeddings={reduced_embeddings.shape}, block_ids={reduced_block_ids.shape}"
    )

    # Simulate processing in reduced space
    processed_reduced = reduced_embeddings + torch.randn_like(reduced_embeddings) * 0.1

    # Expand back to original length for loss calculation
    expanded_embeddings = merger.expand_sequence(processed_reduced, seq_length)
    print(f"Expanded embeddings: {expanded_embeddings.shape}")

    # Calculate example reconstruction error
    mse = torch.mean((embeddings - expanded_embeddings) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")

    # Verify block structure is preserved
    original_blocks_per_seq = [block_ids[b].unique().numel() for b in range(batch_size)]
    reduced_blocks_per_seq = [
        reduced_block_ids[b].unique().numel() for b in range(batch_size)
    ]

    print("\nBlock structure preservation:")
    for b in range(batch_size):
        print(
            f"  Sequence {b}: Original blocks={original_blocks_per_seq[b]}, Reduced blocks={reduced_blocks_per_seq[b]}"
        )

    print("\nAll tests completed!")
