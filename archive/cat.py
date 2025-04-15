import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# from: https://arxiv.org/abs/2504.06704
class CATAttention(nn.Module):
    """
    Circular-convolutional Attention (CAT) for Transformer models.

    This implements the CAT mechanism from the paper "CAT: Circular-convolutional Attention for Sub-Quadratic Transformers"
    with support for causal language modeling. The implementation provides both FFT-based and gather-based approaches.

    Args:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.0)
        use_fft (bool): Whether to use FFT-based approach (O(N log N)) or gather-based approach (O(N²)) (default: False)
        causal (bool): Whether to use causal attention (for language modeling) (default: True)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, use_fft=False, causal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.causal = causal
        self.use_fft = use_fft

        # In CAT, we use a merged query-key projection (W_A) and a separate value projection (W_V)
        # This reduces parameters compared to standard attention with separate Q, K, V projections
        self.W_A = nn.Linear(
            embed_dim, num_heads, bias=False
        )  # Merged query-key projection per head
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)  # Value projection

        # Output projection (similar to standard attention)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _roll_matrix(self, z, causal=True):
        """
        Create a roll matrix from vector z.
        If causal=True, ensures that no future token information leaks.

        Args:
            z (torch.Tensor): Input tensor of shape [batch_size, seq_len, num_heads]
            causal (bool): Whether to ensure causality

        Returns:
            torch.Tensor: The rolled matrix with shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, num_heads = z.shape

        # Reshape z for easier handling
        z = z.permute(0, 2, 1)  # [batch_size, num_heads, seq_len]

        if causal:
            # For causal setting, create a lower triangular mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=z.device))

            # Initialize the rolled matrix with all zeros
            roll_mat = torch.zeros(
                batch_size, num_heads, seq_len, seq_len, device=z.device
            )

            # Fill in the roll matrix in a causal manner
            for i in range(seq_len):
                # For each position i, we place elements z[0], z[1], ..., z[i] at positions (i, 0), (i, 1), ..., (i, i)
                for j in range(i + 1):
                    idx = i - j  # Distance from current position
                    roll_mat[:, :, i, j] = z[:, :, idx]

            # Apply mask to ensure causality
            roll_mat = roll_mat * mask.unsqueeze(0).unsqueeze(0)
        else:
            # For non-causal setting, create a standard circular matrix
            indices = (
                torch.arange(seq_len, device=z.device).unsqueeze(0).repeat(seq_len, 1)
            )

            # Calculate circular indices
            row_indices = torch.arange(seq_len, device=z.device).unsqueeze(1)
            circular_indices = (indices - row_indices) % seq_len

            # Gather elements from z using circular indices
            roll_mat = z.unsqueeze(2).expand(-1, -1, seq_len, -1)
            roll_mat = torch.gather(
                roll_mat,
                3,
                circular_indices.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, num_heads, -1, -1),
            )

        return roll_mat

    def _cat_fft(self, z_softmax, values, causal=True):
        """
        Compute attention using FFT-based circular convolution.

        Args:
            z_softmax (torch.Tensor): Softmax-normalized attention scores [batch_size, seq_len, num_heads]
            values (torch.Tensor): Value tensors [batch_size, seq_len, embed_dim]
            causal (bool): Whether to use causal attention

        Returns:
            torch.Tensor: Output tensor after applying attention [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, num_heads = z_softmax.shape

        # Reshape for multi-head processing
        z_softmax = z_softmax.permute(
            0, 2, 1
        ).contiguous()  # [batch_size, num_heads, seq_len]
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch_size, num_heads, seq_len, head_dim]

        if causal:
            # For causal setting, we need a different approach as FFT assumes circular convolution
            # We'll use a lower triangular mask to ensure causality

            # Create causal mask and apply to z_softmax
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=z_softmax.device)
            )

            # Expand z_softmax for matrix multiplication
            z_exp = z_softmax.unsqueeze(2).expand(
                -1, -1, seq_len, -1
            )  # [batch_size, num_heads, seq_len, seq_len]

            # Apply causal mask
            z_masked = z_exp * causal_mask.unsqueeze(0).unsqueeze(0)

            # Perform matrix multiplication (this is still O(N²) but ensures causality)
            # We're doing this because causal FFT requires special handling
            output = torch.matmul(
                z_masked, values
            )  # [batch_size, num_heads, seq_len, head_dim]
        else:
            # For non-causal setting, we can use FFT directly
            # Pad sequences to power of 2 for more efficient FFT
            next_power_of_2 = 2 ** (seq_len - 1).bit_length()

            # Pad z_softmax and values
            z_padded = F.pad(z_softmax, (0, next_power_of_2 - seq_len))
            values_padded = F.pad(values, (0, 0, 0, next_power_of_2 - seq_len))

            # Apply FFT
            z_fft = torch.fft.rfft(z_padded, dim=2)
            values_fft = torch.fft.rfft(values_padded, dim=2)

            # Multiply in frequency domain (for each head dimension)
            output_fft = []
            for i in range(self.head_dim):
                v_fft = values_fft[:, :, :, i]
                prod = z_fft.unsqueeze(3) * v_fft.unsqueeze(3)
                output_fft.append(prod)

            output_fft = torch.cat(output_fft, dim=3)

            # Apply inverse FFT and extract original sequence length
            output = torch.fft.irfft(output_fft, dim=2)[:, :, :seq_len, :]

        # Reshape output to original dimensions
        output = output.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, self.embed_dim)

        return output

    def _cat_gather(self, z_softmax, values, causal=True):
        """
        Compute attention using gather-based circular convolution.

        Args:
            z_softmax (torch.Tensor): Softmax-normalized attention scores [batch_size, seq_len, num_heads]
            values (torch.Tensor): Value tensors [batch_size, seq_len, embed_dim]
            causal (bool): Whether to use causal attention

        Returns:
            torch.Tensor: Output tensor after applying attention [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = z_softmax.shape

        # Reshape values for multi-head processing
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch_size, num_heads, seq_len, head_dim]

        # Create the roll matrix using z_softmax
        roll_mat = self._roll_matrix(
            z_softmax, causal=causal
        )  # [batch_size, num_heads, seq_len, seq_len]

        # Apply the roll matrix to values using matrix multiplication
        output = torch.matmul(
            roll_mat, values
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape output to original dimensions
        output = output.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, self.embed_dim)

        return output

    def forward(self, x, attention_mask=None):
        """
        Forward pass for CAT attention.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
                                                   1 for tokens to attend to, 0 for tokens to ignore

        Returns:
            torch.Tensor: Output tensor after applying attention [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project inputs to get z (query-key) and values
        z = self.W_A(x)  # [batch_size, seq_len, num_heads]
        values = self.W_V(x)  # [batch_size, seq_len, embed_dim]

        # Apply scaling
        z = z * self.scaling

        # Apply mask (if provided)
        if attention_mask is not None:
            # Expand mask to same dimensions as z
            expanded_mask = attention_mask.unsqueeze(-1).expand(-1, -1, self.num_heads)

            # Apply mask by setting masked positions to a large negative value
            z = z.masked_fill(expanded_mask == 0, -1e10)

        # Apply softmax to get attention weights
        z_softmax = F.softmax(z, dim=1)
        z_softmax = self.dropout(z_softmax)

        # Apply CAT using either FFT or gather approach
        if self.use_fft:
            output = self._cat_fft(z_softmax, values, causal=self.causal)
        else:
            output = self._cat_gather(z_softmax, values, causal=self.causal)

        # Apply output projection
        output = self.out_proj(output)

        return output


class CATTransformerBlock(nn.Module):
    """
    A Transformer block using CAT attention instead of standard attention.

    Args:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Dimension of the feed-forward network
        dropout (float): Dropout probability (default: 0.1)
        use_fft (bool): Whether to use FFT-based approach for CAT (default: False)
        causal (bool): Whether to use causal attention (default: True)
    """

    def __init__(
        self, embed_dim, num_heads, ff_dim, dropout=0.1, use_fft=False, causal=True
    ):
        super().__init__()
        self.attention = CATAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_fft=use_fft,
            causal=causal,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        # Apply attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + attn_output

        # Apply feed-forward network with residual connection and layer norm
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output

        return x


def visualize_attention_pattern(model, seq_len=20):
    """
    Visualize the attention pattern for a CAT attention model.

    Args:
        model (CATAttention): The CAT attention model
        seq_len (int): Sequence length for visualization
    """
    # Create a sample input
    x = torch.randn(1, seq_len, model.embed_dim)

    # Forward pass to get attention weights
    with torch.no_grad():
        # Extract z_softmax from the model
        z = model.W_A(x)
        z = z * model.scaling
        z_softmax = F.softmax(z, dim=1)

        # Create roll matrix
        roll_mat = model._roll_matrix(z_softmax, causal=model.causal)

    # Visualize the roll matrix for the first head
    plt.figure(figsize=(8, 6))
    plt.imshow(roll_mat[0, 0].cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.title(f"CAT Attention Pattern ({'Causal' if model.causal else 'Non-Causal'})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig(
        f"cat_attention_pattern_{'causal' if model.causal else 'non_causal'}.png"
    )
    plt.close()


def compare_speed(seq_lens, embed_dim=256, num_heads=8, num_runs=5):
    """
    Compare the speed of CAT attention with standard attention.

    Args:
        seq_lens (list): List of sequence lengths to test
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_runs (int): Number of runs for each test (for averaging)
    """
    # Initialize models
    cat_attention = CATAttention(embed_dim, num_heads, causal=True, use_fft=False)
    cat_fft_attention = CATAttention(embed_dim, num_heads, causal=True, use_fft=True)

    # Standard attention for comparison
    class StandardAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )

        def forward(self, x, attention_mask=None):
            # Convert mask format for nn.MultiheadAttention
            if attention_mask is not None:
                attn_mask = attention_mask == 0
                attn_mask = attn_mask.to(dtype=torch.bool)
            else:
                attn_mask = None

            return self.multihead_attn(
                x, x, x, key_padding_mask=attn_mask, need_weights=False
            )[0]

    std_attention = StandardAttention(embed_dim, num_heads)

    # Results storage
    results = {"seq_lens": seq_lens, "standard": [], "cat_gather": [], "cat_fft": []}

    # Run tests
    for seq_len in seq_lens:
        print(f"Testing sequence length: {seq_len}")

        # Create sample input
        x = torch.randn(1, seq_len, embed_dim)

        # Measure standard attention time
        std_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = std_attention(x)
            std_times.append(time.time() - start_time)
        avg_std_time = sum(std_times) / len(std_times)
        results["standard"].append(avg_std_time)

        # Measure CAT gather-based attention time
        cat_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = cat_attention(x)
            cat_times.append(time.time() - start_time)
        avg_cat_time = sum(cat_times) / len(cat_times)
        results["cat_gather"].append(avg_cat_time)

        # Measure CAT FFT-based attention time
        cat_fft_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = cat_fft_attention(x)
            cat_fft_times.append(time.time() - start_time)
        avg_cat_fft_time = sum(cat_fft_times) / len(cat_fft_times)
        results["cat_fft"].append(avg_cat_fft_time)

        print(
            f"  Standard: {avg_std_time:.4f}s, CAT (gather): {avg_cat_time:.4f}s, CAT (FFT): {avg_cat_fft_time:.4f}s"
        )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results["seq_lens"], results["standard"], "o-", label="Standard Attention")
    plt.plot(
        results["seq_lens"], results["cat_gather"], "s-", label="CAT (gather-based)"
    )
    plt.plot(results["seq_lens"], results["cat_fft"], "^-", label="CAT (FFT-based)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (seconds)")
    plt.title("Attention Computation Time vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig("attention_speed_comparison.png")
    plt.close()

    return results


if __name__ == "__main__":
    # Test parameters
    embed_dim = 256
    num_heads = 8
    seq_len = 32
    batch_size = 2

    print("=== Testing CAT Attention Implementation ===")

    # Create models
    cat_causal = CATAttention(embed_dim, num_heads, causal=True, use_fft=False)
    cat_non_causal = CATAttention(embed_dim, num_heads, causal=False, use_fft=False)
    cat_fft_causal = CATAttention(embed_dim, num_heads, causal=True, use_fft=True)

    print(f"Model parameter count: {sum(p.numel() for p in cat_causal.parameters())}")

    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test forward passes
    print("\nTesting forward passes...")

    output_causal = cat_causal(x)
    print(f"Causal gather-based output shape: {output_causal.shape}")

    output_non_causal = cat_non_causal(x)
    print(f"Non-causal gather-based output shape: {output_non_causal.shape}")

    output_fft_causal = cat_fft_causal(x)
    print(f"Causal FFT-based output shape: {output_fft_causal.shape}")

    # Test with attention mask
    print("\nTesting with attention mask...")
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len // 2 :] = 0  # Mask out second half of sequence

    output_masked = cat_causal(x, attention_mask=mask)
    print(f"Output with attention mask shape: {output_masked.shape}")

    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    visualize_attention_pattern(cat_causal, seq_len=20)
    visualize_attention_pattern(cat_non_causal, seq_len=20)

    # Test transformer block
    print("\nTesting transformer block...")
    transformer_block = CATTransformerBlock(
        embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim * 4, causal=True
    )

    transformer_output = transformer_block(x)
    print(f"Transformer block output shape: {transformer_output.shape}")

    # Speed comparison
    print("\nComparing speed with standard attention...")
    seq_lens = [32, 64, 128, 256, 512]
    speed_results = compare_speed(seq_lens, embed_dim=embed_dim, num_heads=num_heads)

    print("\nTest completed successfully!")
