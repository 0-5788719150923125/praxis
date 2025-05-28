import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache


class SyntaxesAttention(nn.Module):
    """
    Syntaxes Attention mechanism that selects top-k tokens, computes attention
    in reduced space, then interpolates back to full sequence length.

    This reduces computational complexity while maintaining multi-token dependencies
    through interpolation.

    Note: the original reddit comment that described this process was fixed on the "sorting"
    aspect, for some reason. They thought it could be used to extrapolate into a probability
    distribution of terms over time, sort of like a symbolic representation of what the original
    sequence was.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        
        # Ensure num_heads divides hidden_size evenly
        while self.hidden_size % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        
        self.head_dim = self.hidden_size // self.num_heads
        self.causal = getattr(config, "causal", True)

        # Syntaxes specific parameters
        self.compression_size = getattr(
            config, "compression_size", 64
        )  # Target compressed size
        self.compression_method = getattr(
            config, "compression_method", "learnable_interpolation"
        )  # learnable_interpolation, linear_interpolation, pooling
        self.scoring_method = getattr(
            config, "scoring_method", "norm"
        )  # norm or learned (for backwards compat)
        self.selection_method = getattr(
            config, "selection_method", "interpolation"
        )  # interpolation, sliding_window, top_k

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Learned scoring function (alternative to norm-based)
        if self.scoring_method == "learned":
            self.score_proj = nn.Linear(self.hidden_size, 1, bias=False)

        # Learnable interpolation components
        if self.compression_method == "learnable_interpolation":
            # Learnable interpolation weights - maps from seq_len positions to compression_size positions
            self.interpolation_proj = nn.Linear(
                self.hidden_size, self.compression_size, bias=False
            )
            # Position embeddings for the compressed sequence
            self.compressed_pos_emb = nn.Parameter(
                torch.randn(1, self.compression_size, self.hidden_size) * 0.02
            )
            # Learnable expansion weights - maps from compressed space back to sequence space
            # We'll initialize this with a reasonable max sequence length
            max_seq_len = getattr(config, "max_position_embeddings", 2048)
            self.expansion_proj = nn.Linear(self.hidden_size, max_seq_len, bias=False)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _compute_token_scores(self, x: Tensor) -> Tensor:
        """Compute importance scores for each token."""
        if self.scoring_method == "norm":
            # Use L2 norm as importance score
            scores = torch.norm(x, dim=-1)
        elif self.scoring_method == "learned":
            # Use learned scoring function
            scores = self.score_proj(x).squeeze(-1)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        return scores

    def _compress_sequence(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compress input sequence to smaller representation."""
        batch_size, seq_len, hidden_size = x.shape
        k = min(self.compression_size, seq_len)

        if self.selection_method == "interpolation":
            return self._interpolate_sequence(x, k)
        elif self.selection_method == "top_k":
            # Get top-k indices based on importance scores
            scores = self._compute_token_scores(x)
            _, selected_indices = torch.topk(scores, k, dim=-1, sorted=True)
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
            selected_tokens = x[batch_indices, selected_indices]
            return selected_tokens, selected_indices
        elif self.selection_method == "sliding_window":
            # Always select the last k tokens (sliding window)
            start_idx = max(0, seq_len - k)
            selected_indices = (
                torch.arange(start_idx, seq_len, device=x.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
            selected_tokens = x[batch_indices, selected_indices]
            return selected_tokens, selected_indices
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _interpolate_sequence(
        self, x: Tensor, target_length: int
    ) -> Tuple[Tensor, None]:
        """Interpolate sequence to target length using different methods."""
        batch_size, seq_len, hidden_size = x.shape

        if self.compression_method == "learnable_interpolation":
            # Use learnable interpolation weights
            # Project each token to compression_size interpolation weights
            interp_weights = self.interpolation_proj(
                x
            )  # [batch, seq_len, compression_size]
            
            # If target_length is different from compression_size, we need to adjust
            if target_length < self.compression_size:
                # Use only the first target_length dimensions
                interp_weights = interp_weights[:, :, :target_length]
            elif target_length > self.compression_size:
                # Interpolate to get more dimensions
                interp_weights = F.interpolate(
                    interp_weights.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            interp_weights = F.softmax(
                interp_weights, dim=1
            )  # Normalize over sequence dimension

            # Weighted sum to create compressed representation
            compressed = torch.bmm(
                interp_weights.transpose(1, 2), x
            )  # [batch, target_length, hidden]

            # Add learned positional embeddings for compressed sequence
            # Handle case where target_length might be smaller than compression_size
            if target_length <= self.compression_size:
                compressed = compressed + self.compressed_pos_emb[:, :target_length, :]
            else:
                # If target_length > compression_size, we need to interpolate the positional embeddings
                pos_emb = F.interpolate(
                    self.compressed_pos_emb.transpose(1, 2),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                compressed = compressed + pos_emb

            return compressed, None

        elif self.compression_method == "linear_interpolation":
            # Simple linear interpolation between positions
            if target_length >= seq_len:
                # If target is larger, just return original (this shouldn't happen in compression)
                return x[:, :target_length], None

            # Create interpolation indices
            indices = torch.linspace(0, seq_len - 1, target_length, device=x.device)
            floor_indices = indices.floor().long()
            ceil_indices = torch.clamp(indices.ceil().long(), max=seq_len - 1)
            weights = indices - floor_indices.float()

            # Interpolate
            compressed = []
            for i in range(target_length):
                floor_val = x[:, floor_indices[i], :]
                ceil_val = x[:, ceil_indices[i], :]
                interpolated = floor_val * (1 - weights[i]) + ceil_val * weights[i]
                compressed.append(interpolated)

            compressed = torch.stack(
                compressed, dim=1
            )  # [batch, target_length, hidden]
            return compressed, None

        elif self.compression_method == "pooling":
            # Use adaptive pooling to compress sequence
            # Reshape for pooling: [batch * hidden, 1, seq_len]
            x_reshaped = x.transpose(1, 2).contiguous()  # [batch, hidden, seq_len]
            pooled = F.adaptive_avg_pool1d(
                x_reshaped, target_length
            )  # [batch, hidden, target_length]
            compressed = pooled.transpose(
                1, 2
            ).contiguous()  # [batch, target_length, hidden]
            return compressed, None

        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")

    def _interpolate_to_full_sequence(
        self, sparse_output: Tensor, selected_indices: Tensor, target_seq_len: int
    ) -> Tensor:
        """Interpolate sparse attention output back to full sequence length."""
        batch_size, k, hidden_size = sparse_output.shape
        device = sparse_output.device

        if self.selection_method == "sliding_window":
            # For sliding window, we can use more efficient linear interpolation
            # since tokens are consecutive
            full_output = torch.zeros(
                batch_size, target_seq_len, hidden_size, device=device
            )

            # Get the starting position of the sliding window
            start_pos = selected_indices[
                0, 0
            ].item()  # Same for all batches in sliding window

            # Direct copy for the selected region
            full_output[:, start_pos : start_pos + k] = sparse_output

            # Linear interpolation for positions before the window
            if start_pos > 0:
                # Linearly interpolate from zero to first selected token
                for i in range(start_pos):
                    alpha = (i + 1) / (start_pos + 1)
                    full_output[:, i] = alpha * sparse_output[:, 0]

            return full_output
        else:
            # Original method for top-k selection
            # Create position indices for the full sequence [batch_size, target_seq_len, 1]
            positions = (
                torch.arange(target_seq_len, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(batch_size, -1, -1)
            )

            # Expand selected indices for broadcasting [batch_size, 1, k]
            indices_expanded = selected_indices.unsqueeze(1).float()

            # Compute distances between all positions and selected indices [batch_size, target_seq_len, k]
            distances = torch.abs(positions - indices_expanded)

            # Compute weights using softmax over k dimension [batch_size, target_seq_len, k]
            weights = F.softmax(-distances * 5.0, dim=-1)

            # Perform weighted sum: [batch_size, target_seq_len, k] @ [batch_size, k, hidden_size]
            full_output = torch.bmm(weights, sparse_output)

            return full_output

    def _expand_compressed_sequence(
        self, compressed_output: Tensor, target_seq_len: int
    ) -> Tensor:
        """Expand compressed attention output back to full sequence length."""
        batch_size, compressed_len, hidden_size = compressed_output.shape
        device = compressed_output.device

        if self.compression_method == "learnable_interpolation":
            # Use learned expansion (inverse of compression)
            # Project to expansion weights, but only use the first target_seq_len dimensions
            full_expansion_weights = self.expansion_proj(
                compressed_output
            )  # [batch, compressed_len, max_seq_len]
            expansion_weights = full_expansion_weights[
                :, :, :target_seq_len
            ]  # [batch, compressed_len, target_seq_len]
            expansion_weights = F.softmax(
                expansion_weights, dim=1
            )  # Normalize over compressed dimension

            # Weighted sum to expand
            expanded = torch.bmm(
                expansion_weights.transpose(1, 2), compressed_output
            )  # [batch, target_seq_len, hidden]
            return expanded

        elif self.compression_method in ["linear_interpolation", "pooling"]:
            # Use linear interpolation to expand back
            if compressed_len >= target_seq_len:
                return compressed_output[:, :target_seq_len, :]

            # Create interpolation indices
            indices = torch.linspace(
                0, compressed_len - 1, target_seq_len, device=device
            )
            floor_indices = indices.floor().long()
            ceil_indices = torch.clamp(indices.ceil().long(), max=compressed_len - 1)
            weights = indices - floor_indices.float()

            # Interpolate
            expanded = []
            for i in range(target_seq_len):
                floor_val = compressed_output[:, floor_indices[i], :]
                ceil_val = compressed_output[:, ceil_indices[i], :]
                interpolated = floor_val * (1 - weights[i]) + ceil_val * weights[i]
                expanded.append(interpolated)

            expanded = torch.stack(expanded, dim=1)  # [batch, target_seq_len, hidden]
            return expanded

        else:
            raise ValueError(
                f"Unknown compression method for expansion: {self.compression_method}"
            )

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """Forward pass of soft sparse attention."""
        batch_size, seq_len, hidden_size = inputs.shape

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Step 1: Compress the input sequence
        compressed_tokens, selected_indices = self._compress_sequence(inputs)
        k = compressed_tokens.size(1)

        # Step 2: Compute attention in compressed space
        q = self.q_proj(compressed_tokens)  # [batch_size, k, hidden_size]
        k_compressed = self.k_proj(compressed_tokens)  # [batch_size, k, hidden_size]
        v = self.v_proj(compressed_tokens)  # [batch_size, k, hidden_size]

        # Reshape for multi-head attention
        q = q.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        k_compressed = k_compressed.view(
            batch_size, k, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores_compressed = torch.matmul(q, k_compressed.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed (in compressed space)
        if self.causal and selected_indices is not None:
            # Create causal mask based on original token positions (only for selection methods)
            # Use broadcasting to create mask efficiently
            # [batch_size, 1, k] > [batch_size, k, 1] -> [batch_size, k, k]
            pos_i = selected_indices.unsqueeze(-1)  # [batch_size, k, 1]
            pos_j = selected_indices.unsqueeze(1)  # [batch_size, 1, k]
            causal_mask = (
                pos_i <= pos_j
            )  # True where we should mask (can't attend to future)

            # Expand mask to match attention scores shape [batch_size, num_heads, k, k]
            causal_mask = causal_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            scores_compressed = scores_compressed.masked_fill(~causal_mask, -1e9)
        elif self.causal and self.selection_method == "interpolation":
            # For interpolation methods, apply standard causal mask
            causal_mask = torch.triu(
                torch.ones(k, k, device=inputs.device), diagonal=1
            ).bool()
            causal_mask = (
                causal_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, self.num_heads, -1, -1)
            )
            scores_compressed = scores_compressed.masked_fill(causal_mask, -1e9)

        # Compute attention weights and output
        weights = F.softmax(scores_compressed, dim=-1)
        attention_output = torch.matmul(weights, v)

        # Reshape back
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, k, hidden_size)
        )

        # Step 3: Expand back to full sequence length
        if self.selection_method == "interpolation":
            output = self._expand_compressed_sequence(attention_output, seq_len)
        else:
            output = self._interpolate_to_full_sequence(
                attention_output, selected_indices, seq_len
            )

        # Final output projection
        output = self.o_proj(output)

        return output, None, 0
