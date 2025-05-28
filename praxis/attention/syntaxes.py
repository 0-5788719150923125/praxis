import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache


class SyntaxesAttention(nn.Module):
    """
    Syntaxes attention with hierarchical compression based on importance sorting.

    This implementation follows the original author's insight about sorting being crucial
    for creating meaningful compressed representations. The key ideas are:

    1. **Importance Sorting**: Tokens are sorted by their importance scores (norm or learned),
       creating a natural hierarchy where early compressed slots capture the most important
       patterns and later slots capture progressively less important details.

    2. **Hierarchical Compression**: Instead of arbitrary compression, sorting creates a
       frequency-like decomposition similar to Fourier/wavelet transforms, where compressed
       tokens represent different "frequency bands" or importance tiers.

    3. **Full-to-Compressed Attention**: Unlike the original implementation that lost
       information during expansion, we now compute attention from all positions (queries)
       to the compressed representation (keys/values), preserving full sequence information
       while benefiting from the compressed attention patterns.

    This approach prevents model collapse by maintaining information flow while still
    reducing computational complexity from O(n²) to O(n*k) where k << n.
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
            # Note: expansion_proj was removed as we now use full-to-compressed attention
            # instead of compressing and expanding

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
    
    def _sort_by_importance(self, x: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """Sort tokens by importance scores (descending)."""
        # Sort scores and get indices
        sorted_scores, sort_indices = torch.sort(scores, dim=-1, descending=True)
        
        # Use torch.gather for efficient gathering
        batch_size, seq_len, hidden_size = x.shape
        sort_indices_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        sorted_x = torch.gather(x, dim=1, index=sort_indices_expanded)
        
        return sorted_x, sort_indices

    def _compress_sequence(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compress input sequence to smaller representation with sorting."""
        batch_size, seq_len, hidden_size = x.shape
        k = min(self.compression_size, seq_len)
        
        # Always compute importance scores for sorting
        scores = self._compute_token_scores(x)
        
        if self.selection_method == "interpolation":
            # Sort tokens by importance first
            sorted_x, sort_indices = self._sort_by_importance(x, scores)
            # Apply interpolation on sorted sequence
            compressed, _ = self._interpolate_sequence(sorted_x, k)
            return compressed, sort_indices
        elif self.selection_method == "top_k":
            # Get top-k tokens (already sorted by importance)
            sorted_x, sort_indices = self._sort_by_importance(x, scores)
            selected_tokens = sorted_x[:, :k, :]
            selected_indices = sort_indices[:, :k]
            return selected_tokens, selected_indices
        elif self.selection_method == "sliding_window":
            # For sliding window, still sort within the window
            start_idx = max(0, seq_len - k)
            
            # Get tokens in window
            window_tokens = x[:, start_idx:, :]
            window_scores = scores[:, start_idx:]
            
            # Sort within window
            sorted_window, local_sort_indices = self._sort_by_importance(window_tokens, window_scores)
            
            # Map back to global indices efficiently
            selected_indices = local_sort_indices + start_idx
            
            return sorted_window, selected_indices
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
            interp_weights = self.interpolation_proj(x)  # [batch, seq_len, compression_size]
            
            # If target_length is different from compression_size, we need to adjust
            if target_length < self.compression_size:
                # Use only the first target_length dimensions
                interp_weights = interp_weights[:, :, :target_length]
            elif target_length > self.compression_size:
                # Pad with zeros instead of interpolating for efficiency
                pad_size = target_length - self.compression_size
                padding = torch.zeros(batch_size, seq_len, pad_size, device=x.device)
                interp_weights = torch.cat([interp_weights, padding], dim=-1)
            
            # Normalize over sequence dimension
            interp_weights = F.softmax(interp_weights, dim=1)

            # Weighted sum using einsum for efficiency
            compressed = torch.einsum('bsk,bsh->bkh', interp_weights, x)

            # Add learned positional embeddings for compressed sequence
            if target_length <= self.compression_size:
                compressed = compressed + self.compressed_pos_emb[:, :target_length, :]
            else:
                # Repeat and interpolate positional embeddings
                repeat_factor = (target_length + self.compression_size - 1) // self.compression_size
                pos_emb_repeated = self.compressed_pos_emb.repeat(1, repeat_factor, 1)
                compressed = compressed + pos_emb_repeated[:, :target_length, :]

            return compressed, None

        elif self.compression_method == "linear_interpolation":
            # Simple linear interpolation between positions
            if target_length >= seq_len:
                # If target is larger, just return original (this shouldn't happen in compression)
                return x[:, :target_length], None

            # Vectorized interpolation
            indices = torch.linspace(0, seq_len - 1, target_length, device=x.device)
            floor_indices = indices.floor().long()
            ceil_indices = torch.clamp(indices.ceil().long(), max=seq_len - 1)
            weights = indices - floor_indices.float()

            # Gather all floor and ceil values at once
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
            floor_vals = x[batch_indices, floor_indices.unsqueeze(0)]  # [batch, target_length, hidden]
            ceil_vals = x[batch_indices, ceil_indices.unsqueeze(0)]   # [batch, target_length, hidden]
            
            # Vectorized interpolation
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, target_length, 1]
            compressed = floor_vals * (1 - weights) + ceil_vals * weights
            
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

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[Tensor, DynamicCache]] = None,
        block_ids: Optional[Tensor] = None,
        current_depth: int = 0,
    ) -> Tuple[Tensor, Optional[Union[Tensor, DynamicCache]], Union[int, float]]:
        """Forward pass of soft sparse attention with hierarchical compression."""
        batch_size, seq_len, hidden_size = inputs.shape

        if past_key_values is not None:
            raise NotImplementedError(
                "KV caching not yet supported for SyntaxesAttention"
            )

        # Step 1: Compress the input sequence (now with sorting)
        compressed_tokens, selected_indices = self._compress_sequence(inputs)
        k = compressed_tokens.size(1)

        # Step 2: Compute queries from full sequence, keys/values from compressed
        # This allows every position to attend to the compressed representation
        q_full = self.q_proj(inputs)  # [batch_size, seq_len, hidden_size]
        k_compressed = self.k_proj(compressed_tokens)  # [batch_size, k, hidden_size]
        v_compressed = self.v_proj(compressed_tokens)  # [batch_size, k, hidden_size]

        # Reshape for multi-head attention
        q_full = q_full.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_compressed = k_compressed.view(
            batch_size, k, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_compressed = v_compressed.view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: full queries attend to compressed keys
        # Shape: [batch_size, num_heads, seq_len, k]
        scores = torch.matmul(q_full, k_compressed.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            # Create a mask where each position can only attend to compressed tokens
            # that represent earlier positions in the original sequence
            if selected_indices is not None and self.selection_method != "interpolation":
                # For top-k and sliding window, use actual position indices
                # Expand dimensions for broadcasting
                query_positions = torch.arange(seq_len, device=inputs.device).view(1, 1, seq_len, 1)
                key_positions = selected_indices.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, k]
                
                # Can attend if query position >= key position
                causal_mask = query_positions >= key_positions
                scores = scores.masked_fill(~causal_mask, -1e9)
            else:
                # For interpolation, vectorized causal masking
                # Each query position can attend to compressed tokens up to its relative position
                query_positions = torch.arange(seq_len, device=inputs.device, dtype=torch.float32)
                # Calculate how many compressed tokens each position can attend to
                allowed_compressed = torch.minimum(
                    torch.tensor(k, device=inputs.device),
                    ((query_positions + 1) / seq_len * k).long() + 1
                )
                
                # Create mask using broadcasting
                compressed_positions = torch.arange(k, device=inputs.device).unsqueeze(0)
                mask = compressed_positions >= allowed_compressed.unsqueeze(1)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                scores = scores.masked_fill(mask, -1e9)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Attention mask shape: [batch, seq_len]
            # Scores shape: [batch, num_heads, seq_len, k]
            # We need to mask query positions, not key positions
            query_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            scores = scores.masked_fill(query_mask == 0, -1e9)

        # Compute attention weights and output
        weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, k]
        
        # Each position in the full sequence attends to compressed values
        attention_output = torch.matmul(weights, v_compressed)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, hidden_size)
        )

        # Final output projection
        output = self.o_proj(attention_output)

        return output, None, 0
