"""Utility functions for language modeling heads."""

import torch
from typing import Optional, Tuple


def create_bidirectional_attention_masks(
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create attention masks for both forward and backward directions.
    
    Args:
        seq_length: Sequence length
        device: Device to create tensors on
        dtype: Data type for the masks
    
    Returns:
        Tuple of (forward_mask, backward_mask) with shape [1, 1, seq_length, seq_length]
    """
    # Forward causal mask (lower triangular)
    forward_mask = torch.tril(
        torch.ones(seq_length, seq_length, device=device, dtype=dtype)
    )
    
    # Backward anti-causal mask (upper triangular)
    backward_mask = torch.triu(
        torch.ones(seq_length, seq_length, device=device, dtype=dtype)
    )
    
    # Add batch and head dimensions
    forward_mask = forward_mask.unsqueeze(0).unsqueeze(0)
    backward_mask = backward_mask.unsqueeze(0).unsqueeze(0)
    
    return forward_mask, backward_mask


def create_attention_mask_from_padding(
    input_ids: torch.Tensor,
    pad_token_id: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create attention mask from padding tokens.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_length]
        pad_token_id: ID of the padding token
        dtype: Data type for the mask
    
    Returns:
        Attention mask with shape [batch_size, seq_length]
    """
    return (input_ids != pad_token_id).to(dtype)