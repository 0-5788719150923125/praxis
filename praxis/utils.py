import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_scaling(normalized_x, depth):
    """
    Apply depth-based scaling to already-normalized inputs to address the Curse of Depth.
    https://arxiv.org/abs/2502.05795
    """
    # Ensure depth is at least 1
    depth = max(1, depth)

    # Apply depth-based scaling: divide by sqrt(depth)
    depth_scaling_factor = 1.0 / math.sqrt(depth)
    scaled_x = normalized_x * depth_scaling_factor

    return scaled_x


def create_block_ids(
    input_ids: torch.LongTensor,
    special_tokens: Union[List[int], Tuple[int, ...], torch.LongTensor],
) -> torch.LongTensor:
    # Convert special tokens to tensor if needed
    if not isinstance(special_tokens, torch.Tensor):
        special_tokens = torch.tensor(
            special_tokens, device=input_ids.device, dtype=input_ids.dtype
        )

    # Create padding token mask
    padding_mask = torch.isin(input_ids, special_tokens)

    # Create boundaries only when transitioning FROM non-padding TO padding
    # Get previous token's padding status (shift padding mask right)
    prev_padding = torch.cat(
        [
            torch.zeros(
                (input_ids.size(0), 1), device=input_ids.device, dtype=torch.bool
            ),
            padding_mask[:, :-1],
        ],
        dim=1,
    )

    # Create block boundaries at first token and after non-padding to padding transitions
    block_boundaries = torch.cat(
        [
            torch.ones(
                (input_ids.size(0), 1), device=input_ids.device, dtype=torch.bool
            ),
            (~prev_padding[:, :-1]) & padding_mask[:, :-1],
        ],
        dim=1,
    )

    # Cumsum for block numbering
    block_ids = block_boundaries.cumsum(dim=1)

    return block_ids
