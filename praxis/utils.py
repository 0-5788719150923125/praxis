import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_alternating_values(
    size: int, interval: int = 1, capacity: float = 0.125
) -> List[float]:
    """
    Generate a list of alternating values between 1.0 and specified capacity.

    Args:
        size: Number of values to generate
        interval: Interval between capacity values
        capacity: Value to use at specified intervals (default: 0.125)

    Returns:
        List of alternating float values
    """
    result = [1.0] * size

    for i in range(1, size, interval + 1):
        for j in range(i, min(i + interval, size)):
            result[j] = capacity

    return result


def generate_decay_values(
    depth: int,
    reverse: bool = False,
    center: float = 0.5,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> List[float]:
    """
    Generate a list of S-shaped decaying values with adjustable center point and bounds

    Args:
        depth (int): Number of values to generate
        reverse (bool): If True, reverse the order of values
        center (float): Position of the center point (0.5 is middle, <0.5 shifts left, >0.5 shifts right)
                        Value should be between 0 and 1
        lower_bound (float): Minimum value for the decay (default: 0.0)
        upper_bound (float): Maximum value for the decay (default: 1.0)

    Returns:
        list: List of float values showing S-shaped decay
    """
    # Generate evenly spaced x values (adjusted range for S-shape)
    x = np.linspace(-6, 6, depth)

    # Calculate the shift needed based on the center parameter
    # When center = 0.5, shift = 0 (no shift)
    # When center < 0.5, shift is positive (shifts curve left)
    # When center > 0.5, shift is negative (shifts curve right)
    shift = (0.5 - center) * 12  # Scale by the range of x (-6 to 6 = 12)

    # Apply the shift to x values
    x = x + shift

    # Calculate S-shaped values using sigmoid function (0 to 1 range)
    base_values = 1 - (1 / (1 + np.exp(x)))

    # Scale values to the desired range [lower_bound, upper_bound]
    values = lower_bound + (upper_bound - lower_bound) * base_values

    # Convert to list and optionally reverse
    result = values.tolist()
    if reverse:
        result.reverse()

    return result


def generate_u_shape_values(
    depth: int,
    decay_point: float = 0.3,
    ramp_point: float = 0.7,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    steepness: float = 5.0,
) -> List[float]:
    """
    Generate a list of U-shaped values that start at upper_bound, decay to lower_bound,
    and then ramp back up to upper_bound.

    Args:
        depth (int): Number of values to generate
        decay_point (float): Position where the initial decay occurs (0.0-1.0)
        ramp_point (float): Position where the final ramp-up occurs (0.0-1.0)
        lower_bound (float): Minimum value in the U-shape (default: 0.0)
        upper_bound (float): Maximum value in the U-shape (default: 1.0)
        steepness (float): Controls how steep the decay and ramp are (default: 5.0)

    Returns:
        list: List of float values showing U-shaped pattern
    """
    # Create normalized positions from 0 to 1
    positions = np.linspace(0, 1, depth)
    values = np.zeros(depth)

    # We'll use a modified sigmoid approach that ensures values start and end at 1.0
    for i, pos in enumerate(positions):
        # For decay: Use a modified function that equals 1.0 at position 0
        # Adjusted decay function that starts at 1.0 when pos = 0
        if pos < decay_point:
            # Normalize position to 0-1 range within the decay region
            norm_pos = pos / decay_point if decay_point > 0 else 0
            # Use a function that starts at 1 and approaches 0
            decay_factor = (1 - norm_pos) ** steepness
        else:
            decay_factor = 0

        # For ramp: Use a modified function that equals 1.0 at position 1
        if pos > ramp_point:
            # Normalize position to 0-1 range within the ramp region
            norm_pos = (pos - ramp_point) / (1 - ramp_point) if ramp_point < 1 else 0
            # Use a function that ends at 1
            ramp_factor = norm_pos**steepness
        else:
            ramp_factor = 0

        # Combine factors and scale to bounds
        combined_factor = max(decay_factor, ramp_factor)
        values[i] = lower_bound + (upper_bound - lower_bound) * combined_factor

    return values.tolist()


def norm_scaling(
    normalized_x: Union[torch.Tensor, float], depth: int
) -> Union[torch.Tensor, float]:
    """
    Apply depth-based scaling to already-normalized inputs to address the Curse of Depth.
    https://arxiv.org/abs/2502.05795

    Args:
        normalized_x: Normalized input values (tensor or scalar)
        depth: Depth value for scaling factor calculation

    Returns:
        Scaled values with same type as input
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
