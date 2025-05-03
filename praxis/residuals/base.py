import random
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualConnection(nn.Module):
    """
    Base class for residual connections that connect hidden states across layers.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the residual connection.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        super().__init__()

    def connect_width(self, h: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Connect width dimension of hidden states.

        Args:
            h: Input tensor

        Returns:
            Tuple containing processed tensor and None
        """
        return h, None

    def connect_depth(self, mix_h: Tensor, h_o: Tensor, beta: Tensor) -> Tensor:
        """
        Connect depth dimension of hidden states.

        Args:
            mix_h: Mixed hidden state tensor
            h_o: Output hidden state tensor
            beta: Beta tensor for scaling

        Returns:
            Combined tensor after connection
        """
        return mix_h + h_o

    def format_state(self, h: Tensor) -> Tensor:
        """
        Format state for further processing.

        Args:
            h: Input tensor

        Returns:
            Formatted tensor
        """
        return h[..., 0, :] if h.dim() == 4 else h
