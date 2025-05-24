"""Base class for language modeling heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for language modeling heads.
    """

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize the base head.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_size={self.hidden_size}, vocab_size={self.vocab_size})"

    @abstractmethod
    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """
        Forward pass through the head.

        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden_size]
            **kwargs: Additional arguments

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        pass

    @property
    @abstractmethod
    def classifier(self) -> nn.Module:
        """
        Get the classifier module for loss computation.

        Returns:
            The linear layer used as classifier
        """
        pass
