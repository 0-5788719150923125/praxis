from typing import Union

import torch
from torch import Tensor


class LossContainer:
    def __init__(self, **initial_losses):
        """Initialize LossContainer with optional initial loss values.
        
        Args:
            **initial_losses: Key-value pairs of initial losses to add.
        """
        self.loss_dict = {"main": 0.0}
        for key, value in initial_losses.items():
            self.add_loss(key, value)

    def add_loss(self, key: str = "main", loss: Union[Tensor, float, int] = 0):
        # Handle scalar and tensor types
        if isinstance(loss, (int, float)):
            if loss == 0:
                return 0
        elif isinstance(loss, Tensor):
            if loss.item() == 0:
                return 0
        else:
            return 0
            
        if key not in self.loss_dict:
            self.loss_dict[key] = 0
        self.loss_dict[key] = self.loss_dict[key] + loss
        return self.loss_dict[key]

    def get_loss(self, key: str = "main"):
        return self.loss_dict[key]

    def get_loss_values(self):
        return list(self.loss_dict.values())
    
    def add_loss_container(self, other_container: "LossContainer"):
        """Merge another LossContainer's values into this one."""
        for key, value in other_container.loss_dict.items():
            if key not in self.loss_dict:
                self.loss_dict[key] = 0
            self.loss_dict[key] = self.loss_dict[key] + value
