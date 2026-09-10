from typing import Union

import torch
from torch import Tensor


class LossContainer:
    def __init__(self, **initial_losses):
        """Initialize LossContainer with optional initial loss values.

        Args:
            **initial_losses: Key-value pairs of initial losses to add.
        """
        self.loss_dict = {"main": torch.tensor(0.0)}
        for key, value in initial_losses.items():
            self.add_loss(key, value)

    def get_loss(self, key: str = "main"):
        return self.loss_dict[key]

    def get_loss_values(self):
        return list(self.loss_dict.values())

    def get_named_losses(self):
        """``(names, values)`` in registration order.

        The values alone are what ``get_loss_values`` returns, and a strategy
        that weights per objective cannot use them: the key set is conditional
        (policies, MTP, the encoder's pending losses, a head's arm objectives),
        so position is not a stable identity for a term across steps.
        """
        return list(self.loss_dict.keys()), list(self.loss_dict.values())

    def add_loss(self, key: str = "main", loss: Union[Tensor, float, int] = 0):
        # Convert all loss values to tensors for consistency
        if isinstance(loss, (int, float)):
            loss_value = torch.tensor(loss, dtype=torch.float32)
        elif isinstance(loss, Tensor):
            loss_value = loss
        else:
            loss_value = torch.tensor(0.0, dtype=torch.float32)

        if key not in self.loss_dict:
            self.loss_dict[key] = torch.tensor(0.0, dtype=torch.float32)
        self.loss_dict[key] = self.loss_dict[key] + loss_value
        return self.loss_dict[key]

    def add_loss_container(self, other_container: "LossContainer"):
        """Merge another LossContainer's values into this one."""
        for key, value in other_container.loss_dict.items():
            if key not in self.loss_dict:
                self.loss_dict[key] = 0
            self.loss_dict[key] = self.loss_dict[key] + value

    def __contains__(self, key: str) -> bool:
        """Check if a loss key exists in this container."""
        return key in self.loss_dict
