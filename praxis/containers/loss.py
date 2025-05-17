import torch
from torch import Tensor


class LossContainer:
    def __init__(self):
        self.loss_dict = {"main": 0.0}

    def add_loss(self, key: str = "main", loss: Tensor = 0):
        if key not in self.loss_dict:
            self.loss_dict[key] = 0
        self.loss_dict[key] = self.loss_dict[key] + loss
        return self.loss_dict[key]

    def get_loss(self, key: str = "main"):
        return self.loss_dict[key]

    def get_loss_values(self):
        return list(self.loss_dict.values())
