import torch
from torch import Tensor, nn


class NaiveSummation(nn.Module):
    def forward(self, main_loss: Tensor, aux_loss: Tensor):
        return main_loss + aux_loss
