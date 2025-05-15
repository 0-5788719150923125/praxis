import torch
from torch import Tensor, nn


class UncertaintyWeighted(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize log variances
        self.log_var_main = nn.Parameter(torch.zeros(1))
        self.log_var_aux = nn.Parameter(torch.zeros(1))

    def forward(self, main_loss: Tensor, aux_loss: Tensor):
        # Get weights from uncertainties
        precision_main = torch.exp(-self.log_var_main)
        precision_aux = torch.exp(-self.log_var_aux)

        # Calculate weighted loss
        loss = (
            precision_main * main_loss
            + 0.5 * self.log_var_main
            + precision_aux * aux_loss
            + 0.5 * self.log_var_aux
        )

        return loss
