import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NMDA(nn.Module):
    """
    Implements NMDA - an activation function which mimics N-methyl-D-aspartic acid receptors (NMDAR)
    in the brain. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory,
    thus enhancing a process that is similar to memory consolidation in the mammalian brain.
    https://openreview.net/forum?id=vKpVJxplmB
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, x: Tensor) -> Tensor:
        alpha = 0 if self.alpha <= 0 else self.alpha.log()
        return x * torch.sigmoid(self.beta * x - alpha)
