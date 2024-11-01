import math

import torch
import torch.nn.functional as F
from torch import nn


class NMDA(nn.Module):
    """
    Implements NMDA - an activation function which mimics N-methyl-D-aspartic acid receptors (NMDAR)
    in the brain. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory,
    thus enhancing a process that is similar to memory consolidation in the mammalian brain.
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.alpha, 1.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x):
        alpha = 0 if self.alpha <= 0 else self.alpha.log()
        return x * torch.sigmoid(self.beta * x - alpha)
