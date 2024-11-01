import math

import torch
import torch.nn.functional as F
from torch import nn


class NMDA(nn.Module):
    """
    Implements NMDA - an activation function which mimics N-methyl-D-aspartic acid receptors (NMDAR)
    in the brain. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory
    in transformers, thus enhancing a process that is similar to memory consolidation in the mammalian brain.
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html
    """

    def __init__(self):
        super().__init__()
        self.alpha = 1.0
        self.beta = 1.0
        self.a = math.log(self.alpha)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x - self.a)
