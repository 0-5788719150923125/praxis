from typing import OrderedDict
from torch import nn
from transformers import AutoConfig
from praxis.activations import ACT2FN


class PraxisMLP(nn.Sequential):
    """
    A standard Multi-Layer Perceptron.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(
            OrderedDict(
                [
                    ("up", nn.Linear(config.num_dims, 4 * config.num_dims)),
                    ("act", ACT2FN[config.activation]),
                    ("dropout", nn.Dropout(config.dropout)),
                    ("down", nn.Linear(4 * config.num_dims, config.num_dims)),
                ]
            )
        )


class PraxisGLU(nn.Module):
    """
    A standard MLP, augmented with a Gated Linear Units.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.up = nn.Linear(config.num_dims, int((8 / 3) * config.num_dims))
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.down = nn.Linear(int((4 / 3) * config.num_dims), config.num_dims)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(a * self.act(b)))
