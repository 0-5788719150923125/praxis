from typing import Any, Optional, TypeVar

import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class MultiLayerPerceptron(nn.Sequential):
    """A multi-layer perceptron mapping ``input_dim -> input_dim``.

    ``num_layers`` is the number of linear layers (default 2). Deeper stacks add
    ``hidden_dim``-wide layers, with an activation and dropout between linears.
    """

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        activation = activation or config.activation
        input_dim = input_dim or config.hidden_size
        hidden_dim = hidden_dim or input_dim * 4
        num_layers = max(2, num_layers)

        widths = [input_dim] + [hidden_dim] * (num_layers - 1) + [input_dim]
        layers: list = []
        for i in range(num_layers):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < num_layers - 1:
                layers.append(ACT2CLS[activation]())
                layers.append(nn.Dropout(config.dropout))

        super().__init__(*layers)

    def init_weights(
        self, in_std: float = 0.02, out_std: Optional[float] = None
    ) -> None:
        """Normal-init the first and last linear weights (e.g. a depth-scaled
        output projection), without callers reaching into layer internals."""
        linears = [m for m in self if isinstance(m, nn.Linear)]
        nn.init.normal_(linears[0].weight, mean=0.0, std=in_std)
        nn.init.normal_(
            linears[-1].weight, mean=0.0, std=in_std if out_std is None else out_std
        )

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return super().forward(inputs)
