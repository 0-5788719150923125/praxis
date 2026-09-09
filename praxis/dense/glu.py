import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ActivationSpec, build_activation
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class GatedLinearMLP(BaseDense):
    """
    A standard MLP, augmented with Gated Linear Units.

    A GLU multiplies a LINEAR branch by an ACTIVATED one, ``down(a * act(b))``,
    so half the channels never meet a nonlinearity - they exist to scale the
    other half. ``activation_value`` fills that empty slot with a second,
    different function, which makes both halves nonlinear and turns the product
    into two function classes multiplying rather than one steering a linear
    half. That is a real architectural change and it is why the arm exists
    (``dual_act``), but it is one PARAMETER here rather than a class of its own:
    the widths, the dropout placement and the down-projection are identical
    either way, so a subclass would have differed by a single call.

    WHY MULTIPLICATION IS THE POINT, when the slot is filled. Concatenating two
    differently-activated halves would couple them only through the NEXT matmul
    - additive, and a layer late. Multiplying couples them pointwise and
    immediately: one branch decides where the other is allowed to operate. With
    a periodic value activation and a non-periodic gate, that is a non-periodic
    function steering a periodic one.

    RELATION TO A MIXTURE. This is a second SLOT, not a second entry in one
    slot, so it is not what ``ActivationMixture`` does and the two compose:
    either slot can name a mixture, e.g. ``activation={"type": "mix_split",
    "values": [...]}``. ``Servant`` does a version of the steering idea one
    level down, modulating a periodic FREQUENCY by a non-periodic ``tanh`` of
    live token energy - and it has a known failure where that signal saturated
    and the modulation silently became a constant. The lesson carried here: a
    steering branch needs to be observable.
    """

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[ActivationSpec] = None,
        activation_value: Optional[ActivationSpec] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a GLU-based MLP module.

        Args:
            config: Configuration object with model parameters
            activation: the GATE branch's activation - a registry name or a
                ``{type, values}`` spec. Default: ``config.activation``.
            activation_value: the VALUE branch's activation. Default: none,
                i.e. the ordinary GLU's linear branch. Naming one here is the
                ``dual_act`` arm.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        activation = activation or config.activation

        # First calculate the target size after chunking (down projection input size)
        down_size = int((4 / 3) * config.hidden_size)
        # Double it for up projection to ensure chunks match
        up_size = 2 * down_size

        self.up: nn.Linear = nn.Linear(config.hidden_size, up_size)
        self.act: nn.Module = build_activation(activation, **kwargs)
        # Identity by default, so a plain GLU is byte-for-byte unchanged.
        self.act_value: nn.Module = (
            build_activation(activation_value) if activation_value else nn.Identity()
        )
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.down: nn.Linear = nn.Linear(down_size, config.hidden_size)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass through the GLU module.

        Args:
            inputs: Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after GLU processing
        """
        a, b = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(self.act_value(a) * self.act(b)))
