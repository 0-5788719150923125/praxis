"""Two activation branches, multiplied. No linear path.

WHAT THIS IS AND IS NOT. A GLU splits its up-projection in half and multiplies a
LINEAR branch by an ACTIVATED one: ``down(a * act(b))``. Half the channels have
never had a nonlinearity applied to them; they exist to scale the other half.
This module fills that empty slot with a second, DIFFERENT activation:

    down( act_gate(a) * act_value(b) )

so both halves are nonlinear and the two function classes multiply rather than
merely sit beside each other. It is therefore not a gated LINEAR unit any more,
which is why it is its own class rather than a flag on ``GatedLinearMLP``.

WHY MULTIPLICATION IS THE POINT. Concatenating two differently-activated halves
would couple them only through the NEXT matmul - additive, and a layer late.
Multiplying couples them pointwise and immediately: the gate branch decides
where the value branch is allowed to operate. With a periodic value activation
and a non-periodic gate, that is a non-periodic function steering a periodic
one, which is the mechanism this class exists to test. A harmonic mode that has
gone dormant can be revived by its gate instead of having to fix itself.

RELATION TO WHAT IS ALREADY HERE. ``Servant`` does a version of this one level
down, modulating a periodic FREQUENCY by a non-periodic ``tanh`` of live token
energy - and it has a known failure where that signal saturated and the
modulation silently became a constant. The lesson carried in here: a steering
branch needs to be observable. ``KolmogorovArnoldNetwork`` is the extreme of the
same idea (a learned activation per edge) and is already in the registry, so if
a fixed two-way split earns its keep, the scale-up path exists.

A NOTE ON THE 25/25/50 VARIANT. Splitting into two quarter-width activated
branches plus a half-width linear one would restore the GLU's linear path
alongside both function classes. Still not built HERE, but the question it asks
now has an answer path: ``peer_split`` (praxis/dense/peer.py, ``act_alt``) keeps
the GLU's linear branch and makes the GATE heterogeneous instead - half the
expert bank periodic, half swish. Same underlying hypothesis, that the model
wants both function classes AVAILABLE rather than COMPOSED, reached without
narrowing either branch. If that pays, a channel-split version of this class is
the obvious next step; if it does not, there is nothing here worth widening.
"""

from typing import Any, Optional, TypeVar

import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2CLS
from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class DualActivationMLP(BaseDense):
    """An MLP whose two projection halves carry different activations and are
    multiplied together."""

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,
        activation_gate: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            config: model config.
            activation: the VALUE branch's activation. Defaults to
                ``config.activation`` - so the value half matches whatever the
                rest of the model uses and the gate is the declared change.
            activation_gate: the GATE branch's activation. Defaults to ``gelu``:
                parameter-free and non-periodic, which is the contrast worth
                drawing in a model whose every other nonlinearity is periodic.
        """
        super().__init__()
        activation = activation or config.activation
        activation_gate = activation_gate or "gelu"

        # Same widths as GatedLinearMLP, so a swap is parameter-comparable
        # except for whatever the gate activation itself carries.
        down_size = int((4 / 3) * config.hidden_size)
        up_size = 2 * down_size

        self.up: nn.Linear = nn.Linear(config.hidden_size, up_size)
        self.act: nn.Module = ACT2CLS[activation]()
        self.act_gate: nn.Module = ACT2CLS[activation_gate]()
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.down: nn.Linear = nn.Linear(down_size, config.hidden_size)
        self._names = (activation_gate, activation)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        gate, value = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(self.act_gate(gate) * self.act(value)))

    def extra_repr(self) -> str:
        return f"gate={self._names[0]}, value={self._names[1]}"
