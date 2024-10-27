from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor

from praxis import PraxisConfig
from praxis.activations import ACT2FN
from praxis.modules.attention import PraxisAttention
from praxis.modules.peer import PraxisPEER
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.modules.smear import PraxisSMEAR

# from praxis.orchestration.hivemind import PraxisSwarm


class PraxisExpert(nn.Module):
    def __init__(self, config: PraxisConfig, swarm):
        super().__init__()
        self.swarm = swarm
        self.expert = swarm.register_expert(config) if swarm else PraxisBlock(config)
        if config.sparse:
            self.router = PraxisMixtureOfDepths(config)

    def forward(self, inputs: Tensor, attention_mask: Tensor, use_router: bool):
        if use_router:
            hidden_states, aux_loss = self.router(self.expert, inputs, attention_mask)
        else:
            hidden_states = self.expert(inputs, attention_mask)
            aux_loss = 0
        return hidden_states, aux_loss


# input_shape = lambda batch_size, hid_dim: (
#     torch.empty((batch_size, 1, hid_dim)),
#     torch.empty((batch_size, 1)),
#     # torch.empty((1)),
# )
input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class HivemindExpert(nn.Module):
    """
    A Hivemind expert has certain limitations, which make it difficult to work with:
    1. All inputs to the `forward()` method must be Tensors.
    2. No inputs may be empty.
    3. All inputs/outputs must be a part of the computation graph (i.e. returning detached aux_loss tensors is invalid).
    Essentially, Hivemind experts must define static inputs/outputs - which negates
    the "dynamic" nature of Pytorch.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        # self.max_batch_size = 4 // TODO: will need to figure out how to handle the disparities in batch size/sequence length between experts
        self.expert = PraxisBlock(config)

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        hidden_states = self.expert(inputs, attention_mask)
        return hidden_states


class PraxisBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.mlp = EXPERT_REGISTRY[config.expert["type"]](config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ):
        residual = inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask, token_indices)
        outputs = self.dropout(outputs)
        outputs = outputs + residual
        residual = outputs
        normalized = self.mlp_norm(outputs)
        outputs = self.mlp(normalized)
        outputs = self.dropout(outputs)
        if torch.is_tensor(router_weights):
            outputs *= router_weights
        outputs = outputs + residual
        return outputs


class PraxisMLP(nn.Sequential):
    """
    A standard Multi-Layer Perceptron.
    """

    def __init__(self, config: PraxisConfig):
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

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.up = nn.Linear(config.num_dims, 8 * config.num_dims)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.down = nn.Linear(4 * config.num_dims, config.num_dims)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(a * self.act(b)))


EXPERT_REGISTRY = {
    "mlp": PraxisMLP,
    "glu": PraxisGLU,
    "peer": PraxisPEER,
    "smear": PraxisSMEAR,
}
