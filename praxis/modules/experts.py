from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN
from praxis.modules.attention import PraxisAttention
from praxis.modules.peer import PraxisPEER
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.modules.smear import PraxisSMEAR

# from praxis.orchestration.hivemind import PraxisSwarm


class PraxisExpert(nn.Module):
    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig, manager: bool):
        super().__init__()
        self.manager = manager
        self.block = manager.register_expert(config) if manager else PraxisBlock(config)
        if config.sparse:
            self.router = PraxisMixtureOfDepths(config)

    def forward(self, inputs: Tensor, attention_mask: Tensor, use_router: bool):
        if use_router:
            hidden_states, aux_loss = self.router(self.block, inputs, attention_mask)
        else:
            dummy_router_weights = None
            dummy_token_indices = None
            if self.manager:
                dummy_router_weights = torch.zeros_like(inputs)
                dummy_token_indices = torch.zeros_like(
                    attention_mask, dtype=torch.int64
                )
            hidden_states = self.block(
                inputs, attention_mask, dummy_router_weights, dummy_token_indices
            )
            aux_loss = 0
        return hidden_states, aux_loss


input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class HivemindExpert(nn.Module):
    """
    A Hivemind expert has certain limitations, which make it difficult to work with:
    1. All inputs to the `forward()` method must be Tensors.
    2. No inputs may be empty (None) types.
    3. All inputs must be of a consistent shape.
    3. All inputs/outputs must be a part of the computation graph (i.e. returning detached aux_loss tensors is invalid).
    Essentially, Hivemind experts must define static inputs/outputs - negating the "dynamic" nature of Pytorch.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        # self.max_batch_size = 4 // TODO: will need to figure out how to handle the disparities in batch size/sequence length between experts
        self.block = PraxisBlock(config)

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)


class PraxisBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
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
        # this is a super hack because hivemind
        if torch.is_tensor(router_weights) and self._is_zero_tensor(router_weights):
            router_weights = None
        if torch.is_tensor(token_indices) and self._is_zero_tensor(token_indices):
            token_indices = None
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
            outputs = outputs + router_weights
        outputs = outputs + residual
        return outputs

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        """Check if a tensor is filled with zeros (within numerical tolerance)"""
        try:
            if tensor.dtype == torch.int64:
                return torch.all(tensor == 0).item()
            return torch.abs(tensor).max().item() < tolerance
        except Exception as e:
            return True


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
