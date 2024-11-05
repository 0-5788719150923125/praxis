from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.attention import PraxisAttention
from praxis.modules.dense import PraxisGLU, PraxisMLP
from praxis.modules.peer import PraxisPEER
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.modules.smear import PraxisSMEAR

input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))

EXPERT_REGISTRY = {
    "mlp": PraxisMLP,
    "glu": PraxisGLU,
    "peer": PraxisPEER,
    "smear": PraxisSMEAR,
}

EXPERT_CONFIGS = {
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "smear": {"num_experts": 3},
    "glu": {},
    "mlp": {},
}


class PraxisExpert(nn.Module):
    """
    This class is a wrapper around the orchestration of both local and remote experts.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: AutoConfig,
        manager: Optional = False,
        block: nn.Module = False,
        router: nn.Module = False,
        is_remote=False,
    ):
        super().__init__()
        self.is_remote = is_remote
        self.block = (
            block
            if block
            else (manager.register_expert(config) if manager else PraxisBlock(config))
        )
        if config.sparse:
            self.router = router if router else PraxisMixtureOfDepths(config)

    def forward(self, inputs: Tensor, attention_mask: Tensor, use_router: bool):
        if self.is_remote:
            return self._remote_forward(inputs, attention_mask, use_router)
        else:
            return self._local_forward(inputs, attention_mask, use_router)

    def _local_forward(self, inputs: Tensor, attention_mask: Tensor, use_router: bool):
        aux_losses = []
        if use_router:
            hidden_states, aux_loss = self.router(self.block, inputs, attention_mask)
            aux_losses.append(aux_loss)
        else:
            hidden_states = self.block(inputs, attention_mask)
        return hidden_states, sum(aux_losses)

    def _remote_forward(self, inputs, attention_mask, use_router):
        # because we would otherwise break gradient flow
        residual = inputs
        aux_losses = []
        if use_router:
            hidden_states, aux_loss = self.router(
                self.block, inputs, attention_mask, safe_grad=True
            )
            aux_losses.append(aux_loss)
        else:
            # because hivemind cannot receive undefined arguments in the forward pass
            dummy_router_weights = torch.zeros_like(inputs)
            dummy_token_indices = torch.zeros_like(attention_mask, dtype=torch.int64)
            # because we do not backpropagate through remote experts
            with torch.no_grad():
                hidden_states = self.block(
                    inputs.to("cpu"),
                    attention_mask.to("cpu"),
                    dummy_router_weights.to("cpu"),
                    dummy_token_indices.to("cpu"),
                ).to(residual.device)
        # TODO: we could possibly add some differentiable noise here; perhaps as a penalty on slow experts?
        hidden_states = hidden_states + residual
        return hidden_states, sum(aux_losses)


@register_expert_class("hivemind_expert", input_shape)
class PraxisBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".

    When using Hivemind, there are certain limitations:

    1. All inputs to the `forward()` method must be Tensors.
    2. No inputs are allowed to be empty (None) types.
    3. All inputs must be of a constant shape.
    3. All inputs/outputs must be a part of the computation graph (i.e. returning detached aux_loss tensors is invalid).

    Essentially, Hivemind experts have static inputs/outputs - in contrast to the "dynamic" nature of Pytorch.
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
