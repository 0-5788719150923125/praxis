from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.dense import PraxisGLU, PraxisMLP
from praxis.modules.peer import PraxisPEER
from praxis.modules.smear import PraxisSMEAR
from praxis.modules.router import PraxisMixtureOfDepths

EXPERT_REGISTRY = {
    "mlp": PraxisMLP,
    "glu": PraxisGLU,
    "peer": PraxisPEER,
    "smear": PraxisSMEAR,
}

EXPERT_CONFIGS = {
    "mlp": {},
    "glu": {},
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "smear": {"num_experts": 3},
}


class PraxisExpert(nn.Module):
    """
    This class is a wrapper around the orchestration of both local and remote experts.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: AutoConfig,
        block: nn.Module = False,
        router: nn.Module = False,
        is_remote=False,
    ):
        super().__init__()
        self.sparse = config.sparse
        self.is_remote = is_remote
        self.block = block
        if config.sparse:
            self.router = router if router else PraxisMixtureOfDepths(config)

    def forward(self, inputs: Tensor, attention_mask: Tensor, current_depth: int):
        use_router = True if self.sparse and current_depth % 2 != 0 else False
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
        inputs = inputs.to("cpu")
        attention_mask = attention_mask.to("cpu")
        if use_router:
            hidden_states, aux_loss = self.router(self.block, inputs, attention_mask)
            aux_losses.append(aux_loss)
        else:
            # because hivemind cannot receive undefined arguments in the forward pass
            dummy_router_weights = torch.zeros_like(inputs)
            dummy_token_indices = torch.zeros_like(attention_mask, dtype=torch.int64)
            # because we do not backpropagate through remote experts
            hidden_states = self.block(
                inputs,
                attention_mask,
                dummy_router_weights,
                dummy_token_indices,
            )
        # TODO: we could possibly add some differentiable noise here; perhaps as a penalty on slow experts?
        hidden_states = hidden_states.to(residual.device) + residual
        return hidden_states, sum(aux_losses)
