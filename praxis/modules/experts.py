from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly, PraxisScatter
from praxis.modules.kan import PraxisKAN
from praxis.modules.peer import PraxisPEER
from praxis.modules.recurrent import PraxisRecurrent
from praxis.modules.router import PraxisMixtureOfDepths
from praxis.utils import generate_decay_values

EXPERT_REGISTRY = {
    "glu": PraxisGLU,
    "kan": PraxisKAN,
    "mlp": PraxisMLP,
    "peer": PraxisPEER,
    "poly": PraxisPoly,
    "recurrent": PraxisRecurrent,
    "scatter": PraxisScatter,
}

EXPERT_CONFIGS = {
    "glu": {},
    "kan": {},
    "mlp": {},
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "poly": {},
    "recurrent": {},
    "scatter": {},
}


def get_expert_config(expert: str or dict):
    # Handle expert configuration
    if isinstance(expert, str):
        if expert not in EXPERT_CONFIGS:
            raise ValueError(f"Unknown expert type: {expert}")
        return {"type": expert, **EXPERT_CONFIGS[expert]}
    elif isinstance(expert, dict):
        return expert
    else:
        raise ValueError("Expert must be either a string or a dictionary")


class PraxisExpert(nn.Module):
    """
    This class is a wrapper around the orchestration of both local and remote experts.
    TODO: There is some unreliable routing in this class. We need to fix the local/remote parity, while
    also addressing the depth and state handling.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config: "AutoConfig",
        block: nn.Module = False,
        router: nn.Module = False,
        is_remote=False,
    ):
        super().__init__()
        self.sparse = config.sparse
        self.is_remote = is_remote
        self.router = router
        if config.sparse and not self.router:
            self.capacities = generate_decay_values(config.depth)
            self.router = PraxisMixtureOfDepths(config)
        self.block = block

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        past_key_values: Tensor,
        current_state: Tensor,
        current_depth: int,
        block_ids,
    ):
        # use_router = self.sparse and current_depth % 2 != 0
        use_router = self.sparse
        if self.is_remote:
            return self._remote_forward(inputs, attention_mask, use_router)
        else:
            return self._local_forward(
                inputs,
                current_state,
                attention_mask,
                past_key_values,
                use_router,
                current_depth,
                block_ids,
            )

    def _local_forward(
        self,
        inputs: Tensor,
        current_state: Tensor,
        attention_mask: Tensor,
        past_key_values: Tensor,
        use_router: bool,
        current_depth: int,
        block_ids=None,
    ):
        aux_losses = []
        if use_router:
            hidden_states, layer_kv, state_update, aux_loss = self.router(
                self.block,
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
                self.capacities[current_depth],
            )
            aux_losses.append(aux_loss)
        else:
            hidden_states, layer_kv, state_update, aux_loss = self.block(
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
        return hidden_states, layer_kv, state_update, sum(aux_losses)

    def _remote_forward(self, inputs, attention_mask, use_router: bool):
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
            # because we do not backpropagate through remote experts
            hidden_states = self.block(
                inputs,
                attention_mask,
                dummy_router_weights,
            )
        # TODO: we could possibly add some differentiable noise here; perhaps as a penalty on slow experts?
        hidden_states = hidden_states.to(residual.device) + residual
        return hidden_states, sum(aux_losses)
