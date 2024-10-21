import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe import RemoteExpert
from hivemind.p2p import P2PDaemonError, P2PHandlerError
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisExpert
from praxis.orchestration.swarm import PraxisHivemind


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire decoder block (and all intermediate layers)
    in a single class.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        self.checkpoint_indices = self._checkpoint_strategy(
            config.memory_profile, config.num_layers
        )
        if config.hivemind:
            self.swarm = PraxisHivemind(config)
            self.experts = self.swarm.get_experts()
        else:
            self.experts = nn.ModuleList(
                [PraxisExpert(config) for _ in range(config.num_layers)]
            )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.experts)
        if self.shuffle:
            random.shuffle(experts)

        if hasattr(self, "swarm"):
            self.swarm._search_for_experts()

        hidden_states = inputs
        aux_losses = []

        for i, expert in enumerate(experts):
            use_router = True if self.sparse and i % 2 != 0 else False
            bit_tensor = torch.tensor([1 if use_router else 0], dtype=torch.bool)
            gradient_checkpointing = True if i in self.checkpoint_indices else False
            try:
                hidden_states = self._create_forward(
                    expert,
                    hidden_states,
                    attention_mask,
                    bit_tensor,
                    gradient_checkpointing,
                ).to(inputs.device)
                if hasattr(expert, "get_losses"):
                    aux_loss = expert.get_losses()
                    aux_losses.append(aux_loss)
            except P2PDaemonError as e:
                self.swarm.handle_failure(expert)
            # except Exception as e:
            #     print(e)

        return hidden_states, sum(aux_losses)

    def _checkpoint_strategy(self, strategy="speed", num_layers=0):
        if strategy == "aggressive":
            # every layer
            return [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            return [i for i in range(num_layers) if i % 4 == 0]
        else:
            # no gradient checkpointing
            return []

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor,
        bit_tensor: Tensor,
        gradient_checkpointing=False,
    ):
        def custom_forward(*inputs):
            return expert(*inputs)

        if isinstance(expert, RemoteExpert):
            hidden_states = hidden_states.to("cpu")
            attention_mask = attention_mask.to("cpu")
            bit_tensor = bit_tensor.to("cpu")

        if gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                bit_tensor,
                use_reentrant=False,
            )
        else:
            return custom_forward(hidden_states, attention_mask, bit_tensor)
