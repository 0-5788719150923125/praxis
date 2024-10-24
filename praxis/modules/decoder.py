import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.controller import PraxisController
from praxis.modules.experts import PraxisExpert
from praxis.orchestration.hivemind import PraxisSwarm


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
            config.memory_profile, config.depth
        )
        self.remote_experts = []
        if config.hivemind:
            self.swarm = PraxisSwarm(config)
            self.local_experts = nn.ModuleList(self.swarm.active_local_experts)
            self.remote_experts = self.swarm.active_remote_experts
        else:
            self.local_experts = nn.ModuleList(
                [PraxisExpert(config) for _ in range(config.depth)]
            )
        self.use_autopilot = config.autopilot
        if self.use_autopilot:
            self.pilot = PraxisController(
                hidden_size=config.num_dims,
                max_num_experts=len(self.local_experts) * 3,
            )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.local_experts) + list(self.remote_experts)
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
                new_states = self._create_forward(
                    expert,
                    hidden_states,
                    attention_mask,
                    bit_tensor,
                    gradient_checkpointing,
                ).to(inputs.device)

                # Dead peers will return a zero tensor
                if hasattr(self, "swarm") and self._is_zero_tensor(new_states):
                    raise Exception("received a zero tensor; pruning expert")

                # Hivemind forces expert outputs to require gradients, so we retrieve dummy tensors differently
                if hasattr(expert, "retrieve_loss"):
                    aux_loss = expert.retrieve_loss()
                    aux_losses.append(aux_loss)

                # Predict the "true" index of each expert
                if self.use_autopilot:
                    _, aux_loss, next_pred = self.pilot(experts, expert, new_states)
                    aux_losses.append(aux_loss)

                # Commit to self
                hidden_states = new_states

            except Exception as e:
                # Prune dead peers
                if hasattr(self, "swarm"):
                    self.swarm.handle_failure(expert)
                    continue
                # Crash on unhandled exceptions
                raise Exception(e)

        return hidden_states, sum(aux_losses)

    def get_prediction_accuracies(self):
        """Return current prediction accuracies"""
        if self.use_autopilot:
            return {
                "mean": self.pilot.get_mean_accuracy(),
                "per_expert": self.pilot.get_all_accuracies(),
            }
        return None

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

        if hasattr(self, "swarm") and self.swarm.is_remote(expert):
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

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        """Check if a tensor is filled with zeros (within numerical tolerance)"""
        return torch.abs(tensor).max().item() < tolerance
