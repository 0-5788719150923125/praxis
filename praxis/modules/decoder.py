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
        self.debug = config.debug
        self.depth = config.depth
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        self.remote_experts = []
        if config.hivemind:
            self.swarm = PraxisSwarm(config)
            self.local_experts = nn.ModuleList(self.swarm.active_local_experts)
            self.remote_experts = self.swarm.active_remote_experts
        else:
            self.local_experts = nn.ModuleList(
                [PraxisExpert(config) for _ in range(self.depth)]
            )
        self.use_autopilot = config.autopilot
        if self.use_autopilot:
            self.copilot = PraxisController(config, len(self.local_experts) * 3)
        self._define_checkpoints(config.memory_profile, self.depth)

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        experts = list(self.local_experts) + list(self.remote_experts)
        original_order = experts.copy()
        if self.shuffle:
            random.shuffle(experts)

        if hasattr(self, "swarm"):
            self.swarm._search_for_experts()

        hidden_states = inputs
        aux_losses = []
        next_expert_idx = None

        first_expert_idx = original_order.index(experts[0])
        route = [str(first_expert_idx)]

        exit_score = 0

        for i in range(self.depth):
            use_router = True if self.sparse and i % 2 != 0 else False
            bit_tensor = torch.tensor([1 if use_router else 0], dtype=torch.bool)
            gradient_checkpointing = True if i in self.checkpoint_indices else False
            try:
                expert = experts[i]
                if not self.training and next_expert_idx is not None:
                    expert = experts[next_expert_idx]
                    route.append(str(next_expert_idx))

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
                    aux_loss, next_expert_idx, exit_score = self.copilot(
                        experts, expert, new_states, i
                    )
                    aux_losses.append(aux_loss)
                    threshold = 0.66
                    should_exit = exit_score > threshold
                    if should_exit:
                        break

                # Commit to self
                hidden_states = new_states

            except Exception as e:
                # Prune dead peers
                if hasattr(self, "swarm"):
                    self.swarm.handle_failure(expert)
                    continue
                # Crash on unhandled exceptions
                raise Exception(e)

        if self.debug and not self.training:
            print(
                f"DEBUG: Routing through experts {' -> '.join(route)} (exit score: {exit_score.item():.4f})"
            )

        return hidden_states, sum(aux_losses)

    def get_prediction_accuracies(self):
        """Return current prediction accuracies"""
        if self.use_autopilot:
            return {
                "mean": self.copilot.get_mean_accuracy(),
                "per_expert": self.copilot.get_all_accuracies(),
            }
        return None

    def _define_checkpoints(self, strategy="speed", num_layers=0):
        self.checkpoint_indices = []  # no gradient checkpointing
        if strategy == "aggressive":
            # every layer
            self.checkpoint_indices = [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            self.checkpoint_indices = [i for i in range(num_layers) if i % 4 == 0]

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
