from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from praxis import PraxisConfig
from praxis.activations import ACT2FN


class PraxisController(nn.Module):
    """
    This controller implements an expert-prediction mechanism, which trains a small
    router to intelligently route through layers in the network. It also implements
    an early-exit strategy, inspired by CALM:
    https://arxiv.org/abs/2207.07061
    """

    def __init__(self, config: PraxisConfig, max_num_experts):
        super().__init__()
        self.depth = config.depth
        self.max_num_experts = max_num_experts
        self.decay = 0.99
        self.loss_scale = 0.01

        # Early exit logic
        self.calm = True
        # especially important during training, when the threshold is an average across all examples.
        self.exit_threshold = 0.7

        # Simplify tracking to just accuracy
        self.expert_accuracies = {i: 0.0 for i in range(max_num_experts)}
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()
        self.expert_to_idx = {}
        self.next_free_idx = 0

        # Predictor network
        hidden_size = config.num_dims
        self.prism = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(config.dropout),
            ACT2FN["relu"],
            nn.Linear(hidden_size // 2, max_num_experts * 4),
        )

        # Initialize predictor weights
        nn.init.normal_(self.prism[-1].weight, std=0.01)
        nn.init.constant_(self.prism[-1].bias, 0.1)

        self.map = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.merges = nn.Linear(hidden_size, 1)
        self.scale = nn.LayerNorm(hidden_size)
        self.state = 0

    def forward(self, experts, expert, expert_output, actual_index):
        depth = self.depth
        current_num_experts = len(experts)
        device = expert_output.device
        batch_size = expert_output.size(0)

        # Get all predictions at once and split
        logits = self.prism(expert_output.mean(dim=1))
        current_logits, routing_logits, exit_logits, gating_logits = torch.split(
            logits, self.max_num_experts, dim=1
        )

        # Slice active experts
        current_logits = current_logits[:, :current_num_experts]
        routing_logits = routing_logits[:, :current_num_experts]
        exit_logits = exit_logits[:, :current_num_experts]
        gating_logits = gating_logits[:, :current_num_experts]

        should_exit = False
        if self.calm:
            # Compute the early exit score
            exit_score = exit_logits.sigmoid().mean()
            should_exit = exit_score > self.exit_threshold

        self._accumulate_state(
            expert_output,
            actual_index,
            current_logits,
            routing_logits,
            exit_logits,
            gating_logits,
        )

        # Get expert index and compute current loss
        expert_idx = self._get_expert_idx(expert)
        current_true_index = torch.full((batch_size,), expert_idx, device=device)
        current_loss = F.cross_entropy(current_logits, current_true_index)
        aux_loss = current_loss * self.loss_scale

        recommended_next = None
        if self.training:
            # Handle training mode
            next_expert = (
                experts[actual_index + 1] if actual_index < depth - 1 else None
            )

            if next_expert is not None:
                next_idx = self._get_expert_idx(next_expert)
                routing_true_index = torch.full((batch_size,), next_idx, device=device)
                routing_loss = F.cross_entropy(routing_logits, routing_true_index)
                aux_loss += routing_loss * self.loss_scale

                # Simple exit loss based on progress
                if self.calm:
                    exit_target = torch.full_like(exit_score, actual_index / depth)
                    aux_loss += (
                        F.binary_cross_entropy(exit_score, exit_target)
                        * self.loss_scale
                    )
        else:
            # Handle inference mode
            recommended_next = torch.mode(torch.argmax(routing_logits, dim=-1))[
                0
            ].item()
            self._update_tracking(
                expert_idx,
                current_logits,
                current_true_index,
                current_num_experts,
            )

        return aux_loss, recommended_next, should_exit

    # return expert_contribution
    def _accumulate_state(
        self,
        expert_output,
        actual_index,
        current_logits,
        routing_logits,
        exit_logits,
        gating_logits,
    ):

        B, S, H = expert_output.size()
        num_experts = current_logits.size(1)

        # Add dimension for matmul
        logits = current_logits.unsqueeze(-1)  # [B, E, 1]
        routes = routing_logits.unsqueeze(-1)  # [B, E, 1]
        exits = exit_logits.unsqueeze(-1)  # [B, E, 1]
        gates = gating_logits.unsqueeze(-1)  # [B, E, 1]

        # Combine current and routing predictions
        entries = torch.matmul(logits * routes, self.map)  # [B, E, H]

        # Normalize by exit predictions and reduce across experts
        reduced = torch.var(entries / (exits + 1e-6), dim=1, keepdim=True)  # [B, 1, H]

        # Apply gating
        gated = torch.sigmoid(gates.mean(dim=1, keepdim=True)) * reduced  # [B, 1, H]

        # Add positional modulation
        pos = torch.arange(S, device=expert_output.device).float()
        pos = pos.view(1, -1, 1) / S  # Normalize to [0,1]
        pos_signal = torch.sin(2.0 * torch.pi * pos) * self.position

        expert_contribution = expert_output * gated + pos_signal  # [B, S, H]

        # Smooth accumulation using exponential moving average
        if actual_index == 0:
            self.state = expert_contribution
        else:
            alpha = torch.sigmoid(torch.tensor(actual_index / self.depth))
            self.state = (1 - alpha) * self.state + alpha * expert_contribution

    def merge_states(self, inputs):
        # Compute adaptive mixing ratio
        gate = torch.sigmoid(self.merges(inputs))
        # Normalize accumulated state
        normalized = self.scale(self.state)
        # Gated addition with residual connection
        return inputs + gate * F.relu(normalized)

    def _update_tracking(
        self,
        expert_idx: int,
        current_logits: torch.Tensor,
        current_true_index: torch.Tensor,
        current_num_experts: int,
    ):
        # Current accuracy only
        current_correct = (
            (torch.argmax(current_logits, dim=-1) == current_true_index)
            .float()
            .mean()
            .item()
        )

        # Update tracking
        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies[expert_idx] = current_correct
        else:
            self.expert_accuracies[expert_idx] = (
                self.decay * self.expert_accuracies[expert_idx]
                + (1 - self.decay) * current_correct
            )

        self.update_counts[expert_idx] += 1
        self.active_experts.add(expert_idx)

    def _get_expert_idx(self, expert: nn.Module) -> int:
        expert_id = id(expert)
        if expert_id not in self.expert_to_idx:
            self.expert_to_idx[expert_id] = self.next_free_idx
            self.next_free_idx += 1
        return self.expert_to_idx[expert_id]

    def get_expert_accuracy(self, expert_idx: int) -> float:
        if expert_idx not in self.active_experts:
            return 0.0
        return self.expert_accuracies[expert_idx]

    def get_mean_accuracy(self) -> float:
        if not self.active_experts:
            return 0.0
        return sum(self.expert_accuracies[i] for i in self.active_experts) / len(
            self.active_experts
        )

    def get_all_accuracies(self) -> dict:
        return {i: self.expert_accuracies[i] for i in self.active_experts}
