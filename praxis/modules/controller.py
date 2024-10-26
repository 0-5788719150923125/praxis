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

        # A single dict with tuple values for current/confidence
        self.expert_accuracies = {i: [0.0, 0.0] for i in range(max_num_experts)}
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()
        self.expert_to_idx = {}
        self.next_free_idx = 0

        # Predictor network
        hidden_size = config.num_dims
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(config.dropout),
            ACT2FN["prelu"],
            nn.Linear(hidden_size // 2, max_num_experts * 3),
        )

        # Initialize predictor weights
        # nn.init.normal_(self.predictor[-1].weight, std=0.01)
        # nn.init.constant_(self.predictor[-1].bias, 0.1)

        # Transition tracking buffer
        self.register_buffer(
            "transition_counts", torch.zeros(max_num_experts, max_num_experts)
        )

    def forward(
        self,
        experts: List[nn.Module],
        expert: nn.Module,
        expert_output: torch.Tensor,
        actual_index: int,
    ):
        depth = self.depth
        current_num_experts = len(experts)
        device = expert_output.device
        batch_size = expert_output.size(0)

        # Get all predictions at once and split
        logits = self.predictor(expert_output.mean(dim=1))
        current_logits, routing_logits, exit_logits = torch.split(
            logits, self.max_num_experts, dim=1
        )

        # Slice active experts
        current_logits = current_logits[:, :current_num_experts]
        routing_logits = routing_logits[:, :current_num_experts]
        exit_logits = exit_logits[:, :current_num_experts]

        # Get expert index and compute current loss
        expert_idx = self._get_expert_idx(expert)
        current_true_index = torch.full((batch_size,), expert_idx, device=device)
        current_loss = F.cross_entropy(current_logits, current_true_index)

        aux_loss = current_loss * self.loss_scale
        recommended_next = None
        exit_score = exit_logits.sigmoid().mean()

        if self.training:
            # Handle training mode
            next_expert = (
                experts[actual_index + 1] if actual_index < depth - 1 else None
            )

            if next_expert is not None:
                next_idx = self._get_expert_idx(next_expert)
                routing_true_index = torch.full((batch_size,), next_idx, device=device)
                aux_loss += (
                    F.cross_entropy(routing_logits, routing_true_index)
                    * self.loss_scale
                )
                self.transition_counts[expert_idx, next_idx] += 1

            # Compute exit loss
            layer_progress = actual_index / depth
            routing_confidence = (
                F.softmax(routing_logits, dim=-1).max(dim=-1)[0].mean().item()
            )
            blended_target = torch.full_like(
                exit_score,
                min(1.0, max(0.0, (layer_progress + routing_confidence) / 2.0)),
            )
            aux_loss += (
                F.binary_cross_entropy(exit_score, blended_target) * self.loss_scale
            )

        else:
            # Handle inference mode
            recommended_next = torch.mode(torch.argmax(routing_logits, dim=-1))[
                0
            ].item()
            self._update_tracking(
                expert_idx,
                current_logits,
                routing_logits,
                current_true_index,
                current_num_experts,
            )

        return aux_loss, recommended_next, exit_score

    def _update_tracking(
        self,
        expert_idx: int,
        current_logits: torch.Tensor,
        routing_logits: torch.Tensor,
        true_index: torch.Tensor,
        current_num_experts: int,
    ):
        # Current accuracy
        current_correct = (
            (torch.argmax(current_logits, dim=-1) == true_index).float().mean().item()
        )

        # Get routing prediction
        routing_pred = torch.argmax(routing_logits, dim=-1)
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Compute confidence based on both predicted route and historical transitions
        transition_probs = self.transition_counts[expert_idx, :current_num_experts]
        if transition_probs.sum() > 0:
            # Historical confidence
            transition_probs = transition_probs / transition_probs.sum()
            optimal_route = torch.argmax(transition_probs)
            historical_confidence = (
                (routing_pred == optimal_route).float().mean().item()
            )

            # Current confidence from softmax
            current_confidence = (
                routing_probs.gather(1, routing_pred.unsqueeze(1)).mean().item()
            )

            # Blend historical and current confidence
            route_confidence = (historical_confidence + current_confidence) / 2
        else:
            # If no transition history, use only current confidence
            route_confidence = (
                routing_probs.gather(1, routing_pred.unsqueeze(1)).mean().item()
            )

        # Normalize confidence
        self.active_experts = set(range(current_num_experts))
        random_confidence = 1.0 / current_num_experts
        relative_confidence = min(
            1.0,
            max(
                0.0, (route_confidence - random_confidence) / (1.0 - random_confidence)
            ),
        )

        # Update EMAs
        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies[expert_idx] = [current_correct, relative_confidence]
        else:
            self.expert_accuracies[expert_idx] = [
                self.decay * old + (1 - self.decay) * new
                for old, new in zip(
                    self.expert_accuracies[expert_idx],
                    [current_correct, relative_confidence],
                )
            ]

        self.update_counts[expert_idx] += 1

    def _get_expert_idx(self, expert: nn.Module) -> int:
        expert_id = id(expert)
        if expert_id not in self.expert_to_idx:
            self.expert_to_idx[expert_id] = self.next_free_idx
            self.next_free_idx += 1
        return self.expert_to_idx[expert_id]

    def get_expert_accuracy(self, expert_idx: int) -> dict:
        if expert_idx not in self.active_experts:
            return {"current": 0.0, "confidence": 0.0}
        return {
            "current": self.expert_accuracies[expert_idx][0],
            "confidence": self.expert_accuracies[expert_idx][1],
        }

    def get_mean_accuracy(self) -> dict:
        if not self.active_experts:
            return {"current": 0.0, "confidence": 0.0}

        accuracies = list(
            zip(*(self.expert_accuracies[i] for i in self.active_experts))
        )
        return {
            "current": sum(accuracies[0]) / len(self.active_experts),
            "confidence": sum(accuracies[1]) / len(self.active_experts),
        }

    def get_all_accuracies(self) -> dict:
        return {
            "current": {i: self.expert_accuracies[i][0] for i in self.active_experts},
            "confidence": {
                i: self.expert_accuracies[i][1] for i in self.active_experts
            },
        }
