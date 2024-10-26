from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from praxis import PraxisConfig
from praxis.activations import ACT2FN


class PraxisController(nn.Module):
    """
    This controller implements an expert-prediction mechanism, which trains
    the network to learn how to intelligently route through layers in the network.
    It also implements an early-exit strategy, inspired by CALM:
    https://arxiv.org/abs/2207.07061
    """

    def __init__(self, config: PraxisConfig, max_num_experts):
        super().__init__()
        hidden_size = config.num_dims
        self.max_num_experts = max_num_experts

        self.decay = 0.99
        self.loss_scale = 0.01

        # Expert mapping dictionary: id(expert) -> index
        self.expert_to_idx = {}
        self.next_free_idx = 0

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            ACT2FN["sinlu"],
            nn.Linear(hidden_size // 2, max_num_experts * 3),
        )
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.predictor[-1].bias, 0.1)

        # Keep transition tracking
        self.register_buffer(
            "transition_counts",
            torch.zeros(max_num_experts, max_num_experts),  # [from_expert, to_expert]
        )

        self.expert_accuracies = {
            "current": {i: 0.0 for i in range(max_num_experts)},
            "confidence": {i: 0.0 for i in range(max_num_experts)},
        }
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()

    def forward(
        self,
        experts: List[nn.Module],
        expert: nn.Module,
        expert_output: torch.Tensor,
        actual_index: int,
    ):
        current_num_experts = len(experts)
        batch_size = expert_output.size(0)

        # Pool hidden states
        pooled = torch.mean(expert_output, dim=1)

        # Get predictions
        full_logits = self.predictor(pooled)
        current_logits, routing_logits, exit_logits = torch.split(
            full_logits, self.max_num_experts, dim=1
        )

        # Slice to active experts
        current_logits = current_logits[:, :current_num_experts]
        routing_logits = routing_logits[:, :current_num_experts]
        exit_logits = exit_logits[:, :current_num_experts]
        exit_score = exit_logits.sigmoid().mean()

        # Get stable index for current expert
        expert_idx = self._get_expert_idx(expert)
        current_true_index = torch.full(
            (batch_size,), expert_idx, device=current_logits.device
        )
        current_loss = F.cross_entropy(current_logits, current_true_index)

        # Get the next expert, if one exists
        next_expert = (
            experts[actual_index + 1]
            if actual_index < len(experts) - 1 and self.training
            else None
        )

        # During training: learn from randomly selected next expert
        routing_loss = 0
        if self.training and next_expert is not None:
            next_idx = self._get_expert_idx(next_expert)
            routing_true_index = torch.full(
                (batch_size,), next_idx, device=routing_logits.device
            )
            routing_loss = F.cross_entropy(routing_logits, routing_true_index)

            # Update transition counts using stable indices
            self.transition_counts[expert_idx, next_idx] += 1

        # Combined loss with scaling
        aux_loss = (current_loss + routing_loss) * self.loss_scale

        # During inference: recommend next expert
        recommended_next = None
        if not self.training:
            batch_predictions = torch.argmax(routing_logits, dim=-1)
            recommended_next = torch.mode(batch_predictions)[0].item()

            # Update accuracy tracking
            self._update_tracking(
                expert_idx=expert_idx,
                current_logits=current_logits,
                routing_logits=routing_logits,
                true_index=current_true_index,
                current_num_experts=current_num_experts,
                from_expert=expert_idx,
            )

        if self.training:
            # Option 1: Encourage later exits with layer-dependent target
            layer_progress = actual_index / len(experts)  # 0 to 1

            # Option 2: Get routing confidence from softmax probabilities
            routing_probs = F.softmax(routing_logits, dim=-1)
            routing_confidence = (
                routing_probs.max(dim=-1)[0].mean().item()
            )  # Convert to scalar with .item()

            # Blend both signals - we want exits to be more likely when:
            # 1. We're deeper in the network (layer_progress)
            # 2. We're confident about routing (routing_confidence)
            blended_target = (layer_progress + routing_confidence) / 2.0

            # Clamp to ensure we stay in [0,1]
            blended_target = max(
                0.0, min(1.0, blended_target)
            )  # Use regular min/max for scalars

            # Create target tensor and compute loss
            exit_target = torch.full_like(exit_score, blended_target)
            exit_loss = F.binary_cross_entropy(exit_score, exit_target)

            # Add to total loss
            aux_loss = aux_loss + exit_loss * self.loss_scale

        return aux_loss, recommended_next, exit_score

    def _update_tracking(
        self,
        expert_idx: int,
        current_logits: torch.Tensor,
        routing_logits: torch.Tensor,
        true_index: torch.Tensor,
        current_num_experts: int,
        from_expert: int,
    ):
        # Current expert identification accuracy
        current_pred = torch.argmax(current_logits, dim=-1)
        current_correct = (current_pred == true_index).float().mean().item()

        # Measure routing confidence/decisiveness
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Get the confidence score for the chosen route
        chosen_route = torch.argmax(routing_logits, dim=-1)
        route_confidence = (
            routing_probs.gather(1, chosen_route.unsqueeze(1)).mean().item()
        )

        # Normalize confidence
        self.active_experts = set(range(current_num_experts))
        random_confidence = 1.0 / current_num_experts
        normalized_confidence = (route_confidence - (random_confidence)) / (
            1.0 - (random_confidence)
        )
        relative_confidence = max(
            0.0, min(1.0, normalized_confidence)
        )  # Clamp to [0, 1]

        # Update EMAs
        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies["current"][expert_idx] = current_correct
            self.expert_accuracies["confidence"][expert_idx] = relative_confidence
        else:
            self.expert_accuracies["current"][expert_idx] = (
                self.decay * self.expert_accuracies["current"][expert_idx]
                + (1 - self.decay) * current_correct
            )
            self.expert_accuracies["confidence"][expert_idx] = (
                self.decay * self.expert_accuracies["confidence"][expert_idx]
                + (1 - self.decay) * relative_confidence
            )

        self.update_counts[expert_idx] += 1

    def _get_expert_idx(self, expert: nn.Module) -> int:
        """Get or assign stable index for an expert"""
        expert_id = id(expert)
        if expert_id not in self.expert_to_idx:
            self.expert_to_idx[expert_id] = self.next_free_idx
            self.next_free_idx += 1
        return self.expert_to_idx[expert_id]

    def get_expert_accuracy(self, expert_idx: int) -> dict:
        """Returns both current and confidence accuracies for given expert"""
        if expert_idx not in self.active_experts:
            return {"current": 0.0, "confidence": 0.0}
        return {
            "current": self.expert_accuracies["current"][expert_idx],
            "confidence": self.expert_accuracies["confidence"][expert_idx],
        }

    def get_mean_accuracy(self) -> dict:
        """Returns mean accuracy for both current and confidence predictions"""
        if not self.active_experts:
            return {"current": 0.0, "confidence": 0.0}

        current_accs = [
            self.expert_accuracies["current"][i] for i in self.active_experts
        ]
        confidence_accs = [
            self.expert_accuracies["confidence"][i] for i in self.active_experts
        ]

        return {
            "current": sum(current_accs) / len(self.active_experts),
            "confidence": sum(confidence_accs) / len(self.active_experts),
        }

    def get_all_accuracies(self) -> dict:
        """Returns all accuracies for active experts only"""
        return {
            "current": {
                i: self.expert_accuracies["current"][i] for i in self.active_experts
            },
            "confidence": {
                i: self.expert_accuracies["confidence"][i] for i in self.active_experts
            },
        }
