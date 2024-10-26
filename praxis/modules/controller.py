from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class PraxisController(nn.Module):
    def __init__(self, hidden_size, max_num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_num_experts = max_num_experts

        # Expert mapping dictionary: id(expert) -> index
        self.expert_to_idx = {}
        self.next_free_idx = 0

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_num_experts * 2),
        )
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.predictor[-1].bias, 0.1)

        # Keep transition tracking
        self.register_buffer(
            "transition_counts",
            torch.zeros(max_num_experts, max_num_experts),  # [from_expert, to_expert]
        )

        self.decay = 0.99
        self.expert_accuracies = {
            "current": {i: 0.0 for i in range(max_num_experts)},
            "routing": {i: 0.0 for i in range(max_num_experts)},
        }
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()

    def _get_expert_idx(self, expert: nn.Module) -> int:
        """Get or assign stable index for an expert"""
        expert_id = id(expert)
        if expert_id not in self.expert_to_idx:
            self.expert_to_idx[expert_id] = self.next_free_idx
            self.next_free_idx += 1
        return self.expert_to_idx[expert_id]

    def forward(
        self,
        experts: List[nn.Module],
        expert: nn.Module,
        expert_output: torch.Tensor,
        next_expert: Optional[nn.Module] = None,
    ):
        current_num_experts = len(experts)
        batch_size = expert_output.size(0)

        # Pool hidden states
        pooled = (
            torch.mean(expert_output, dim=1)
            if len(expert_output.shape) == 3
            else expert_output
        )

        # Get predictions
        full_logits = self.predictor(pooled)
        current_logits, routing_logits = torch.split(
            full_logits, self.max_num_experts, dim=1
        )

        # Slice to active experts
        current_logits = current_logits[:, :current_num_experts]
        routing_logits = routing_logits[:, :current_num_experts]

        # Get stable index for current expert
        expert_idx = self._get_expert_idx(expert)
        current_true_index = torch.full(
            (batch_size,), expert_idx, device=current_logits.device
        )
        current_loss = F.cross_entropy(current_logits, current_true_index)

        # During training: learn from randomly selected next expert
        if self.training and next_expert is not None:
            next_idx = self._get_expert_idx(next_expert)
            routing_true_index = torch.full(
                (batch_size,), next_idx, device=routing_logits.device
            )
            routing_loss = F.cross_entropy(routing_logits, routing_true_index)

            # Update transition counts using stable indices
            self.transition_counts[expert_idx, next_idx] += 1
        else:
            routing_loss = torch.tensor(0.0, device=current_logits.device)

        # Combined loss with scaling
        loss_scale = 0.01
        aux_loss = (current_loss + routing_loss) * loss_scale

        # During inference: recommend next expert
        recommended_next = None
        if not self.training:
            batch_predictions = torch.argmax(routing_logits, dim=-1)
            recommended_next = torch.mode(batch_predictions)[0].item()

            # Update accuracy tracking
            self.update_tracking(
                expert_idx=expert_idx,
                current_logits=current_logits,
                routing_logits=routing_logits,
                true_index=current_true_index,
                current_num_experts=current_num_experts,
                from_expert=expert_idx,
            )

        return aux_loss, recommended_next

    def update_tracking(
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

        # Get routing prediction
        routing_pred = torch.argmax(routing_logits, dim=-1)

        # Use transition counts to determine optimal route
        transition_probs = self.transition_counts[from_expert, :current_num_experts]
        if transition_probs.sum() > 0:  # If we've seen any transitions
            transition_probs = transition_probs / transition_probs.sum()
            optimal_route = torch.argmax(transition_probs)
            routing_correct = (routing_pred == optimal_route).float().mean().item()
        else:
            routing_correct = 0.0

        self.active_experts = set(range(current_num_experts))

        # Update EMAs
        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies["current"][expert_idx] = current_correct
            self.expert_accuracies["routing"][expert_idx] = routing_correct
        else:
            self.expert_accuracies["current"][expert_idx] = (
                self.decay * self.expert_accuracies["current"][expert_idx]
                + (1 - self.decay) * current_correct
            )
            self.expert_accuracies["routing"][expert_idx] = (
                self.decay * self.expert_accuracies["routing"][expert_idx]
                + (1 - self.decay) * routing_correct
            )

        self.update_counts[expert_idx] += 1

    def get_expert_accuracy(self, expert_idx: int) -> dict:
        """Returns both current and routing accuracies for given expert"""
        if expert_idx not in self.active_experts:
            return {"current": 0.0, "routing": 0.0}
        return {
            "current": self.expert_accuracies["current"][expert_idx],
            "routing": self.expert_accuracies["routing"][expert_idx],
        }

    def get_mean_accuracy(self) -> dict:
        """Returns mean accuracy for both current and routing predictions"""
        if not self.active_experts:
            return {"current": 0.0, "routing": 0.0}

        current_accs = [
            self.expert_accuracies["current"][i] for i in self.active_experts
        ]
        routing_accs = [
            self.expert_accuracies["routing"][i] for i in self.active_experts
        ]

        return {
            "current": sum(current_accs) / len(self.active_experts),
            "routing": sum(routing_accs) / len(self.active_experts),
        }

    def get_all_accuracies(self) -> dict:
        """Returns all accuracies for active experts only"""
        return {
            "current": {
                i: self.expert_accuracies["current"][i] for i in self.active_experts
            },
            "routing": {
                i: self.expert_accuracies["routing"][i] for i in self.active_experts
            },
        }
