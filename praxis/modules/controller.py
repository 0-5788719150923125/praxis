from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class PraxisController(nn.Module):
    def __init__(self, hidden_size, max_num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_num_experts = max_num_experts

        # Store previous predictions with batch dimension
        self.register_buffer(
            "previous_logits", torch.zeros(1, max_num_experts)  # [1, max_num_experts]
        )
        self.first_forward = True

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_num_experts * 2),
        )
        nn.init.normal_(self.predictor[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.predictor[-1].bias, 0.1)

        self.decay = 0.99
        self.expert_accuracies = {
            "current": {i: 0.0 for i in range(max_num_experts)},
            "future": {i: 0.0 for i in range(max_num_experts)},
        }
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()

    def forward(
        self, experts: List[nn.Module], expert: nn.Module, expert_output: torch.Tensor
    ):
        current_num_experts = len(experts)
        batch_size = expert_output.size(0)

        if len(expert_output.shape) == 3:
            pooled = torch.mean(expert_output, dim=1)
        else:
            pooled = expert_output

        # Get raw logits for current expert identification and next expert prediction
        full_logits = self.predictor(pooled)  # [batch_size, max_num_experts * 2]
        current_logits, next_logits = torch.split(
            full_logits, self.max_num_experts, dim=1
        )

        # Slice to only get active experts
        current_logits = current_logits[:, :current_num_experts]
        next_logits = next_logits[:, :current_num_experts]

        # Current expert prediction and loss
        current_probs = F.softmax(current_logits, dim=-1)
        expert_idx = experts.index(expert)
        current_true_index = torch.full(
            (batch_size,), expert_idx, device=current_logits.device
        )
        current_loss = F.cross_entropy(current_logits, current_true_index)

        # Next expert prediction
        next_probs = F.softmax(next_logits, dim=-1)
        next_pred = torch.argmax(next_logits, dim=-1)

        # If this isn't the first forward pass, compute loss for previous prediction
        next_loss = torch.tensor(0.0, device=current_logits.device)
        if not self.first_forward:
            # Expand previous prediction to match current batch size
            prev_expanded = self.previous_logits[:, :current_num_experts].expand(
                batch_size, -1
            )
            next_loss = F.cross_entropy(prev_expanded, current_true_index)
        else:
            self.first_forward = False

        # Store current next_logits for future comparison
        # Keep only the first item in batch for inference consistency
        self.previous_logits = next_logits[0:1].detach()

        # Combined loss
        # aux_loss = current_loss + next_loss
        loss_scale = 0.01
        aux_loss = current_loss * loss_scale

        if not self.training:
            self.update_tracking(
                expert_idx=expert_idx,
                current_logits=current_logits,
                next_logits=next_logits,
                true_index=current_true_index,
                current_num_experts=current_num_experts,
                previous_prediction=self.previous_logits,
            )

        return aux_loss, next_pred

    def update_tracking(
        self,
        expert_idx: int,
        current_logits: torch.Tensor,
        next_logits: torch.Tensor,
        true_index: torch.Tensor,
        current_num_experts: int,
        previous_prediction: torch.Tensor,
    ):
        self.active_experts = set(range(current_num_experts))

        # Current expert identification accuracy
        current_pred = torch.argmax(current_logits, dim=-1)
        current_correct = (current_pred == true_index).float().mean().item()

        # Previous prediction accuracy (how well we predicted this expert)
        prev_pred = torch.argmax(previous_prediction, dim=-1)
        pred_correct = (prev_pred == true_index).float().mean().item()

        # Update EMAs
        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies["current"][expert_idx] = current_correct
            self.expert_accuracies["future"][expert_idx] = pred_correct
        else:
            self.expert_accuracies["current"][expert_idx] = (
                self.decay * self.expert_accuracies["current"][expert_idx]
                + (1 - self.decay) * current_correct
            )
            self.expert_accuracies["future"][expert_idx] = (
                self.decay * self.expert_accuracies["future"][expert_idx]
                + (1 - self.decay) * pred_correct
            )

        self.update_counts[expert_idx] += 1

    def get_expert_accuracy(self, expert_idx: int) -> dict:
        """Returns both current and future accuracies for given expert"""
        if expert_idx not in self.active_experts:
            return {"current": 0.0, "future": 0.0}
        return {
            "current": self.expert_accuracies["current"][expert_idx],
            "future": self.expert_accuracies["future"][expert_idx],
        }

    def get_mean_accuracy(self) -> dict:
        """Returns mean accuracy for both current and future predictions"""
        if not self.active_experts:
            return {"current": 0.0, "future": 0.0}

        current_accs = [
            self.expert_accuracies["current"][i] for i in self.active_experts
        ]
        future_accs = [self.expert_accuracies["future"][i] for i in self.active_experts]

        return {
            "current": sum(current_accs) / len(self.active_experts),
            "future": sum(future_accs) / len(self.active_experts),
        }

    def get_all_accuracies(self) -> dict:
        """Returns all accuracies for active experts only"""
        return {
            "current": {
                i: self.expert_accuracies["current"][i] for i in self.active_experts
            },
            "future": {
                i: self.expert_accuracies["future"][i] for i in self.active_experts
            },
        }
