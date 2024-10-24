from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class PraxisController(nn.Module):
    def __init__(self, hidden_size, max_num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_num_experts = max_num_experts

        # Prediction network with max capacity
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_num_experts),
        )

        self.tracker = DynamicExpertTracker(max_num_experts=max_num_experts)

    def forward(
        self, experts: List[nn.Module], expert: nn.Module, expert_output: torch.Tensor
    ):
        current_num_experts = len(experts)

        # Average pooling across sequence length if needed
        if len(expert_output.shape) == 3:
            pooled = torch.mean(expert_output, dim=1)
        else:
            pooled = expert_output

        # Get raw logits for all possible experts
        full_logits = self.predictor(pooled)  # [batch_size, max_num_experts]

        # Slice to only get active experts
        logits = full_logits[
            :, :current_num_experts
        ]  # [batch_size, current_num_experts]

        # Get probabilities for scaling factor (only for active experts)
        pred_probs = F.softmax(logits, dim=-1)

        # Get the expert index
        expert_idx = experts.index(expert)
        true_index = torch.full((logits.size(0),), expert_idx, device=logits.device)

        aux_loss = F.cross_entropy(logits, true_index)

        # Use pred_probs for scaling
        correct_prob = pred_probs[:, expert_idx]
        scaling_factor = correct_prob.view(-1, 1, 1)
        new_states = expert_output * scaling_factor

        if not self.training:
            self.tracker.update(expert_idx, logits, true_index, current_num_experts)

        return new_states, aux_loss

    def get_expert_accuracy(self):
        return self.tracker.get_expert_accuracy()

    def get_mean_accuracy(self):
        return self.tracker.get_mean_accuracy()

    def get_all_accuracies(self):
        return self.tracker.get_all_accuracies()


class DynamicExpertTracker:
    def __init__(self, max_num_experts: int, decay: float = 0.99):
        self.max_num_experts = max_num_experts
        self.decay = decay
        self.expert_accuracies = {i: 0.0 for i in range(max_num_experts)}
        self.update_counts = {i: 0 for i in range(max_num_experts)}
        self.active_experts = set()  # Track which experts are currently active

    def update(
        self,
        expert_idx: int,
        pred_logits: torch.Tensor,
        true_indices: torch.Tensor,
        current_num_experts: int,
    ) -> float:
        # Update set of active experts
        self.active_experts = set(range(current_num_experts))

        predictions = torch.argmax(pred_logits, dim=-1)
        correct = (predictions == true_indices).float().mean().item()

        if self.update_counts[expert_idx] == 0:
            self.expert_accuracies[expert_idx] = correct
        else:
            self.expert_accuracies[expert_idx] = (
                self.decay * self.expert_accuracies[expert_idx]
                + (1 - self.decay) * correct
            )

        self.update_counts[expert_idx] += 1
        return self.expert_accuracies[expert_idx]

    def get_expert_accuracy(self, expert_idx: int) -> float:
        if expert_idx not in self.active_experts:
            return 0.0
        return self.expert_accuracies[expert_idx]

    def get_mean_accuracy(self) -> float:
        """Get mean accuracy across only active experts"""
        if not self.active_experts:
            return 0.0
        active_accuracies = [self.expert_accuracies[i] for i in self.active_experts]
        return sum(active_accuracies) / len(self.active_experts)

    def get_all_accuracies(self) -> dict:
        """Get dictionary of accuracies for active experts only"""
        return {i: self.expert_accuracies[i] for i in self.active_experts}
