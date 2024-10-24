from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class PraxisController(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts),
        )

        self.tracker = ExpertPredictionTracker(num_experts=num_experts)

    def forward(
        self, experts: List[nn.Module], expert: nn.Module, expert_output: torch.Tensor
    ):

        # Average pooling across sequence length if needed
        if len(expert_output.shape) == 3:
            pooled = torch.mean(expert_output, dim=1)
        else:
            pooled = expert_output

        # Get raw logits
        logits = self.predictor(pooled)

        # Get probabilities for scaling factor
        pred_probs = F.softmax(logits, dim=-1)

        # Get the expert index
        expert_idx = experts.index(expert)
        # Create tensor with same expert index repeated for each item in batch
        true_index = torch.full((logits.size(0),), expert_idx, device=logits.device)

        # Use cross_entropy which combines log_softmax and nll_loss
        aux_loss = F.cross_entropy(logits, true_index)

        # Use pred_probs for scaling
        correct_prob = pred_probs[:, expert_idx]

        # Reshape for broadcasting
        scaling_factor = correct_prob.view(-1, 1, 1)  # [16, 1, 1]

        # Scale hidden states by prediction confidence
        new_states = expert_output * scaling_factor

        if not self.training:  # Only track during evaluation
            self.tracker.update(expert_idx, logits, true_index)

        return new_states, aux_loss

    def get_expert_accuracy(self):
        return self.tracker.get_expert_accuracy()

    def get_mean_accuracy(self):
        return self.tracker.get_mean_accuracy()

    def get_all_accuracies(self):
        return self.tracker.get_all_accuracies()


class ExpertPredictionTracker:
    def __init__(self, num_experts: int, decay: float = 0.99):
        """
        Args:
            num_experts: Number of experts to track
            decay: EMA decay factor (higher = smoother/slower changes)
        """
        self.num_experts = num_experts
        self.decay = decay
        # Initialize EMAs for each expert
        self.expert_accuracies = {i: 0.0 for i in range(num_experts)}
        # Track number of updates for better initial estimates
        self.update_counts = {i: 0 for i in range(num_experts)}

    def update(
        self, expert_idx: int, pred_logits: torch.Tensor, true_indices: torch.Tensor
    ) -> float:
        """
        Update accuracy EMA for a specific expert

        Args:
            expert_idx: True index of the expert
            pred_logits: Model predictions [batch_size, num_experts]
            true_indices: Ground truth indices [batch_size]

        Returns:
            Current accuracy for this expert
        """
        # Get predictions
        predictions = torch.argmax(pred_logits, dim=-1)
        # Calculate accuracy for this batch
        correct = (predictions == true_indices).float().mean().item()

        # Update EMA
        if self.update_counts[expert_idx] == 0:
            # First update
            self.expert_accuracies[expert_idx] = correct
        else:
            # EMA update
            self.expert_accuracies[expert_idx] = (
                self.decay * self.expert_accuracies[expert_idx]
                + (1 - self.decay) * correct
            )

        self.update_counts[expert_idx] += 1
        return self.expert_accuracies[expert_idx]

    def get_expert_accuracy(self, expert_idx: int) -> float:
        """Get current accuracy EMA for specific expert"""
        return self.expert_accuracies[expert_idx]

    def get_mean_accuracy(self) -> float:
        """Get mean accuracy across all experts"""
        return sum(self.expert_accuracies.values()) / self.num_experts

    def get_all_accuracies(self) -> dict:
        """Get dictionary of all expert accuracies"""
        return self.expert_accuracies.copy()
