import torch
import torch.nn.functional as F
from torch import nn


class ExpertIndexPredictor(nn.Module):
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

    def forward(self, expert_output: torch.Tensor):
        # Average pooling across sequence length if needed
        if len(expert_output.shape) == 3:
            pooled = torch.mean(expert_output, dim=1)
        else:
            pooled = expert_output

        # Predict expert index probabilities
        logits = self.predictor(pooled)
        return F.log_softmax(logits, dim=-1)


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
