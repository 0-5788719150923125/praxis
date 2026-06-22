import torch
from torch import Tensor

from praxis.losses.regularizer_base import BaseRegularizer

# Activation (AR) and temporal-activation (TAR) penalties from "Regularizing and
# Optimizing LSTM Language Models" (arxiv 1708.02182). Fixed, model-agnostic
# constants - the penalties are mean-reduced so they don't scale with width.
ALPHA = 2e-3  # AR: weight on activation magnitude
BETA = 1e-3  # TAR: weight on step-to-step activation change


class ActivationRegularizer(BaseRegularizer):
    """Keeps hidden states small and temporally smooth. AR penalizes the
    magnitude of the activations; TAR penalizes how much they jump between
    adjacent positions. Both are additive to the main objective.

    From "Regularizing and Optimizing LSTM Language Models" (arxiv 1708.02182).
    """

    name = "activation_reg"

    metric_descriptions = {
        "activation_ar": {
            "description": "Mean squared activation magnitude (AR penalty term).",
            "chart": {
                "title": "Activation Magnitude (AR)",
                "y_label": "Mean Sq.",
                "y_scale": "linear",
                "group": "activation_reg",
                "group_order": 95,
                "order": 10,
            },
        },
        "activation_tar": {
            "description": "Mean squared change between adjacent positions (TAR term).",
            "chart": {
                "title": "Activation Smoothness (TAR)",
                "y_label": "Mean Sq.",
                "y_scale": "linear",
                "group": "activation_reg",
                "order": 20,
            },
        },
    }

    def __init__(self, pad_id: int = 0, alpha: float = ALPHA, beta: float = BETA):
        super().__init__()
        self.pad_id = pad_id
        self.alpha = alpha
        self.beta = beta
        self._metrics: dict = {}

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        # hidden_states: [B, T, D]. Mean-reduce so the constants stay width- and
        # length-independent.
        ar = hidden_states.pow(2).mean()
        if hidden_states.size(1) > 1:
            tar = (hidden_states[:, 1:] - hidden_states[:, :-1]).pow(2).mean()
        else:
            tar = hidden_states.new_zeros(())
        loss = self.alpha * ar + self.beta * tar

        with torch.no_grad():
            self._metrics = {
                "activation_ar": float(ar.detach()),
                "activation_tar": float(tar.detach()),
            }
        return loss

    def training_metrics(self) -> dict:
        return dict(self._metrics)
