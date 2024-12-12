import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


class PraxisController(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.epsilon = 1e-8
        self.tau = 0.5

        self.recurrent = nn.GRU(
            config.hidden_size, config.hidden_size, batch_first=True
        )
        self.reduction = SequenceReduction(config.hidden_size, config.hidden_size // 2)
        self.psi = nn.Linear(config.hidden_size, config.num_layers)
        self.alpha = nn.Linear(1, config.hidden_size)

        self.register_buffer(
            "order_ema", torch.zeros(config.num_layers, config.num_layers)
        )
        self.ema_decay = 0.99
        self.aux_weight = 1.0

    def forward(self, inputs):
        # Pass through GRU
        gru_out, _ = self.recurrent(inputs)  # Shape: (batch_size, seq_len, hidden_size)

        # Learnable reduction of GRU outputs
        reduced_gru = self.reduction(gru_out)  # Shape: (batch_size, hidden_size)

        # Average across batches
        reduced_gru = reduced_gru.mean(dim=0)  # Shape: (hidden_size)

        # Compute logits for each expert
        logits = self.psi(reduced_gru)  # Shape: (num_experts)

        if self.training:
            # Apply Gumbel-Softmax during training
            probs = gumbel_sigmoid(logits, tau=self.tau, hard=False)
        else:
            # Normalize the probs during inference
            probs = logits.sigmoid()

        # Get sequence for ordering
        values, sequence = torch.sort(probs, dim=-1, descending=True)

        # Scale the return weights with a learnable alpha
        weights = self.alpha(values.unsqueeze(-1))

        aux_loss = 0
        if self.training:
            # Create a position-aware encoding
            num_experts = logits.size(0)
            current_order = torch.zeros(
                num_experts, num_experts, device=sequence.device
            )
            current_order[torch.arange(num_experts), sequence] = 1.0

            # Update EMA of expert order
            self.order_ema = (
                self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
            )

            # Compute diversity loss
            target_distribution = torch.ones_like(self.order_ema) / num_experts
            aux_loss = (
                F.kl_div(
                    (self.order_ema + self.epsilon).log(),
                    target_distribution,
                    reduction="batchmean",
                )
                * self.aux_weight
            )

        return sequence, weights, aux_loss


class SequenceReduction(nn.Module):
    def __init__(self, input_dim, hiddehidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hiddehidden_size)
        self.fc2 = nn.Linear(hiddehidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        proj = self.fc1(x)  # (batch_size, seq_len, hiddehidden_size)
        weights = F.softmax(self.fc2(proj).squeeze(-1), dim=1)  # (batch_size, seq_len)
        x_reduced = torch.bmm(weights.unsqueeze(1), x)  # (batch_size, 1, input_dim)
        return x_reduced.squeeze(1)  # (batch_size, input_dim)


def gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5
) -> torch.Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
