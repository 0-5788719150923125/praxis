import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig
from .experts import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.ctrl = PraxisController(config)
        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList() if config.sparse else None
        for i in range(config.n_layer):
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                self.routers.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        hidden_states = inputs  # Shape: (batch_size, seq_len, n_dim)
        aux_losses = []

        sequence, expert_biases, aux_loss = self.ctrl(hidden_states)
        aux_losses.append(aux_loss)

        for i, choice in enumerate(sequence):
            expert = self.experts[choice]
            residual = hidden_states
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                outputs = self.routers[(i - 1) // 2](
                    hidden_states, expert, attention_mask
                )
            else:
                outputs = expert(hidden_states, attention_mask)
            expert_bias = expert_biases[i]
            hidden_states = outputs["hidden_states"] + expert_bias + residual
            if "aux_loss" in outputs:
                aux_losses.append(outputs["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))


class PraxisController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(config.n_layer))
        self.epsilon = 1e-8
        self.tau = 0.5

        self.recurrent = nn.GRU(config.n_dim, config.n_dim, batch_first=True)
        self.reduction = SequenceReduction(config.n_dim, config.n_dim // 2)
        self.psi = nn.Linear(config.n_dim, config.n_layer)

        self.register_buffer("order_ema", torch.zeros(config.n_layer, config.n_layer))
        self.ema_decay = 0.99
        self.aux_weight = 1.0

    def forward(self, inputs):
        # Pass through GRU
        gru_out, _ = self.recurrent(inputs)  # Shape: (batch_size, seq_len, n_dim)

        # Learnable reduction of GRU outputs
        reduced_gru = self.reduction(gru_out)  # Shape: (batch_size, n_dim)

        # Average across batches
        reduced_gru = reduced_gru.mean(dim=0)  # Shape: (n_dim)

        # Compute logits for each expert
        logits = self.psi(reduced_gru)  # Shape: (num_experts)

        n_experts = logits.size(0)

        if self.training:
            # Apply Gumbel-Softmax during training
            probs = gumbel_sigmoid(logits, tau=self.tau, hard=False)
        else:
            # Normalize the probs during inference
            probs = logits.sigmoid()

        # Get sequence for ordering
        sequence = torch.argsort(probs, dim=-1, descending=True)

        # if self.training:
        #     # Apply Gumbel-Softmax during training
        #     probs = gumbel_sigmoid(logits, tau=self.tau, hard=False)
        #     sequence = torch.argsort(probs, dim=-1, descending=True)
        # else:
        #     # Use multinomial sampling during inference
        #     probs = logits.sigmoid()
        #     sequence = torch.multinomial(
        #         probs, num_samples=n_experts, replacement=False
        #     )

        # Scale the return weights with a learnable alpha
        weights = probs * self.alpha

        aux_loss = 0
        if self.training:
            # Create a position-aware encoding
            current_order = torch.zeros(n_experts, n_experts, device=sequence.device)
            current_order[torch.arange(n_experts), sequence] = 1.0

            # Update EMA of expert order
            self.order_ema = (
                self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
            )

            # Compute diversity loss
            target_distribution = torch.ones_like(self.order_ema) / n_experts
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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        proj = self.fc1(x)  # (batch_size, seq_len, hidden_dim)
        activated = self.act(proj)
        weights = F.softmax(
            self.fc2(activated).squeeze(-1), dim=1
        )  # (batch_size, seq_len)
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


# class PraxisController(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # self.tau_start = 1.0
#         # self.tau_end = 0.1
#         # self.annealing_steps = 10_000
#         # self.current_step = 0

#         self.alpha = nn.Parameter(torch.ones(config.n_layer))
#         self.epsilon = 1e-8
#         self.tau = 0.5

#         self.recurrent = nn.GRU(config.n_dim, config.n_dim, batch_first=True)
#         self.reducer = SequenceReduction(config.n_dim, config.n_dim // 2)
#         self.psi = nn.Linear(config.n_dim, config.n_layer)

#         self.register_buffer("order_ema", torch.zeros(config.n_layer))
#         self.ema_decay = 0.99
#         self.aux_weight = 1.0

#     def forward(self, inputs):
#         # Pass through GRU
#         gru_out, _ = self.recurrent(inputs)  # Shape: (batch_size, seq_len, n_dim)

#         # Learnable reduction of GRU outputs
#         reduced_gru = self.reducer(gru_out)  # Shape: (batch_size, n_dim)

#         # Average across batches
#         reduced_gru = reduced_gru.mean(dim=0)  # Shape: (n_dim)

#         # Compute logits for each expert
#         logits = self.psi(reduced_gru)  # Shape: (num_experts)

#         if self.training:
#             # Apply Gumbel-Softmax during training
#             # tau = self.get_current_tau()
#             probs = gumbel_sigmoid(logits, tau=self.tau, hard=False)
#         else:
#             # Normalize the probs during inference
#             probs = logits.sigmoid()

#         # Get sequence for ordering
#         sequence = torch.argsort(probs, dim=-1, descending=True)

#         # Scale the probs to return weights
#         weights = probs * self.alpha

#         aux_loss = 0
#         if self.training:
#             num_experts = logits.size(0)

#             # Create a position-aware encoding
#             current_order = torch.zeros(
#                 num_experts, num_experts, device=sequence.device
#             )
#             current_order[torch.arange(num_experts), sequence] = 1.0

#             # Update EMA of expert order
#             self.order_ema = (
#                 self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
#             )

#             # Compute diversity loss
#             target_distribution = torch.ones_like(self.order_ema) / num_experts
#             aux_loss = (
#                 F.kl_div(
#                     (self.order_ema + self.epsilon).log(),
#                     target_distribution,
#                     reduction="batchmean",
#                 )
#                 * self.aux_weight
#             )

#         return sequence, weights, aux_loss

#     # def get_current_tau(self):
#     #     if self.current_step >= self.annealing_steps:
#     #         return self.tau_end

#     #     self.current_step += 1

#     #     cos_factor = (math.cos(math.pi * self.current_step / self.annealing_steps)) / 2

#     #     return self.tau_end + (self.tau_start - self.tau_end) * cos_factor


# class PraxisController(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.n_layer = config.n_layer

#         self.tau = 0.1
#         self.epsilon = 1e-8

#         self.gru = nn.GRU(config.n_dim, config.n_dim, batch_first=True)
#         self.psi = nn.Linear(config.n_dim, config.n_layer)

#         self.b_red = BatchReduction(config.n_dim, hidden_dim=64)
#         self.s_red = SequenceReduction(config.n_dim, hidden_dim=64)

#         self.register_buffer("order_ema", torch.zeros(self.n_layer))
#         self.ema_decay = 0.99
#         self.aux_weight = 10.0

#     def forward(self, inputs):
#         # Aggregate inputs (via mean) to create a single context vector
#         inputs_reduced = self.b_red(inputs)

#         # Pass through GRU
#         gru_out, _ = self.gru(inputs_reduced)  # Shape: (seq_len, dims)

#         # Reduce GRU outputs to just a single time step
#         reduced_gru = self.s_red(gru_out)  # Shape: (dims)

#         # Compute logits for each expert
#         logits = self.psi(reduced_gru)  # Shape: (num_experts)

#         if self.training:
#             # Apply Gumbel-Softmax during training
#             logits = F.gumbel_softmax(logits, tau=self.tau, hard=False)

#         sequence = torch.argsort(logits, dim=-1, descending=True)

#         aux_loss = 0
#         if self.training:
#             # Create a position-aware encoding
#             current_order = torch.zeros(
#                 self.n_layer, self.n_layer, device=sequence.device
#             )
#             current_order[torch.arange(self.n_layer), sequence] = 1.0

#             # Update EMA of expert order
#             self.order_ema = (
#                 self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
#             )

#             # Compute diversity loss
#             target_distribution = torch.ones_like(self.order_ema) / self.n_layer
#             aux_loss = (
#                 F.kl_div(
#                     (self.order_ema + self.epsilon).log(),
#                     target_distribution,
#                     reduction="batchmean",
#                 )
#                 * self.aux_weight
#             )

#         return sequence, aux_loss


# class BatchReduction(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
#         self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 1), padding=(1, 0))
#         self.activation = nn.GELU()

#     def forward(self, x):
#         # x shape: (batch_size, seq_len, dim)
#         x = x.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
#         x = self.activation(self.conv1(x))
#         x = self.conv2(x)
#         x = x.squeeze(1)  # (batch_size, seq_len, dim)
#         return x.mean(dim=0)  # (seq_len, dim)


# class SequenceReduction(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
#         self.activation = nn.GELU()

#     def forward(self, x):
#         # x shape: (seq_len, dim)
#         x = x.transpose(0, 1).unsqueeze(0)  # (1, dim, seq_len)
#         x = self.activation(self.conv1(x))
#         x = self.conv2(x)
#         print(x.shape)
#         return x.squeeze(0).mean(dim=1)  # (dim)


# class PraxisController(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.n_layer = config.n_layer

#         self.tau = 0.1
#         self.epsilon = 1e-8

#         self.gru = nn.GRU(config.n_dim, config.n_dim, batch_first=True)
#         self.psi = nn.Linear(config.n_dim, config.n_layer)

#         self.b_red = LearnableReduction(config.n_dim, hidden_dim=64)
#         # self.s_red = LearnableReduction(config.n_dim, hidden_dim=64)

#         self.register_buffer("order_ema", torch.zeros(self.n_layer))
#         self.ema_decay = 0.99
#         self.aux_weight = 10.0

#     def forward(self, inputs):
#         # Aggregate inputs (via mean) to create a single context vector
#         inputs_reduced = self.b_red(inputs)

#         # Pass through GRU
#         gru_out, _ = self.gru(inputs_reduced.unsqueeze(0))  # Shape: (seq_len, dims)
#         print(gru_out.shape)
#         # Reduce GRU outputs to just a single time step
#         reduced_gru = gru_out.mean(dim=0)  # Shape: (dims)
#         # reduced_gru = self.s_red(gru_out.squeeze(0))

#         # Compute logits for each expert
#         logits = self.psi(reduced_gru.squeeze(0))  # Shape: (num_experts)
#         # logits = logits.mean(dim=0)

#         # logits = self.s_red(psi_out.squeeze(0))

#         # if self.training:
#         #     # Apply Gumbel-Softmax during training
#         #     probs = F.gumbel_softmax(logits, tau=self.tau, hard=False)
#         #     sequence = torch.argsort(probs, dim=-1, descending=True)
#         # else:
#         #     # Use multinomial sampling during inference
#         #     probs = F.softmax(logits / self.tau, dim=-1)
#         #     sequence = torch.multinomial(
#         #         probs, num_samples=self.n_layer, replacement=False
#         #     )

#         if self.training:
#             # Apply Gumbel-Softmax during training
#             logits = F.gumbel_softmax(logits, tau=self.tau, hard=False)

#         sequence = torch.argsort(logits, dim=-1, descending=True)

#         aux_loss = 0
#         if self.training:
#             # Create a position-aware encoding
#             current_order = torch.zeros(
#                 self.n_layer, self.n_layer, device=sequence.device
#             )
#             current_order[torch.arange(self.n_layer), sequence] = 1.0

#             # Update EMA of expert order
#             self.order_ema = (
#                 self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
#             )

#             # Compute diversity loss
#             target_distribution = torch.ones_like(self.order_ema) / self.n_layer
#             aux_loss = (
#                 F.kl_div(
#                     (self.order_ema + self.epsilon).log(),
#                     target_distribution,
#                     reduction="batchmean",
#                 )
#                 * self.aux_weight
#             )

#         return sequence, aux_loss


# class LearnableReduction(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
#         self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 1), padding=(1, 0))
#         self.activation = nn.GELU()

#     def forward(self, x):
#         # x shape: (batch_size, seq_len, dim)
#         x = x.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
#         x = self.activation(self.conv1(x))
#         x = self.conv2(x)
#         x = x.squeeze(1)  # (batch_size, seq_len, dim)
#         return x.mean(dim=0)  # (seq_len, dim)
