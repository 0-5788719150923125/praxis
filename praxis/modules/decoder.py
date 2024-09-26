import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig
from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.sparse = config.sparse
        self.experts = nn.ModuleList()
        self.pipe = nn.ModuleList()
        for i in range(config.n_layer):
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if config.sparse and use_router:
                self.pipe.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        hidden_states = inputs
        aux_losses = []
        for i, expert in enumerate(self.experts):
            outputs = expert(hidden_states, attention_mask)
            use_router = i % 2 != 0  # if layer is odd
            if self.sparse and use_router:
                outputs = self.pipe[i % 2](hidden_states, expert, attention_mask)
            hidden_states = outputs["hidden_states"]
            if "aux_loss" in outputs:
                aux_losses.append(outputs["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))


# class PraxisDecoder(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.sparse = config.sparse
#         self.experts = nn.ModuleList()
#         self.net = nn.ModuleList()
#         self.svm = PraxiSVM(config)
#         for i in range(config.n_layer):
#             self.experts.append(PraxisBlock(config))
#             use_router = i % 2 != 0  # if layer is odd
#             if config.sparse and use_router:
#                 self.net.append(PraxisMixtureOfDepths(config))

#     def forward(self, inputs, attention_mask):
#         hidden_states = inputs
#         aux_losses = []

#         # Predict the best order of experts using the router
#         expert_order, hinge_loss = self.svm(hidden_states)
#         aux_losses.append(hinge_loss)

#         if random.random() < 0.01:
#             print(expert_order)

#         # Iterate over the batch
#         for batch_idx in range(expert_order.shape[0]):
#             # Iterate over the experts based on the predicted order
#             for seq_idx in range(expert_order.shape[1]):
#                 expert_idx = expert_order[batch_idx, seq_idx].item()
#                 expert = self.experts[expert_idx]
#                 outputs = hidden_states
#                 use_router = expert_idx % 2 != 0  # if layer is odd
#                 if self.sparse and use_router:
#                     outputs = self.net[expert_idx % 2](
#                         hidden_states[batch_idx].unsqueeze(0),
#                         expert,
#                         attention_mask[batch_idx].unsqueeze(0),
#                     )
#                 else:
#                     outputs = expert(
#                         hidden_states[batch_idx].unsqueeze(0),
#                         attention_mask[batch_idx].unsqueeze(0),
#                     )
#                 hidden_states = hidden_states.clone()
#                 hidden_states[batch_idx] = outputs["hidden_states"]
#                 if "aux_loss" in outputs:
#                     aux_losses.append(outputs["aux_loss"])

#         return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))


# class PraxiSVM(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.key = nn.Linear(self.n_dim, config.n_layer)
#         self.temperature = 0.9

#     def forward(self, hidden_states, labels=None):
#         # Compute the router logits
#         logits = self.key(hidden_states[:, -1])  # Use the last token for routing

#         # Add Gumbel noise to the logits
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#         noisy_logits = (logits + gumbel_noise) / self.temperature

#         # Apply softmax to obtain the permutation
#         permutation = F.softmax(noisy_logits, dim=-1)

#         # Get the indices of the experts in the permutation order
#         expert_order = torch.argsort(permutation, dim=-1, descending=True)

#         # Compute the hinge loss if labels are provided
#         hinge_loss = 0
#         if labels is not None:
#             hinge_loss = nn.HingeEmbeddingLoss()(logits.squeeze(), labels.float())

#         return expert_order, hinge_loss


# class PraxiSVM(nn.Module):
#     def __init__(self, config: PraxisConfig):
#         super().__init__()
#         self.n_dim = config.n_dim
#         self.hidden_size = config.n_dim // 2
#         self.temporal = nn.GRU(self.n_dim, self.hidden_size, batch_first=True)
#         self.out = nn.Linear(self.hidden_size, config.n_layer)
#         self.temperature = 0.9

#     def forward(self, hidden_states, labels=None):
#         # Pass the hidden states through the GRU
#         replay_output, _ = self.temporal(hidden_states)
#         logits = self.out(replay_output[:, -1])

#         # Add Gumbel noise to the logits
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#         noisy_logits = (logits + gumbel_noise) / self.temperature

#         # Apply softmax to obtain the permutation
#         permutation = F.softmax(noisy_logits, dim=-1)

#         # Get the indices of the experts in the permutation order
#         expert_order = torch.argsort(permutation, dim=-1, descending=True)

#         # Compute the hinge loss if labels are provided
#         hinge_loss = 0
#         if labels is not None:
#             hinge_loss = nn.HingeEmbeddingLoss()(logits.squeeze(), labels.float())

#         return expert_order, hinge_loss
