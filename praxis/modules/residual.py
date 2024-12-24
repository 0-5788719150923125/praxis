import torch
from torch import Tensor, nn


class HyperConnection(nn.Module):
    """
    This module implements static hyper-connections, which are a replacement to
    residual connections.
    https://arxiv.org/abs/2409.19606
    """

    def __init__(self, hidden_size: int, expansion_rate: int = 4, layer_idx: int = 0):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Core HC parameters
        self.alpha_merger = nn.Parameter(torch.ones(expansion_rate, 1))
        self.alpha_residual = nn.Parameter(torch.ones(expansion_rate, expansion_rate))
        self.beta = nn.Parameter(torch.ones(expansion_rate))

        # Initialize as per paper's equation (8)
        with torch.no_grad():
            self.alpha_residual.fill_(0)
            self.alpha_residual.diagonal().fill_(1)
            self.alpha_merger.fill_(0)
            self.alpha_merger[layer_idx % expansion_rate] = 1
            self.beta.fill_(1)

        self.register_buffer("hidden_states", None)

    def forward(self, inputs: Tensor = None, outputs: Tensor = None) -> Tensor:
        if self.hidden_states is None:
            self.hidden_states = inputs.unsqueeze(0).expand(
                self.expansion_rate, *inputs.shape
            )
            return (self.alpha_merger.view(-1, 1, 1, 1) * self.hidden_states).sum(dim=0)

        if outputs is not None:
            # Replace einsum with reshape + mm + reshape
            orig_shape = self.hidden_states.shape
            flat_hidden = self.hidden_states.reshape(self.expansion_rate, -1)
            hidden_states = torch.mm(self.alpha_residual, flat_hidden)
            hidden_states = hidden_states.reshape(orig_shape)

            self.hidden_states = hidden_states + self.beta.view(
                -1, 1, 1, 1
            ) * outputs.unsqueeze(0)
            output = self.hidden_states.sum(dim=0)
            self.hidden_states = None
            return output

        return (self.alpha_merger.view(-1, 1, 1, 1) * self.hidden_states).sum(dim=0)
