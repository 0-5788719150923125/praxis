import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from praxis import PraxisConfig


class SparseMoEMLP(nn.Module):
    def __init__(self, config: PraxisConfig, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_dim = config.n_dim

        # Single set of expert weights
        self.w1 = nn.Parameter(torch.randn(num_experts, config.n_dim, 4 * config.n_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, 4 * config.n_dim, config.n_dim))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 4 * config.n_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, config.n_dim))

        # Router network
        self.router = nn.Linear(config.n_dim, num_experts)

        self.act = ACT2FN[config.activation]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Route the input
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Create masks for selected experts
        masks = F.one_hot(top_k_indices, num_classes=self.num_experts).float()

        # Expert computation (in parallel)
        x_e = x.unsqueeze(2).expand(
            -1, -1, self.num_experts, -1
        )  # (batch_size, seq_len, num_experts, n_dim)
        h = torch.einsum("bsed,edh->bseh", x_e, self.w1) + self.b1.unsqueeze(
            0
        ).unsqueeze(0)
        h = self.act(h)
        y = torch.einsum("bseh,ehd->bsed", h, self.w2) + self.b2.unsqueeze(0).unsqueeze(
            0
        )

        # Combine expert outputs
        combined_output = torch.einsum("bsed,bske,bsk->bsd", y, masks, top_k_probs)

        return combined_output


# Usage
config = PraxisConfig(n_dim=512, activation="gelu")
moe_mlp = SparseMoEMLP(config, num_experts=8, top_k=2)
input_tensor = torch.randn(32, 64, 512)  # (batch_size, seq_len, n_dim)
output = moe_mlp(input_tensor)
print(output)
