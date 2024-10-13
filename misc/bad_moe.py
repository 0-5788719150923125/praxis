import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from praxis import PraxisConfig


class ExpertGLU(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.up = nn.Linear(config.num_dims, 8 * config.num_dims)
        self.act = ACT2FN[config.activation]
        self.down = nn.Linear(4 * config.num_dims, config.num_dims)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * self.act(b))


class SparseMoEMLP(nn.Module):
    def __init__(self, config: PraxisConfig, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create multiple experts
        self.experts = nn.ModuleList([ExpertGLU(config) for _ in range(num_experts)])

        # Router network
        self.router = nn.Linear(config.num_dims, num_experts)

    def forward(self, x):
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape

        # Flatten the input
        flat_x = x.view(-1, x.size(-1))  # (batch_size * seq_len, num_dims)

        # Route the input
        router_logits = self.router(flat_x)  # (batch_size * seq_len, num_experts)

        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Compute expert outputs
        expert_outputs = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            expert_index = top_k_indices[:, k]
            prob = top_k_probs[:, k].unsqueeze(-1)
            for i, expert in enumerate(self.experts):
                # Create a mask for this expert
                mask = expert_index == i
                if mask.any():
                    # Only compute for samples that use this expert
                    expert_inputs = flat_x[mask]
                    expert_output = expert(expert_inputs)
                    # Combine expert outputs weighted by router probabilities
                    expert_outputs[mask] += expert_output * prob[mask]

        # Reshape output back to original shape
        output = expert_outputs.view(batch_size, seq_len, -1)

        return output


# Usage
config = PraxisConfig(num_dims=512, activation="gelu")
moe_mlp = SparseMoEMLP(config, num_experts=8, top_k=2)
input_tensor = torch.randn(32, 64, 512)  # (batch_size, seq_len, num_dims)
output = moe_mlp(input_tensor)
print(output)
