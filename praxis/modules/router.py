import torch
import torch.nn as nn
import torch.nn.functional as F


class PraxisRouter(nn.Module):
    def __init__(self, input_size, num_experts, k):
        super(PraxisRouter, self).__init__()
        self.router = nn.Linear(input_size, num_experts)
        self.num_experts = num_experts
        self.k = k

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # Find top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.k, dim=-1)
        top_k_scores = F.softmax(top_k_logits, dim=-1)

        return top_k_scores, top_k_indices


class SparselyRoutedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, k):
        super(SparselyRoutedMLP, self).__init__()
        self.router = PraxisRouter(input_size, num_experts, k)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, input_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.k = k
        self.num_experts = num_experts

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        top_k_scores, top_k_indices = self.router(x)

        outputs = torch.zeros_like(x)

        for i in range(self.k):
            expert_index = top_k_indices[..., i]
            expert_score = top_k_scores[..., i].unsqueeze(-1)

            # Gather expert outputs
            expert_outputs = torch.stack(
                [
                    self.experts[idx](x[b, s])
                    for b in range(batch_size)
                    for s, idx in enumerate(expert_index[b])
                ]
            )
            expert_outputs = expert_outputs.view(batch_size, seq_len, input_size)

            outputs += expert_score * expert_outputs

        return outputs


if __name__ == "__main__":
    # Usage
    input_size = 512
    hidden_size = 1024
    num_experts = 8
    k = 2
    model = SparselyRoutedMLP(input_size, hidden_size, num_experts, k)

    # Your input tensor
    x = torch.randn(2, 10, input_size)  # [batch_size, seq_len, input_size]

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
