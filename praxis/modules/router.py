import torch
import torch.nn as nn
import torch.nn.functional as F


class PraxisRouter(nn.Module):
    def __init__(self, input_size, num_experts, k, temperature=1.0):
        super(PraxisRouter, self).__init__()
        self.proj = nn.Linear(input_size, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature

    def forward(self, x):
        router_logits = self.proj(x)

        # Apply Gumbel-Softmax
        gumbel_softmax = F.gumbel_softmax(
            router_logits, tau=self.temperature, hard=False, dim=-1
        )

        # Top-k selection
        top_k_gumbel, top_k_indices = torch.topk(gumbel_softmax, self.k, dim=-1)

        # Normalize top-k probabilities
        router_probs = F.normalize(top_k_gumbel, p=1, dim=-1)

        # Calculate load balancing loss
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.k):
            expert_counts += torch.bincount(
                top_k_indices[:, :, i].reshape(-1), minlength=self.num_experts
            )

        expert_probs = expert_counts / expert_counts.sum()
        target_probs = torch.ones_like(expert_probs) / self.num_experts
        load_balancing_loss = F.kl_div(
            expert_probs.log(), target_probs, reduction="batchmean"
        )

        return router_probs, top_k_indices, load_balancing_loss


# Usage example
if __name__ == "__main__":
    input_size = 256
    num_experts = 8
    k = 2
    batch_size = 1
    seq_len = 256

    router = PraxisRouter(input_size, num_experts, k)
    x = torch.randn(batch_size, seq_len, input_size)

    router_probs, top_k_indices, load_balancing_loss = router(x)

    print(f"Router probs shape: {router_probs.shape}")
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Load balancing loss: {load_balancing_loss.item()}")
