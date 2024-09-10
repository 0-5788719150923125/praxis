import math
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class PraxisRouter(nn.Module):
    def __init__(
        self, input_size, num_experts, k, target_temperature=1.0, annealing_steps=10000
    ):
        super(PraxisRouter, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.initial_temperature = 1.0
        self.target_temperature = target_temperature
        self.annealing_steps = annealing_steps
        self.epsilon = 1e-10
        self.switch = nn.Sequential(
            OrderedDict(
                [
                    ("in", nn.Linear(input_size, self.num_experts)),
                    ("act", ACT2FN["gelu_new"]),
                    ("out", nn.Linear(self.num_experts, self.num_experts)),
                ]
            )
        )
        self.current_step = 0

    def forward(self, x):
        logits = self.switch(x)

        # Get current temperature
        temperature = self.get_temperature()

        # Apply Gumbel-Softmax
        gumbel = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)

        # Top-k selection
        top_k_gumbel, top_k_indices = torch.topk(gumbel, self.k, dim=-1)

        # Normalize top-k probabilities
        router_probs = F.normalize(top_k_gumbel + self.epsilon, p=1, dim=-1)

        # Calculate load balancing loss
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.k):
            expert_counts += torch.bincount(
                top_k_indices[:, :, i].reshape(-1), minlength=self.num_experts
            )

        expert_probs = expert_counts / expert_counts.sum()
        target_probs = torch.ones_like(expert_probs) / self.num_experts
        load_balancing_loss = F.kl_div(
            (expert_probs + self.epsilon).log(), target_probs, reduction="batchmean"
        )

        self.current_step += 1

        return (
            router_probs,
            top_k_indices,
            load_balancing_loss,
            expert_counts,
            temperature,
        )

    def get_temperature(self):
        if self.current_step >= self.annealing_steps:
            return self.target_temperature

        progress = self.current_step / self.annealing_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (
            self.target_temperature
            + (self.initial_temperature - self.target_temperature) * cosine_decay
        )

    # def get_temperature(self):
    #     if self.current_step >= self.annealing_steps:
    #         return self.target_temperature

    #     progress = self.current_step / self.annealing_steps
    #     # Using a faster decaying function
    #     decay = 1 - (1 - math.exp(-5 * progress)) / (1 - math.exp(-5))
    #     return (
    #         self.target_temperature
    #         + (self.initial_temperature - self.target_temperature) * decay
    #     )


# Usage example
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    input_size = 256
    num_experts = 8
    k = 2
    batch_size = 1
    seq_len = 256
    total_steps = 10000

    router = PraxisRouter(
        input_size, num_experts, k, target_temperature=0.1, annealing_steps=total_steps
    )
    x = torch.randn(batch_size, seq_len, input_size)

    temperatures = []

    for step in range(total_steps):
        _, _, _, _, current_temp = router(x)
        temperatures.append(current_temp)

        if step % 1000 == 0:
            print(f"Step {step}: Temperature = {current_temp:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(total_steps), temperatures)
    plt.title("Temperature Annealing over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.savefig("temperature_annealing.png")
    plt.show()

    print("Plot saved as 'temperature_annealing.png'")
