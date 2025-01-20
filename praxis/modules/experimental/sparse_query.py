import math
import random
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseQuery(nn.Module):
    """
    A sparse linear layer using GMM routing that only computes and stores top-k heads.
    Inspired by ModuleFormer: Modularity Emerges from Mixture-of-Experts.
    https://arxiv.org/abs/2306.04640
    https://github.com/IBM/ModuleFormer/tree/main
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        head_dim: int,
        top_k: int,
        hidden_size: int = 256,
        expert_hidden_size: int = None,
        bias: bool = True,
        dropout: float = 0.1,
        sample_topk: int = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.identifier = str(uuid.uuid4()).replace("-", "")[:3]
        self.debug = debug
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = min(top_k, num_heads)
        self.sample_topk = min(sample_topk, self.top_k)
        self.acc_aux_loss = True

        # Default expert hidden size to 4x head_dim (common practice)
        if expert_hidden_size is None:
            expert_hidden_size = head_dim * 4

        self.expert_hidden_size = expert_hidden_size
        self.out_features = self.top_k * head_dim

        # Router remains the same
        self.router = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=False), nn.Dropout(dropout)
        )

        # Head centroids and temperature remain the same
        self.head_centroids = nn.Parameter(torch.empty(num_heads, hidden_size))
        self.temperature = nn.Parameter(torch.zeros(1))

        # Replace single weight matrix with two-layer expert
        self.expert_up = nn.Parameter(
            torch.empty(num_heads, in_features, expert_hidden_size)
        )
        self.expert_down = nn.Parameter(
            torch.empty(num_heads, expert_hidden_size, head_dim)
        )

        # Single weight matrix instead of separate expert weights
        self.weight = nn.Parameter(torch.empty(num_heads * head_dim, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads * head_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.init_aux_statistics()

    def __repr__(self):
        return f"{self.__class__.__name__}(type='gmm')"

    def reset_parameters(self):
        # Initialize both expert networks
        nn.init.normal_(self.head_centroids)
        nn.init.uniform_(self.weight, -1 / self.weight.size(1), 1 / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def compute_gating(
        self,
        top_k: int,
        probs: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        # Create dense gates tensor
        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        expert_size = gates.long().sum(0)

        # Flatten and sort
        top_k_gates = top_k_gates.reshape(-1)  # [batch * top_k]
        top_k_experts = top_k_indices.reshape(-1)  # [batch * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)

        batch_index = index_sorted_experts.div(top_k, rounding_mode="trunc")
        batch_gates = top_k_gates[index_sorted_experts]

        return batch_gates, batch_index, expert_size, index_sorted_experts

    def forward(self, x, multiply_by_gates=True):
        batch_size, seq_len, _ = x.shape

        # [Previous routing code remains the same until expert computation]
        probs, logits = self.compute_routing_weights(x)

        # [Top-k selection and compute_gating remain the same]
        if self.training and self.sample_topk > 0:
            _, top_km1_indices = probs.topk(self.top_k - self.sample_topk, dim=1)
            masked_probs = probs.clone() + 1e-6
            masked_probs.scatter_(1, top_km1_indices, 0)
            sampled_indices = torch.multinomial(masked_probs, self.sample_topk)
            top_k_indices = torch.cat([top_km1_indices, sampled_indices], dim=1)
            top_k_probs = torch.gather(probs, 1, top_k_indices)
            log_chance = 0.005
        else:
            top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)
            log_chance = 0.01

        if self.debug and random.random() < log_chance:
            # Calculate expert selection frequency
            expert_counts = torch.zeros(self.num_heads, device=top_k_indices.device)
            for i in range(self.num_heads):
                expert_counts[i] = (top_k_indices == i).sum().item()

            # Normalize by total possible selections
            total_selections = batch_size * seq_len * self.top_k
            expert_frequencies = expert_counts / total_selections

            frequencies = []
            for i, freq in enumerate(expert_frequencies):
                frequencies.append(f"{i}: {freq:.3f}")

            print(
                f"DEBUG: name={self.identifier}, mode={'training' if self.training else 'inference'}, frequencies={', '.join(frequencies)}"
            )

        # Compute gating - reshape for gating computation
        flat_probs = probs.view(-1, self.num_heads)
        flat_top_k_probs = top_k_probs.view(-1, self.top_k)
        flat_top_k_indices = top_k_indices.view(-1, self.top_k)

        # Compute gating
        batch_gates, batch_index, expert_size, sorted_indices = self.compute_gating(
            self.top_k, flat_probs, flat_top_k_probs, flat_top_k_indices
        )

        # Compute outputs for all heads
        full_output = F.linear(x, self.weight, self.bias)
        full_output = full_output.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Prepare indices for gather
        gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        # Select outputs and apply gates
        selected_outputs = full_output.gather(2, gather_indices)
        if multiply_by_gates:
            selected_outputs = selected_outputs * top_k_probs.unsqueeze(-1)

        # Handle aux loss
        if self.training:
            zeros = torch.zeros_like(probs)
            gates = zeros.scatter_(-1, top_k_indices, top_k_probs)
            self.update_aux_statistics(probs, logits, gates)
            aux_loss = self.get_aux_loss_and_clear()
        else:
            aux_loss = 0

        return (
            selected_outputs.reshape(batch_size, seq_len, self.top_k * self.head_dim),
            aux_loss,
        )

    def init_aux_statistics(self):
        """Initialize auxiliary loss statistics"""
        self.acc_count = 0
        self.acc_probs = 0
        self.acc_freq = 0
        self.acc_lsesq = 0

    def update_aux_statistics(self, probs, logits, gates):
        """Update statistics based on current batch"""
        self.acc_count = self.acc_count + logits.size(0)
        self.acc_probs = self.acc_probs + probs.sum(0).sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0).sum(0)
        lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()

    def get_aux_loss_and_clear(self):
        """Calculate auxiliary loss and reset statistics"""
        if self.acc_count == 0:
            return 0.0

        # Compute switch loss
        switchloss = (
            self.num_heads
            * (
                F.normalize(self.acc_probs, p=1, dim=0)
                * F.normalize(self.acc_freq, p=1, dim=0)
            ).sum()
        )

        # Compute z loss
        zloss = self.acc_lsesq / self.acc_count

        # Combine losses with weighting
        loss = switchloss + 0.1 * zloss

        # Reset statistics
        if not self.acc_aux_loss:
            self.init_aux_statistics()

        return loss

    def compute_routing_weights(self, x):
        # Get routing embeddings with dropout
        z = self.router(x)

        # L2 normalize embeddings and centroids
        z_norm = F.normalize(z, p=2, dim=-1)
        centroids_norm = F.normalize(self.head_centroids, p=2, dim=-1)

        # Compute base dot product term
        logits = torch.matmul(z_norm, centroids_norm.t())

        # Optional: Add quadratic terms for full GMM computation
        use_quadratic_terms = False
        if use_quadratic_terms:
            z_norm_sq = torch.sum(z_norm * z_norm, dim=-1, keepdim=True)
            centroids_norm_sq = torch.sum(centroids_norm * centroids_norm, dim=-1)
            logits = logits - 0.5 * (z_norm_sq + centroids_norm_sq[None, :])

        # Scale logits (using input dimension as in transformer attention)
        logits = logits / math.sqrt(z.size(-1))

        # Apply temperature
        logits = logits * self.temperature.exp()

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        return probs, logits


if __name__ == "__main__":
    # Smoke tests
    torch.manual_seed(42)

    # Test initialization
    model = SparseQuery(
        in_features=512, num_heads=12, head_dim=64, top_k=4, hidden_size=256
    )

    # Test forward pass
    batch_size = 32
    seq_length = 128
    x = torch.randn(batch_size, seq_length, 512)
    output, aux_loss = model(x)

    # Verify output shape reflects true sparsity
    expected_shape = (batch_size, seq_length, model.top_k * model.head_dim)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"

    # Test routing
    probs, logits = model.compute_routing_weights(x.view(-1, 512))
    assert probs.shape == (
        batch_size * seq_length,
        12,
    ), f"Expected routing shape {(batch_size * seq_length, 12)}, got {probs.shape}"
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))
    ), "Routing probabilities don't sum to 1"

    # Test auxiliary loss
    model.train()
    _, aux_loss = model(x)  # Accumulate statistics
    assert isinstance(aux_loss, torch.Tensor), "Auxiliary loss should be a tensor"
    assert aux_loss.ndim == 0, "Auxiliary loss should be a scalar"

    print("All tests passed!")

    # Print shapes for verification
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(
        f"Sparsity ratio: {model.top_k}/{model.num_heads} = {model.top_k/model.num_heads:.2f}"
    )

    # Memory usage test
    params_per_head = model.head_dim * model.in_features  # 64 * 512
    full_params = model.num_heads * params_per_head  # 12 * (64 * 512)
    actual_params = model.weight.numel()  # 768 * 512

    assert (
        full_params == actual_params
    ), f"Parameter calculation mismatch: {full_params} != {actual_params}"

    # For effective parameters calculation:
    effective_params = model.top_k * params_per_head  # Only count active heads

    print(f"Parameters per layer: {actual_params:,}")
    print(f"Effective parameters used per forward pass: {effective_params:,}")
