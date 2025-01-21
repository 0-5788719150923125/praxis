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
        hidden_size: int = None,
        gating_size: int = None,
        bias: bool = False,
        dropout: float = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.identifier = str(uuid.uuid4()).replace("-", "")[:3]
        self.debug = debug
        self.log_frequency = 0.001
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = min(top_k, num_heads)
        self.acc_aux_loss = True

        # Default expert hidden size
        if hidden_size is None:
            hidden_size = head_dim

        if gating_size is None:
            gating_size = hidden_size

        # Router remains the same
        self.router = nn.Sequential(
            nn.Linear(in_features, gating_size, bias=False), nn.Dropout(dropout)
        )

        # Head centroids and temperature remain the same
        self.head_centroids = nn.Parameter(torch.empty(num_heads, gating_size))
        self.temperature = nn.Parameter(torch.zeros(1))

        self.input_experts = nn.Parameter(
            torch.empty(num_heads, in_features, hidden_size)
        )
        self.output_experts = nn.Parameter(
            torch.empty(num_heads, hidden_size, head_dim)
        )
        if bias:
            self.input_bias = nn.Parameter(torch.empty(num_heads, hidden_size))
            self.output_bias = nn.Parameter(torch.empty(num_heads, head_dim))
        else:
            self.register_parameter("input_bias", None)
            self.register_parameter("output_bias", None)

        self.reset_parameters()
        self.init_aux_statistics()

    def __repr__(self):
        return f"{self.__class__.__name__}(type='gmm')"

    def reset_parameters(self):
        # Initialize both expert networks
        nn.init.normal_(self.head_centroids)
        nn.init.uniform_(
            self.input_experts,
            -1 / self.input_experts.size(1),
            1 / self.input_experts.size(1),
        )
        nn.init.uniform_(
            self.output_experts,
            -1 / self.output_experts.size(1),
            1 / self.output_experts.size(1),
        )
        if self.input_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.input_experts)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.input_bias, -bound, bound)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_experts)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.output_bias, -bound, bound)

    def forward(self, x, multiply_by_gates=True):
        batch_size, seq_len, _ = x.shape

        # [Previous routing code remains the same until expert computation]
        probs, logits = self.compute_routing_weights(x)

        # [Top-k selection and compute_gating remain the same]
        sample_top_k = int(self.top_k * 0.5)  # 50% deterministic, 50% stochastic
        if self.training and sample_top_k > 0:
            # Flatten batch and sequence dimensions
            batch_size, seq_len = probs.shape[:2]
            flat_probs = probs.view(-1, probs.size(-1))  # [batch*seq, num_heads]

            # Do sampling on flattened probs
            _, top_km1_indices = flat_probs.topk(self.top_k - sample_top_k, dim=1)
            masked_probs = flat_probs + 1e-6
            masked_probs[
                torch.arange(flat_probs.size(0)).unsqueeze(1), top_km1_indices
            ] = 0
            k_indices = torch.multinomial(masked_probs, sample_top_k)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_probs = torch.gather(flat_probs, 1, top_k_indices)

            # Reshape back to original dimensions
            top_k_indices = top_k_indices.view(batch_size, seq_len, -1)
            top_k_probs = top_k_probs.view(batch_size, seq_len, -1)
        else:
            top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)

        if self.debug:
            self._print_frequencies(top_k_indices, batch_size, seq_len)

        # Compute gating
        batch_gates, batch_index, expert_size, sorted_indices = self.compute_gating(
            self.top_k, probs, top_k_probs, top_k_indices
        )

        # First expert transformation
        # [batch, seq, in_features] -> [batch*seq, num_heads, expert_hidden]
        hidden = torch.einsum("bsi,nid->bsnd", x, self.input_experts)
        if self.input_bias is not None:
            hidden = hidden + self.input_bias

        # Apply activation
        hidden = F.gelu(hidden)  # ModuleFormer uses GELU

        # Second expert transformation
        # [batch*seq, num_heads, expert_hidden] -> [batch*seq, num_heads, head_dim]
        outputs = torch.einsum("bsnd,ndh->bsnh", hidden, self.output_experts)
        if self.output_bias is not None:
            outputs = outputs + self.output_bias

        # Prepare indices for gather
        gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        # Select outputs and apply gates
        selected_outputs = outputs.gather(2, gather_indices)
        if multiply_by_gates:
            selected_outputs = selected_outputs * top_k_probs.unsqueeze(-1)

        # Create zero tensor with full dimensions
        full_output = torch.zeros(
            batch_size, seq_len, self.num_heads, self.head_dim, device=x.device
        )

        # Scatter selected outputs into the full tensor
        full_output.scatter_(2, gather_indices, selected_outputs)

        # Handle aux loss
        if self.training:
            zeros = torch.zeros_like(probs)
            gates = zeros.scatter_(-1, top_k_indices, top_k_probs)
            self.update_aux_statistics(probs, logits, gates)
            aux_loss = self.get_aux_loss_and_clear()
        else:
            aux_loss = 0

        return (
            full_output.reshape(batch_size, seq_len, -1),
            aux_loss,
        )

    def compute_gating(
        self,
        top_k: int,
        probs: torch.Tensor,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        # Compute gating - reshape for gating computation
        flat_probs = probs.view(-1, self.num_heads)
        flat_top_k_probs = top_k_probs.view(-1, top_k)
        flat_top_k_indices = top_k_indices.view(-1, top_k)

        # Create dense gates tensor
        zeros = torch.zeros_like(flat_probs)
        gates = zeros.scatter(1, flat_top_k_indices, flat_top_k_probs)
        expert_size = gates.long().sum(0)

        # Flatten and sort
        flat_top_k_probs = flat_top_k_probs.reshape(-1)  # [batch * top_k]
        top_k_experts = flat_top_k_indices.reshape(-1)  # [batch * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)

        batch_index = index_sorted_experts.div(top_k, rounding_mode="trunc")
        batch_gates = flat_top_k_probs[index_sorted_experts]

        return batch_gates, batch_index, expert_size, index_sorted_experts

    def init_aux_statistics(self, window_size=5):
        self.window_size = window_size
        self.acc_count_queue = []  # Also track counts per batch
        self.acc_probs_queue = []
        self.acc_freq_queue = []
        self.acc_lsesq_queue = []  # New queue for lsesq

    def update_aux_statistics(self, probs, logits, gates):
        # Calculate batch count
        batch_count = logits.size(0)
        self.acc_count_queue.append(batch_count)

        # Calculate and append new statistics
        new_probs = probs.sum(0).sum(0)
        new_freq = (gates > 0).float().sum(0).sum(0)
        lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
        new_lsesq = lsesq.sum()

        self.acc_probs_queue.append(new_probs)
        self.acc_freq_queue.append(new_freq)
        self.acc_lsesq_queue.append(new_lsesq)

        # Remove oldest entries if beyond window size
        if len(self.acc_probs_queue) > self.window_size:
            self.acc_count_queue.pop(0)
            self.acc_probs_queue.pop(0)
            self.acc_freq_queue.pop(0)
            self.acc_lsesq_queue.pop(0)

    def get_aux_loss_and_clear(self):
        if not self.acc_probs_queue:
            return 0.0

        # Sum up counts and values within window
        total_count = sum(self.acc_count_queue)
        acc_probs_sum = torch.stack(self.acc_probs_queue).sum(0)
        acc_freq_sum = torch.stack(self.acc_freq_queue).sum(0)
        acc_lsesq_sum = sum(self.acc_lsesq_queue)  # Sum scalar values

        # Compute switch loss
        switchloss = (
            self.num_heads
            * (
                F.normalize(acc_probs_sum, p=1, dim=0)
                * F.normalize(acc_freq_sum, p=1, dim=0)
            ).sum()
        )

        # Compute z loss using windowed values
        zloss = acc_lsesq_sum / total_count

        return switchloss + 0.1 * zloss

    def compute_routing_weights(self, x):
        # Get routing embeddings with dropout
        z = self.router(x)

        # L2 normalize embeddings and centroids
        z_norm = F.normalize(z, p=2, dim=-1)
        centroids_norm = F.normalize(self.head_centroids, p=2, dim=-1)

        # Compute log posterior probabilities
        logits = torch.matmul(z_norm, centroids_norm.t())

        # Scale logits (using input dimension as in transformer attention)
        logits = logits / math.sqrt(z.size(-1))

        # Apply temperature
        logits = logits * self.temperature.exp()

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        return probs, logits

    def _print_frequencies(self, top_k_indices, batch_size, seq_len):
        if random.random() < self.log_frequency:
            expert_counts = torch.zeros(self.num_heads, device=top_k_indices.device)
            for i in range(self.num_heads):
                expert_counts[i] = (top_k_indices == i).sum().item()

            total_selections = batch_size * seq_len * self.top_k
            expert_frequencies = expert_counts / total_selections

            frequencies = [
                f"{i}: {freq:.2f}" for i, freq in enumerate(expert_frequencies)
            ]
            print(
                f"DEBUG: id={self.identifier}, mode={'train' if self.training else 'infer'}, frequencies=({', '.join(frequencies)})"
            )


if __name__ == "__main__":
    # Smoke tests
    torch.manual_seed(42)

    # Test initialization
    model = SparseQuery(
        in_features=512, num_heads=12, head_dim=64, top_k=4, hidden_size=256, bias=True
    )

    # Test forward pass
    batch_size = 32
    seq_length = 128
    x = torch.randn(batch_size, seq_length, 512)
    output, aux_loss = model(x)

    # Verify output shape reflects true sparsity
    expected_shape = (batch_size, seq_length, model.num_heads * model.head_dim)
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
