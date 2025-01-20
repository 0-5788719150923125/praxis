import math

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
        bias: bool = True,
        acc_aux_loss: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = min(top_k, num_heads)
        self.acc_aux_loss = acc_aux_loss

        # Output features will be top_k * head_dim, not num_heads * head_dim
        self.out_features = self.top_k * head_dim

        # Projection for routing embeddings
        self.router = nn.Linear(in_features, hidden_size, bias=False)

        # Learned head centroids
        self.head_centroids = nn.Parameter(torch.empty(num_heads, hidden_size))

        # Temperature parameter for routing
        self.temperature = nn.Parameter(torch.zeros(1))

        # Main projection weights - one for each possible head
        self.weight = nn.Parameter(torch.empty(num_heads, in_features, head_dim))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_heads, head_dim))
        else:
            self.register_parameter("bias", None)

        # Initialize auxiliary loss statistics
        self.init_aux_statistics()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.head_centroids, std=0.01)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def init_aux_statistics(self):
        self.p_e = 0.0
        self.neg_H_e_given_x = 0.0
        self.count_calls = 0

    def compute_routing_weights(self, x):
        # Get routing embeddings
        z = self.router(x)  # [batch_size, hidden_size]

        # Normalize embeddings and centroids
        z_norm = F.normalize(z, p=2, dim=-1)
        centroids_norm = F.normalize(self.head_centroids, p=2, dim=-1)

        # Compute cosine similarities and scale by temperature
        logits = torch.matmul(z_norm, centroids_norm.t()) * self.temperature.exp()

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Update auxiliary statistics if needed
        if self.training and self.acc_aux_loss:
            log_probs = F.log_softmax(logits, dim=-1)
            self.p_e = self.p_e + probs.mean(0)
            self.neg_H_e_given_x = self.neg_H_e_given_x + (
                probs * log_probs
            ).sum() / probs.size(0)
            self.count_calls += 1

        return probs, logits

    def get_aux_loss_and_clear(self, eps=1e-8):
        if self.count_calls == 0:
            return 0.0

        p_e = self.p_e / self.count_calls
        H_e = -(p_e * (p_e + eps).log()).sum()
        neg_H_e_given_x = self.neg_H_e_given_x / self.count_calls
        mi_loss = -(neg_H_e_given_x + H_e)

        self.init_aux_statistics()
        return mi_loss

    def forward(self, x):
        batch_size, *rest, _ = x.shape
        x_flat = x.view(-1, self.in_features)
        flat_size = batch_size * math.prod(rest)

        # Compute routing probabilities
        probs, _ = self.compute_routing_weights(x_flat)

        # Select top-k heads and their probabilities
        top_k_probs, top_k_indices = probs.topk(self.top_k, dim=-1)

        # Normalize selected probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)

        # Gather only the selected head weights and biases
        selected_weights = self.weight[
            top_k_indices
        ]  # [flat_size, top_k, in_features, head_dim]

        if self.bias is not None:
            selected_bias = self.bias[top_k_indices]  # [flat_size, top_k, head_dim]
        else:
            selected_bias = None

        # Compute outputs only for selected heads
        x_expanded = x_flat.unsqueeze(1).unsqueeze(-2)  # [flat_size, 1, 1, in_features]
        outputs = torch.matmul(x_expanded, selected_weights).squeeze(
            -2
        )  # [flat_size, top_k, head_dim]

        if self.bias is not None:
            outputs = outputs + selected_bias

        # Weight outputs by routing probabilities
        outputs = outputs * top_k_probs.unsqueeze(-1)  # [flat_size, top_k, head_dim]

        # Reshape to combine top_k and head_dim dimensions
        outputs = outputs.reshape(flat_size, -1)  # [flat_size, top_k * head_dim]

        # Reshape to final output shape
        output_shape = [batch_size] + list(rest) + [self.out_features]
        return outputs.view(output_shape)


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
    output = model(x)

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
    _ = model(x)  # Accumulate statistics
    aux_loss = model.get_aux_loss_and_clear()
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
    full_params = model.num_heads * model.in_features * model.head_dim
    actual_params = model.weight.numel()
    print(f"Parameters per layer: {actual_params:,}")
    print(
        f"Effective parameters used per forward pass: {model.top_k * model.in_features * model.head_dim:,}"
    )
