import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.modules.controller import PraxisController
from praxis.modules.graph import PraxisGraph
from praxis.modules.visualization import RouteVisualizer


class LayerShuffle(nn.Module):
    """
    This module implements a basic form of LayerShuffle-Position, though we use it
    as a differentiable "context token" and input-manipulation/preparation mechanism,
    rather than a positional encoder.
    https://arxiv.org/abs/2407.04513
    """

    def __init__(self, config: "AutoConfig", num_context_tokens: int = 0):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.current_route = []
        assert (
            config.num_experts == config.depth
        ), "There is no point in making `num_experts` greater than or less than `depth`, when `shuffle != True`. The additional experts would never be used."

        self.navigator = (
            PraxisController(config, max_num_experts=config.num_experts * 3)
            if config.autopilot
            else False
        )

        self.num_context_tokens = num_context_tokens
        if self.num_context_tokens < 1:
            return

        # Keep learned embeddings for each position and context token
        self.embeddings = nn.Parameter(
            torch.randn(config.depth, self.num_context_tokens, config.hidden_size)
        )
        # Initialize with small values for stability
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def add_context(
        self, hidden_states: Tensor, attention_mask: Tensor, position: int
    ) -> tuple[Tensor, Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Get position-based embeddings
        context = self.embeddings[position]  # [num_tokens, dims]
        # Expand to match batch dimension
        context = context.expand(hidden_states.shape[0], -1, -1)

        # Prepare attention mask for context tokens
        extended_attention_mask = None
        if attention_mask is not None:
            context_mask = attention_mask.new_ones(
                attention_mask.shape[0], self.num_context_tokens
            )
            extended_attention_mask = torch.cat([context_mask, attention_mask], dim=1)

        # Add context to hidden states
        extended_hidden_states = torch.cat([context, hidden_states], dim=1)

        return extended_hidden_states, extended_attention_mask

    def remove_context(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Remove context tokens from both hidden states and attention mask
        trimmed_states = hidden_states[:, self.num_context_tokens :, :]

        trimmed_mask = None
        if attention_mask is not None:
            trimmed_mask = attention_mask[:, self.num_context_tokens :]
        return trimmed_states, trimmed_mask

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        depth = self.depth
        if allow_resampling:
            return random.choices(experts, k=depth)
        else:
            return random.sample(experts, k=depth)

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Compute next expert selection and associated loss.
        During inference, returns actual expert index.
        """

        # Record the original index in the route
        original_idx = original_experts.index(current_expert)
        self.current_route.append(original_idx)

        if self.navigator:
            return self.navigator(
                hidden_states,
                current_depth,
                original_experts,
                current_experts,
                current_expert,
            )
        else:
            # Look up the next expert in the shuffled sequence
            if current_depth + 1 < len(current_experts):
                # The next expert we want is at current_depth + 1 in the shuffled list
                next_expert = current_experts[current_depth + 1]
                # Return its position in the shuffled list as next_expert_idx
                return 0, current_experts.index(next_expert)
            else:
                return 0, None

    def reset_route(self):
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training:
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        self.current_route = []


class GatedRouter(nn.Module):
    """
    Implements a gating mechanism for dynamic layer selection in transformer models.
    Each layer uses a gating network to decide which layer to process next based on
    the current hidden state.
    """

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.current_route = []

        # Create a gating network for each layer to decide the next layer
        self.gates = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.depth) for _ in range(config.depth)]
        )

    def add_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def remove_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        """No-op to maintain API compatibility."""
        return experts

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Determine the next layer to process using a gating mechanism.

        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_size]
            current_depth: Current layer depth
            original_experts: List of original experts/layers
            current_experts: List of current experts/layers
            current_expert: Current expert/layer

        Returns:
            Tuple of (gating_loss, next_expert_idx)
        """
        # Record the current layer in the route
        original_idx = original_experts.index(current_expert)
        self.current_route.append(original_idx)

        # Check if we've reached the maximum depth
        if current_depth + 1 >= self.depth:
            return torch.tensor(0.0, device=hidden_states.device), None

        # Pool the hidden states - using mean pooling for simplicity
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Apply the gating network for the current layer
        gate_logits = self.gates[current_depth](pooled_hidden)  # [batch_size, depth]

        # Compute next layer probabilities
        gate_probs = F.softmax(gate_logits, dim=1)

        # For training stability, compute a small entropy loss
        # This encourages exploration of different routing paths
        if self.training:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
            gating_loss = -0.01 * entropy  # Encourage exploration with negative loss
        else:
            gating_loss = torch.tensor(0.0, device=hidden_states.device)

        # Select the next layer (expert) to process
        next_expert_idx = torch.argmax(gate_probs, dim=1)[0].item()

        return gating_loss, next_expert_idx

    def reset_route(self):
        """Reset the tracking of the current route through layers."""
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training:
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        self.current_route = []


class MixtureRouter(nn.Module):
    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth

        # Depth embedding matching the hidden size of a token
        self.depth_embedding = nn.Embedding(config.depth, config.hidden_size)

        # Final linear layer to produce routing logits
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        self.current_route = []

        self.visualizer = (
            RouteVisualizer(
                num_experts=config.num_experts,
                max_history=10000,
                save_rate=100 * config.depth,
            )
            if self.debug
            else False
        )

    def get_next_expert(
        self,
        hidden_states: Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> Tuple[Tensor, Optional[int]]:

        # Record the current expert index
        current_idx = original_experts.index(current_expert)
        self.current_route.append(current_idx)

        available_indices = [
            i for i, expert in enumerate(original_experts) if expert in current_experts
        ]

        if not available_indices:
            return 0, None

        batch_size, seq_len, hidden_size = hidden_states.size()

        # Get depth embeddings
        depth_tensor = torch.full(
            (batch_size,), current_depth, dtype=torch.long, device=hidden_states.device
        )  # Shape: [B]
        depth_emb = self.depth_embedding(depth_tensor)  # Shape: [B, hidden_size]

        # Reshape depth embedding to [B, 1, hidden_size]
        depth_emb = depth_emb.unsqueeze(1)  # Shape: [B, 1, hidden_size]

        # Prepend depth embedding to the hidden states
        extended_sequence = torch.cat(
            [depth_emb, hidden_states], dim=1
        )  # Shape: [B, S+1, hidden_size]

        # Pass the full sequence through the router layer
        router_outputs = self.router(extended_sequence)  # Shape: [B, S+1, router_dim]

        # Perform pooling after the router processes the sequence
        sequence_representation = router_outputs.mean(dim=1)  # Shape: [B, router_dim]

        router_probs = F.softmax(
            sequence_representation, dim=-1
        )  # Shape: [B, num_experts]

        # Compute expert usage over the batch
        expert_usage = router_probs.mean(dim=0)  # Shape: [num_experts]

        # Compute KL divergence loss to encourage balanced expert usage
        num_experts = expert_usage.size(0)
        uniform_distribution = torch.full_like(expert_usage, 1.0 / num_experts)
        kl_loss = F.kl_div(
            torch.log(expert_usage + 1e-9),  # To avoid log(0)
            uniform_distribution,
            reduction="sum",
        )

        # Scale the KL divergence loss based on current depth
        base_loss_coefficient = 0.01
        loss_coefficient = base_loss_coefficient * (current_depth / self.depth)
        aux_loss = loss_coefficient * kl_loss

        # Select expert with highest probability for each sample in the batch
        expert_indices = router_probs.argmax(dim=-1)  # Shape: [B]

        # Optionally, select the most common expert among the batch
        next_idx = expert_indices.mode().values.item()

        # Update route
        if not self.training and self.visualizer and hidden_states.size(0) == 1:
            # Send the immediate transition to the visualizer
            self.visualizer.add_transition(current_idx, next_idx)

        return aux_loss, next_idx

    def reset_route(self):
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training:
                print(f"DEBUG: inferencing through: {' -> '.join(route)}")
            elif random.random() < 0.005:
                print(f"DEBUG: training through: {' -> '.join(route)}")
        self.current_route = []

    # No-op methods
    def add_context(self, hidden_states, attention_mask, position):
        return hidden_states, attention_mask

    def remove_context(self, hidden_states, attention_mask):
        return hidden_states, attention_mask


if __name__ == "__main__":
    # Mock config
    class MockConfig:
        def __init__(self):
            self.hidden_size = 512
            self.num_experts = 8
            self.debug = False
            self.depth = 5

    # Test settings
    batch_size = 4
    seq_length = 16
    hidden_dim = 512
    num_experts = 8

    # Create mock data
    config = MockConfig()
    hidden_states = torch.randn(batch_size, seq_length, hidden_dim)
    mock_experts = [nn.Module() for _ in range(num_experts)]

    # Initialize router
    router = MixtureRouter(config)

    try:
        # Test basic forward pass
        loss, next_idx = router.get_next_expert(
            hidden_states=hidden_states,
            current_depth=0,
            original_experts=mock_experts,
            current_experts=mock_experts[:-1],  # Test with one expert unavailable
            current_expert=mock_experts[0],
        )

        print("✓ Basic forward pass successful")
        print(f"Selected expert index: {next_idx}")
        print(f"Routing loss: {loss.item():.4f}")

        # Test gradient flow
        loss.backward()
        print("✓ Gradient computation successful")

        # Test with empty expert list
        loss, next_idx = router.get_next_expert(
            hidden_states=hidden_states,
            current_depth=0,
            original_experts=mock_experts,
            current_experts=[],
            current_expert=mock_experts[0],
        )
        assert next_idx is None
        print("✓ Empty expert list handling successful")

        # Test no-op methods
        states, mask = router.add_context(
            hidden_states, torch.ones(batch_size, seq_length), 0
        )
        assert torch.equal(states, hidden_states)
        print("✓ No-op methods successful")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        raise
