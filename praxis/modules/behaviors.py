import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

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

    def __init__(self, config: AutoConfig, num_context_tokens: int = 1):
        super().__init__()
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
            torch.randn(config.depth, num_context_tokens, config.num_dims)
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
        trimmed_mask = attention_mask[:, self.num_context_tokens :]
        return trimmed_states, trimmed_mask

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        depth = self.embeddings.shape[0]
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
        if self.navigator:
            return self.navigator(
                hidden_states,
                current_depth,
                original_experts,
                current_experts,
                current_expert,
            )
        else:
            if -len(current_experts) <= current_depth + 1 < len(current_experts):
                return 0, original_experts.index(current_experts[current_depth + 1])
            else:
                return 0, None


class MixtureRouter(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.debug = config.debug
        # Reduce hidden dim to something smaller
        reduced_dim = config.num_dims // 4

        # Transform hidden states before routing
        self.transform = nn.Sequential(
            nn.LayerNorm(config.num_dims),
            nn.Linear(config.num_dims, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, reduced_dim),
        )

        # Router projects from reduced dim to num experts
        self.router = nn.Linear(reduced_dim, config.num_experts)
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
    ) -> tuple[Tensor, Optional[int]]:

        # Reset used experts at the start of new sequence
        current_idx = original_experts.index(current_expert)
        self.current_route.append(current_idx)

        available_indices = [
            i for i, expert in enumerate(original_experts) if expert in current_experts
        ]

        if not available_indices:
            return 0, None

        # Transform and reduce sequence dimension
        transformed = self.transform(hidden_states)  # [B, S, reduced_dim]
        batch_reduced = transformed.sum(dim=1)  # [B, reduced_dim]

        # Get router logits
        router_logits = self.router(batch_reduced)  # [B, num_experts]

        # Mask unavailable experts
        # mask = torch.ones_like(router_logits, dtype=torch.bool)
        # mask[:, available_indices] = False
        # router_logits = router_logits.masked_fill(mask, -1e9)

        # Select single expert
        expert_weights, expert_indices = torch.topk(
            router_logits, k=1, dim=-1, sorted=False
        )

        # Compute balancing loss
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, expert_indices, 1.0)
        aux_loss = F.binary_cross_entropy_with_logits(router_logits, router_targets)

        # Return batch consensus
        next_idx = expert_indices.mode(dim=0).values.item()

        # Update route
        if not self.training and self.visualizer and hidden_states.size(0) == 1:
            # Just send the immediate transition
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
            self.num_dims = 512
            self.num_experts = 8
            self.debug = False

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
