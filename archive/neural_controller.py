from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import NeuralLambda components
from neurallambda.stack import StackState, push_pop_nop
from neurallambda.torch import cosine_similarity
from torch import Tensor

from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class NeuralLambdaRouter(BaseController):
    """
    A neural routing module that uses NeuralLambda's stack to maintain memory of routing
    decisions while being compatible with the Pathfinder interface.

    This module enhances routing decisions by using a differentiable stack to track
    layer visitation history and applying different activation functions as tools.
    """

    def __init__(self, config: ConfigType, allow_early_exits: bool = False) -> None:
        super().__init__(config, allow_visualizer=True)

        # Retrieve dimensions from config
        self.hidden_size = config.hidden_size
        self.stack_depth = min(16, self.depth * 2)  # Allow for some revisits
        self.num_tools = 4  # Number of activation functions to use as tools
        self.allow_early_exits = allow_early_exits

        # Number of possible outputs
        self.extra_vectors = int(allow_early_exits)

        # Embedding for each layer
        self.layer_embeddings = nn.Parameter(
            torch.randn(self.depth, self.hidden_size // 4)
        )

        # Projection for hidden states
        self.state_projector = nn.Linear(self.hidden_size, self.hidden_size // 2)

        # Tool selector - chooses which activation to apply
        self.tool_selector = nn.Linear(self.hidden_size // 2, self.num_tools)

        # Routing decision layer
        self.router = nn.Linear(
            self.hidden_size // 2 + self.hidden_size // 4,
            self.num_experts + self.extra_vectors,
        )

        # Activation functions as tools
        self.tools = [
            F.relu,  # Tool 0: ReLU
            torch.tanh,  # Tool 1: Tanh
            torch.sigmoid,  # Tool 2: Sigmoid
            lambda x: x,  # Tool 3: Identity
        ]

        # Parameter for pointer sharpening in stack
        self.sharpen_pointer = 5.0

        # Store stack states between calls
        self.stack_states = {}

    def initialize_stack(self, batch_size: int, device: torch.device) -> StackState:
        """Initialize a new stack state."""
        # Create initial pointer (focused on position 0)
        pointer = torch.zeros((batch_size, self.stack_depth), device=device)
        pointer[:, 0] = 1.0

        # Create empty stack
        stack = torch.zeros(
            (batch_size, self.stack_depth, self.hidden_size // 4), device=device
        )

        # Zero vector for empty stack positions
        zero_vec = torch.zeros((batch_size, self.hidden_size // 4), device=device)

        return StackState(stack, pointer, zero_vec)

    def get_next_expert(
        self,
        hidden_states: Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, List[int], Optional[int]]:
        """
        Determine the next expert/layer to route to, following the Pathfinder interface.

        Args:
            hidden_states: The input tensor [batch_size, seq_len, hidden_size]
            sequential_experts: List of sequential expert modules
            ordered_experts: List of ordered expert modules
            current_route: The current route list
            current_depth: The current depth/layer index

        Returns:
            gating_loss: Loss tensor for auxiliary loss
            updated_route: Updated route list
            next_expert_idx: Index of the next expert, or None for early exit
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        batch_key = f"batch_{id(hidden_states)}"

        # Get or initialize stack state
        if batch_key not in self.stack_states:
            self.stack_states[batch_key] = self.initialize_stack(batch_size, device)

        stack_state = self.stack_states[batch_key]

        # Get the embedding for the current layer
        layer_embedding = (
            self.layer_embeddings[current_depth].unsqueeze(0).expand(batch_size, -1)
        )

        # Mean-pool the hidden states
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Project hidden states
        projected_hidden = self.state_projector(
            pooled_hidden
        )  # [batch_size, hidden_size//2]

        # Push the current layer embedding to the stack
        # All examples will perform a push operation
        should_push = torch.ones(batch_size, device=device)
        should_pop = torch.zeros(batch_size, device=device)
        should_null_op = torch.zeros(batch_size, device=device)

        # Update the stack with the current layer embedding
        new_stack_state, _ = push_pop_nop(
            stack_state,
            self.sharpen_pointer,
            should_push,
            should_pop,
            should_null_op,
            layer_embedding,
        )

        # Read from the top of the stack by summing over the weighted stack
        stack_info = torch.sum(
            new_stack_state.stack * new_stack_state.pointer.unsqueeze(-1), dim=1
        )  # [batch_size, hidden_size//4]

        # Determine which tool (activation function) to use
        tool_logits = self.tool_selector(projected_hidden)
        tool_weights = F.softmax(tool_logits, dim=1)  # [batch_size, num_tools]

        # Apply tools to the projected hidden state
        processed_hidden = torch.zeros_like(projected_hidden)
        for i, tool in enumerate(self.tools):
            tool_output = tool(projected_hidden)
            processed_hidden += tool_output * tool_weights[:, i].unsqueeze(1)

        # Combine processed hidden state with stack information for routing decision
        combined_features = torch.cat([processed_hidden, stack_info], dim=1)

        # Get routing logits
        routing_logits = self.router(
            combined_features
        )  # [batch_size, num_experts + extra]

        # Compute routing probabilities
        gate_probs = F.softmax(routing_logits, dim=1)

        # Calculate entropy loss for training (similar to original Pathfinder)
        entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
        gating_loss = -0.01 * entropy  # Encourage exploration

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(gate_probs, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(
            batch_votes, minlength=self.num_experts + self.extra_vectors
        )
        next_expert_idx = torch.argmax(vote_counts).item()

        # Store updated stack state for next call
        self.stack_states[batch_key] = new_stack_state

        # Check for early exit
        if self.allow_early_exits and next_expert_idx == self.num_experts:
            return gating_loss, current_route, None

        return gating_loss, current_route, next_expert_idx


# Simple test for the module
if __name__ == "__main__":
    # Mock AutoConfig class for testing
    class AutoConfig:
        def __init__(self):
            self.debug = False
            self.hidden_size = 256
            self.dropout = 0.1
            self.depth = 8
            self.num_experts = 8

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 2
    seq_len = 5
    hidden_size = 256

    # Create sample input (simulating hidden states)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create mock config and sequential experts
    config = AutoConfig()
    sequential_experts = [
        nn.Linear(hidden_size, hidden_size) for _ in range(config.num_experts)
    ]
    ordered_experts = sequential_experts.copy()

    # Create the router module
    router = NeuralLambdaRouter(config, allow_early_exits=True)

    # Test the router
    print("Testing NeuralLambdaRouter with Pathfinder interface:")

    # Simulate routing through layers
    current_route = []
    current_depth = 0

    for step in range(10):  # Test routing for 10 steps
        print(f"Step {step}, Current Depth: {current_depth}")

        gating_loss, current_route, next_expert_idx = router.get_next_expert(
            hidden_states,
            sequential_experts,
            ordered_experts,
            current_route,
            current_depth,
        )

        if next_expert_idx is None:
            print("Early exit taken")
            break

        print(f"  Selected Expert: {next_expert_idx}")
        print(f"  Gating Loss: {gating_loss.item():.6f}")
        print(f"  Current Route: {current_route}")

        current_depth = next_expert_idx

    print("\nFinal route:", current_route)
