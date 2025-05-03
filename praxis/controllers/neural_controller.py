import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.stack import StackState, initialize, push_pop_nop, read
from torch import Tensor

from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class NeuralController(BaseController):
    """
    A neural routing module that uses NeuralLambda's stack to maintain memory of routing
    decisions while being compatible with the Pathfinder interface.

    This version includes gradient stabilization and tensor dimensionality preservation
    to avoid optimizer errors.
    """

    def __init__(self, config: ConfigType) -> None:
        super().__init__(config, allow_visualizer=True)

        # Core dimensions
        self.hidden_size = config.hidden_size
        self.stack_depth = min(16, self.depth * 2)
        self.emb_dim = self.hidden_size // 4

        # Tools (activation functions)
        self.tools = [
            F.relu,  # Tool 0: ReLU
            torch.tanh,  # Tool 1: Tanh
            torch.sigmoid,  # Tool 2: Sigmoid
            lambda x: x,  # Tool 3: Identity
        ]

        # Layer embeddings - unique representations for each layer
        self.layer_embeddings = nn.Parameter(torch.randn(self.depth, self.emb_dim))

        # Projection networks
        self.state_projector = nn.Linear(self.hidden_size, self.hidden_size // 2)

        # Separate decision networks for stack and tools
        self.stack_controller = nn.Linear(
            self.hidden_size // 2, 3
        )  # push, pop, null_op
        self.tool_selector = nn.Linear(self.hidden_size // 2, len(self.tools))

        # Routing decision network (combines information from state and memory)
        self.router = nn.Linear(self.hidden_size // 2 + self.emb_dim, self.num_experts)

        # Stack parameters - use a buffer instead of parameter to avoid scalar gradients
        self.sharpen_pointer = nn.Parameter(torch.tensor([5.0]))

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with appropriate scaling."""
        for module in [
            self.state_projector,
            self.stack_controller,
            self.tool_selector,
            self.router,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initialize_stack(self, batch_size: int, device: torch.device) -> StackState:
        """Initialize a new stack state with proper zeros."""
        return initialize(
            self.emb_dim,
            self.stack_depth,
            batch_size,
            0.0,  # zero offset
            device,
            dtype=torch.float32,
        )

    def _update_stack(
        self, stack_state: StackState, projected_state: Tensor, layer_embedding: Tensor
    ) -> StackState:
        """
        Update stack based on current state with gradient stabilization.
        """
        # Get stack operation probabilities
        stack_logits = self.stack_controller(projected_state)

        # Apply gradient stabilization before softmax
        stack_logits = torch.clamp(stack_logits, min=-15.0, max=15.0)
        stack_probs = F.softmax(stack_logits, dim=1)

        # Extract operation probabilities
        should_push = stack_probs[:, 0]
        should_pop = stack_probs[:, 1]
        should_null_op = stack_probs[:, 2]

        # Apply stack operation
        new_stack_state, _ = push_pop_nop(
            stack_state,
            self.sharpen_pointer,
            should_push,
            should_pop,
            should_null_op,
            layer_embedding,
        )

        return new_stack_state

    def _select_and_apply_tool(self, projected_state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Select and apply a tool (activation function) with gradient stability.
        """
        batch_size = projected_state.shape[0]
        device = projected_state.device

        # Get tool probabilities with stability measures
        tool_logits = self.tool_selector(projected_state)
        tool_logits = torch.clamp(tool_logits, min=-15.0, max=15.0)
        tool_probs = F.softmax(tool_logits, dim=1)

        # Select tool (discrete selection via argmax)
        tool_indices = torch.argmax(tool_probs, dim=1)

        # Apply selected tool to each batch item with gradient stability
        tool_outputs = projected_state.clone()  # Start with identity as fallback

        for idx in range(batch_size):
            # Apply tool
            tool_idx = tool_indices[idx].item()
            tool = self.tools[tool_idx]
            output = tool(projected_state[idx])
            tool_outputs[idx] = output

        return tool_indices, tool_outputs

    def _make_routing_decision(
        self, tool_outputs: Tensor, stack_memory: Tensor
    ) -> Tuple[int, Tensor]:
        """
        Determine the next expert with gradient stability.
        """

        # Combine tool output with stack memory for routing
        combined_features = torch.cat([tool_outputs, stack_memory], dim=1)

        # Calculate routing logits and probabilities with stability
        routing_logits = self.router(combined_features)
        routing_logits = torch.clamp(routing_logits, min=-15.0, max=15.0)
        gate_probs = F.softmax(routing_logits, dim=1)

        # Calculate entropy loss - stabilized to avoid NaN
        log_probs = torch.log(gate_probs + 1e-10)
        entropy = -(gate_probs * log_probs).sum(dim=1).mean()
        gating_loss = -0.01 * entropy.clamp(min=-10.0, max=10.0)

        # Get majority vote for routing decision
        batch_votes = torch.argmax(gate_probs, dim=1)
        vote_counts = torch.bincount(batch_votes, minlength=self.num_experts)
        next_expert_idx = torch.argmax(vote_counts).item()

        return next_expert_idx, gating_loss

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[
            Tuple[StackState, Optional[Tensor], Optional[Tensor]]
        ],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tuple[StackState, Tensor, Tensor], Tensor, List[int], Optional[int]]:
        """
        Determine the next expert/layer to route to with stabilized gradient operations.
        """
        batch_size, _, _ = hidden_states.shape
        device = hidden_states.device

        # Initialize or unpack controller state
        if controller_state is None:
            stack_state = self._initialize_stack(batch_size, device)
            tool_indices = None
            tool_outputs = None
        else:
            stack_state, tool_indices, tool_outputs = controller_state

        # Get embedding for current layer
        layer_embedding = (
            self.layer_embeddings[current_depth].unsqueeze(0).expand(batch_size, -1)
        )

        # Project hidden state (use last token for autoregressive models)
        last_hidden = hidden_states[:, -1, :]
        projected_state = self.state_projector(last_hidden)

        # 1. Update stack with layer information
        new_stack_state = self._update_stack(
            stack_state, projected_state, layer_embedding
        )

        # 2. Read from stack (memory of past routing)
        stack_memory = read(new_stack_state)

        # 3. Select and apply tool
        new_tool_indices, new_tool_outputs = self._select_and_apply_tool(
            projected_state
        )

        # 4. Make routing decision
        next_expert_idx, gating_loss = self._make_routing_decision(
            new_tool_outputs, stack_memory
        )

        # Update route for visualization
        current_route = self._update_route(
            hidden_states, current_route, current_depth, next_expert_idx
        )

        # Prepare controller state for next iteration
        new_controller_state = (new_stack_state, new_tool_indices, new_tool_outputs)

        if self.debug and random.random() < 0.005:
            output = self.visualize_operation(new_controller_state, batch_idx=0)
            print(str(output))

        return new_controller_state, gating_loss, current_route, next_expert_idx

    def visualize_operation(
        self,
        controller_state: Tuple[StackState, Optional[Tensor], Optional[Tensor]],
        batch_idx: int = 0,
    ) -> Dict:
        """Generate visualization data for understanding tool and stack usage."""
        stack_state, tool_indices, _ = controller_state

        # Get tool name with safety checks
        if tool_indices.numel() > batch_idx:
            tool_idx = tool_indices[batch_idx].item()
            tool_names = ["ReLU", "Tanh", "Sigmoid", "Identity"]
            tool_name = (
                tool_names[tool_idx]
                if tool_idx < len(tool_names)
                else f"Tool {tool_idx}"
            )
        else:
            tool_name = "Unknown"

        # Get stack pointer with safety checks
        if hasattr(stack_state, "pointer"):
            pointer = stack_state.pointer[batch_idx].detach().cpu().numpy()
        else:
            pointer = None

        # Return visualization data
        return {
            "tool_selected": tool_name,
            "stack_pointer": pointer,
            "sharpness": self.sharpen_pointer.item(),
        }


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
    router = NeuralController(config)

    # Test the router
    print("Testing NeuralController:")

    # Simulate routing through layers
    controller_state = None
    current_route = []
    current_depth = 0

    for step in range(10):  # Test routing for 10 steps
        print(f"Step {step}, Current Depth: {current_depth}")

        controller_state, gating_loss, current_route, next_expert_idx = (
            router.get_next_expert(
                hidden_states,
                controller_state,
                sequential_experts,
                ordered_experts,
                current_route,
                current_depth,
            )
        )

        if next_expert_idx is None:
            print("Early exit taken")
            break

        # Get visualization data
        viz_data = router.visualize_operation(controller_state)
        print(f"  Selected Expert: {next_expert_idx}")
        print(f"  Selected Tool: {viz_data.get('tool_selected', 'Unknown')}")
        print(f"  Gating Loss: {gating_loss.item():.6f}")
        print(f"  Current Route: {current_route}")

        current_depth = next_expert_idx

    print("\nFinal route:", current_route)
