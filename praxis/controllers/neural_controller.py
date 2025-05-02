from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from neurallambda.stack import StackState, push_pop_nop
from neurallambda.torch import cosine_similarity
from torch import Tensor

from praxis.controllers.base import BaseController


class ToolStackState:
    """Extended stack state that also includes information about the selected tool."""

    def __init__(
        self,
        stack_state: StackState,
        tool_idx: Optional[Tensor] = None,
        tool_output: Optional[Tensor] = None,
    ):
        self.stack_state = stack_state
        self.tool_idx = tool_idx  # [batch_size]
        self.tool_output = tool_output  # [batch_size, hidden_size]


class NeuralController(BaseController):
    """
    An enhanced neural controller that deeply integrates tool selection
    with NeuralLambda's stack operations.
    """

    def __init__(self, config: "AutoConfig") -> None:
        super().__init__(config, allow_visualizer=True)

        # Retrieve dimensions from config
        self.hidden_size = config.hidden_size
        self.stack_depth = min(16, self.depth * 2)

        # Define tools (activation functions)
        self.tools = [
            F.relu,  # Tool 0: ReLU
            torch.tanh,  # Tool 1: Tanh
            torch.sigmoid,  # Tool 2: Sigmoid
            lambda x: x,  # Tool 3: Identity (pass-through)
        ]

        # Embedding for each layer
        self.layer_embeddings = nn.Parameter(
            torch.randn(self.depth, self.hidden_size // 4)
        )

        # Projection for hidden states
        self.state_projector = nn.Linear(self.hidden_size, self.hidden_size // 2)

        # Combined operation controller (stack ops + tools in one decision)
        # 3 stack operations + len(tools) tool operations
        self.op_controller = nn.Linear(self.hidden_size // 2, 3 + len(self.tools))

        # Routing decision network
        self.router = nn.Linear(
            self.hidden_size // 2 + self.hidden_size // 4, self.num_experts
        )

        # Parameter for pointer sharpening in stack
        self.sharpen_pointer = 5.0

    def _initialize_stack(
        self, batch_size: int, device: torch.device
    ) -> ToolStackState:
        """Initialize a new stack state with tool information."""
        # Create initial pointer (focused on position 0)
        pointer = torch.zeros((batch_size, self.stack_depth), device=device)
        pointer[:, 0] = 1.0

        # Create empty stack
        stack = torch.zeros(
            (batch_size, self.stack_depth, self.hidden_size // 4), device=device
        )

        # Zero vector for empty stack positions
        zero_vec = torch.zeros((batch_size, self.hidden_size // 4), device=device)

        # Create basic stack state
        stack_state = StackState(stack, pointer, zero_vec)

        # Initialize with no tool selected
        return ToolStackState(stack_state)

    def _apply_tool_and_stack_op(
        self,
        tool_stack_state: ToolStackState,
        op_probs: Tensor,  # [batch_size, 3+num_tools]
        value: Tensor,  # [batch_size, hidden_size//4]
        projected_hidden: Tensor,  # [batch_size, hidden_size//2]
    ) -> ToolStackState:
        """
        Apply combined stack operation and tool selection.
        """
        batch_size = op_probs.shape[0]
        device = op_probs.device

        # Split probabilities into stack ops and tool selection
        stack_op_probs = op_probs[:, :3]  # [batch_size, 3]
        tool_probs = op_probs[:, 3:]  # [batch_size, num_tools]

        # Normalize probabilities for each group
        stack_op_probs = F.softmax(stack_op_probs, dim=1)
        tool_probs = F.softmax(tool_probs, dim=1)

        # Extract operation probabilities
        should_push = stack_op_probs[:, 0]
        should_pop = stack_op_probs[:, 1]
        should_null_op = stack_op_probs[:, 2]

        # Apply stack operation
        new_stack_state, _ = push_pop_nop(
            tool_stack_state.stack_state,
            self.sharpen_pointer,
            should_push,
            should_pop,
            should_null_op,
            value,
        )

        # Select tool (discrete selection)
        tool_indices = torch.argmax(tool_probs, dim=1)  # [batch_size]

        # Apply the selected tool for each batch item
        tool_outputs = torch.zeros_like(projected_hidden)
        for idx in range(batch_size):
            tool_idx = tool_indices[idx].item()
            tool = self.tools[tool_idx]
            tool_outputs[idx] = tool(projected_hidden[idx])

        # Return updated state with tool information
        return ToolStackState(
            stack_state=new_stack_state, tool_idx=tool_indices, tool_output=tool_outputs
        )

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Optional[ToolStackState],
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[ToolStackState, Tensor, List[int], Optional[int]]:
        """
        Determine the next expert/layer to route to using the integrated tool-stack approach.
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Get or initialize tool stack state
        if controller_state is None:
            controller_state = self._initialize_stack(batch_size, device)

        # Get the embedding for the current layer
        layer_embedding = (
            self.layer_embeddings[current_depth].unsqueeze(0).expand(batch_size, -1)
        )

        # Use the last hidden state (instead of mean pooling)
        reduced_hidden = hidden_states[:, -1, :]
        projected_hidden = self.state_projector(reduced_hidden)

        # Get combined operation probabilities (stack ops + tools)
        op_logits = self.op_controller(projected_hidden)
        op_probs = F.softmax(op_logits, dim=1)

        # Apply stack operation and tool selection
        new_controller_state = self._apply_tool_and_stack_op(
            controller_state, op_probs, layer_embedding, projected_hidden
        )

        # Read from the top of the stack
        stack_info = torch.sum(
            new_controller_state.stack_state.stack
            * new_controller_state.stack_state.pointer.unsqueeze(-1),
            dim=1,
        )

        # Get the tool output from the controller state
        tool_output = new_controller_state.tool_output

        # Combine tool output with stack information for routing decision
        combined_features = torch.cat([tool_output, stack_info], dim=1)

        # Get routing logits and probabilities
        routing_logits = self.router(combined_features)
        gate_probs = F.softmax(routing_logits, dim=1)

        # Calculate entropy loss for training
        entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
        gating_loss = -0.01 * entropy  # Encourage exploration

        # Get batch majority vote for next expert
        batch_votes = torch.argmax(gate_probs, dim=1)
        vote_counts = torch.bincount(batch_votes, minlength=self.num_experts)
        next_expert_idx = torch.argmax(vote_counts).item()

        # Update route
        current_route = self._update_route(
            hidden_states, current_route, current_depth, next_expert_idx
        )

        return new_controller_state, gating_loss, current_route, next_expert_idx

    def visualize_operation(
        self, controller_state: ToolStackState, batch_idx: int = 0
    ) -> Dict:
        """Generate visualization data for understanding tool and stack usage."""
        if controller_state is None or controller_state.tool_idx is None:
            return {}

        # Get tool name
        tool_idx = controller_state.tool_idx[batch_idx].item()
        tool_names = ["ReLU", "Tanh", "Sigmoid", "Identity"]
        tool_name = (
            tool_names[tool_idx] if tool_idx < len(tool_names) else f"Tool {tool_idx}"
        )

        # Get stack pointer
        pointer = controller_state.stack_state.pointer[batch_idx].detach().cpu().numpy()

        # Return visualization data
        return {
            "tool_selected": tool_name,
            "stack_pointer": pointer,
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

    # Set random seed for reproducibility
    torch.manual_seed(5555)

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
    print("Testing NeuralToolController:")

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
