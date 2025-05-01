from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import neurallambda components
from neurallambda.stack import StackState, push_pop_nop
from neurallambda.torch import cosine_similarity


class NeuralStackWithTools(nn.Module):
    """
    A simplified neural reasoning module that combines:
    1. NeuralLambda's differentiable stack for memory
    2. A tool selection mechanism for applying different activation functions
    """

    def __init__(self, hidden_size: int, stack_depth: int = 10, num_tools: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.stack_depth = stack_depth

        # Combined projection for input transformations
        self.projection = nn.Linear(hidden_size, hidden_size * 2)

        # Stack operation controller (push, pop, null_op)
        self.op_controller = nn.Linear(hidden_size, 3)

        # Tool selector
        self.tool_selector = nn.Linear(hidden_size, num_tools)

        # Layer normalization for the output
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Parameter for pointer sharpening
        self.sharpen_pointer = 5.0

        # Define activation functions as tools
        self.tools = [
            F.relu,  # Tool 0: ReLU
            F.gelu,  # Tool 1: GELU
            torch.tanh,  # Tool 2: Tanh
            torch.sigmoid,  # Tool 3: Sigmoid
            lambda x: x,  # Tool 4: Identity (no activation)
        ]

        # Ensure we have the right number of tools
        assert (
            len(self.tools) == num_tools
        ), f"Expected {num_tools} tools, got {len(self.tools)}"

    def forward(
        self, hidden_states: torch.Tensor, stack_state: Optional[StackState] = None
    ) -> Tuple[torch.Tensor, StackState, torch.Tensor]:
        """
        Apply neural stack reasoning with tools to transformer hidden states.

        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            stack_state: Previous stack state (optional)

        Returns:
            output: Transformed hidden states
            new_stack_state: Updated stack state for next forward pass
            tool_weights: Tool selection weights for each position [batch_size, seq_len, num_tools]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Initialize stack state if not provided
        if stack_state is None:
            # Initialize pointer
            pointer = torch.zeros((batch_size, self.stack_depth), device=device)
            pointer[:, 0] = 1.0  # Initial position points to first element

            # Create stack and zero vector
            stack = torch.zeros(
                (batch_size, self.stack_depth, self.hidden_size), device=device
            )
            zero_vec = torch.zeros((batch_size, self.hidden_size), device=device)

            # Create StackState
            stack_state = StackState(stack, pointer, zero_vec)

        # Keep track of outputs and tool weights
        outputs = []
        tool_weights_seq = []
        current_stack_state = stack_state

        # Process each position in the sequence
        for t in range(seq_len):
            # Get current token representation
            x_t = hidden_states[:, t, :]

            # Apply combined projection for value and hidden transformation
            projected = self.projection(x_t)
            h, value = torch.split(projected, self.hidden_size, dim=-1)

            # Determine stack operations
            op_logits = self.op_controller(
                x_t
            )  # Use original input to decide operations
            op_probs = F.softmax(op_logits, dim=-1)
            should_push = op_probs[:, 0]  # [batch_size]
            should_pop = op_probs[:, 1]  # [batch_size]
            should_null_op = op_probs[:, 2]  # [batch_size]

            # Apply stack operations using neurallambda's push_pop_nop
            new_stack_state, _ = push_pop_nop(
                current_stack_state,
                self.sharpen_pointer,
                should_push,
                should_pop,
                should_null_op,
                value,
            )

            # Read from the top of the stack
            stack_top = torch.sum(
                new_stack_state.stack * new_stack_state.pointer.unsqueeze(-1), dim=1
            )

            # Select tools (activation functions) to apply
            tool_logits = self.tool_selector(h)
            tool_weights = F.softmax(tool_logits, dim=-1)  # [batch_size, num_tools]
            tool_weights_seq.append(tool_weights)

            # Apply weighted combination of tools to stack output
            processed = torch.zeros_like(stack_top)
            for i, tool in enumerate(self.tools):
                tool_output = tool(stack_top)
                processed += tool_output * tool_weights[:, i].unsqueeze(1)

            # Combine with hidden state via addition (simpler than another projection)
            output = h + processed
            outputs.append(output)

            # Update stack state for next token
            current_stack_state = new_stack_state

        # Combine outputs along sequence dimension
        combined_outputs = torch.stack(outputs, dim=1)

        # Apply layer normalization with residual connection
        final_output = self.layer_norm(combined_outputs + hidden_states)

        # Stack tool weights for all positions
        all_tool_weights = torch.stack(
            tool_weights_seq, dim=1
        )  # [batch_size, seq_len, num_tools]

        return final_output, current_stack_state, all_tool_weights


# Simple test for the module
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 2
    seq_len = 5
    hidden_size = 256

    # Create sample input (simulating decoder hidden states)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Create the neural stack with tools module
    reasoning_module = NeuralStackWithTools(hidden_size)

    # Run a forward pass
    output, stack_state, tool_weights = reasoning_module(hidden_states)

    # Print shapes to verify
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Stack shape: {stack_state.stack.shape}")
    print(f"Pointer shape: {stack_state.pointer.shape}")
    print(f"Tool weights shape: {tool_weights.shape}")

    # Print average tool usage across the sequence
    avg_tool_weights = tool_weights.mean(dim=(0, 1))
    print("\nAverage tool usage:")
    for i, weight in enumerate(avg_tool_weights):
        tool_name = ["ReLU", "GELU", "Tanh", "Sigmoid", "Identity"][i]
        print(f"  Tool {i} ({tool_name}): {weight.item():.4f}")

    # Verify output differences
    input_output_diff = (output - hidden_states).abs().mean().item()
    print(
        f"\nMean absolute difference between input and output: {input_output_diff:.6f}"
    )
