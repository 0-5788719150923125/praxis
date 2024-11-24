import copy
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class PraxisSMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745
    """

    __version__ = "0.2.0"

    def __init__(self, config: AutoConfig, experts: list[nn.Module]):
        super().__init__()
        if not experts:
            raise ValueError(
                "The experts list must contain at least two expert modules."
            )

        self.num_experts = len(experts)
        self.num_dims = config.num_dims
        self.experts = nn.ModuleList(experts)

        # Router network: simple linear -> softmax
        self.router = nn.Sequential(
            nn.Linear(self.num_dims, self.num_experts),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(config.dropout)

        # Collect all parameter names from the first expert
        self.parameter_names = self._collect_parameter_names(self.experts[0])

    def forward(self, inputs):
        # Get routing probabilities
        routing_probs = self.router(inputs.mean(dim=1))  # [batch_size, num_experts]

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        # Use the first expert as the base module structure
        base_module = self.experts[0]
        # Apply the merged parameters using functional_call
        outputs = torch.func.functional_call(base_module, merged_state_dict, inputs)

        return outputs

    def _collect_parameter_names(self, module, prefix=""):
        """
        Recursively collect all parameter names from a module.
        """
        parameter_names = []
        for name, submodule in module.named_children():
            parameter_names.extend(
                self._collect_parameter_names(submodule, prefix + name + ".")
            )
        for name, _ in module.named_parameters(recurse=False):
            parameter_names.append(prefix + name)
        return parameter_names

    def _merge_expert_parameters(self, routing_probs):
        """
        Merge expert parameters based on routing probabilities.
        """
        # Initialize a dictionary to hold the merged parameters
        merged_state_dict = {}

        # Compute the mean routing probability across the batch for each expert
        expert_weights = routing_probs.mean(dim=0)  # [num_experts]

        # Iterate over all parameter names
        for param_name in self.parameter_names:
            # Initialize the merged parameter with zeros
            merged_param = None

            for expert_idx, expert in enumerate(self.experts):
                # Get the parameter from the expert
                param = self._get_module_parameter(expert, param_name)

                if param is None:
                    raise ValueError(
                        f"Parameter '{param_name}' not found in expert {expert_idx}."
                    )

                # Apply dropout to the parameter
                param_dropped = self.dropout(param)

                # Weight the parameter by the expert's routing probability
                weighted_param = param_dropped * expert_weights[expert_idx]

                if merged_param is None:
                    merged_param = weighted_param
                else:
                    merged_param = merged_param + weighted_param

            merged_state_dict[param_name] = merged_param

        return merged_state_dict

    def _get_module_parameter(self, module, param_name):
        """
        Retrieve a parameter from a module using a fully qualified parameter name.
        """
        parts = param_name.split(".")
        submodule = module
        for part in parts[:-1]:
            if hasattr(submodule, part):
                submodule = getattr(submodule, part)
            else:
                return None
        return getattr(submodule, parts[-1], None)


if __name__ == "__main__":
    # Simple smoke tests for PraxisSMEAR

    # Define a simple MLP expert
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, activation="relu"):
            super(SimpleMLP, self).__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.activation = F.relu
            self.linear2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    # Configuration for SMEAR
    class Config:
        def __init__(self, num_dims, activation, dropout):
            self.num_dims = num_dims
            self.activation = activation
            self.dropout = dropout

    # Instantiate duplicate MLP experts
    input_dim = 16
    hidden_dim = 32
    output_dim = 16
    num_experts = 3

    experts = [
        SimpleMLP(input_dim, hidden_dim, output_dim, activation="relu")
        for _ in range(num_experts)
    ]

    # Create a configuration
    config = Config(num_dims=input_dim, activation="relu", dropout=0.1)

    # Instantiate the SMEAR module with the experts
    smear = PraxisSMEAR(config, experts)

    # Create dummy input
    batch_size = 4
    seq_length = 10
    dummy_input = torch.randn(batch_size, seq_length, input_dim)

    # Forward pass through SMEAR
    output = smear(dummy_input)
    print(
        "Output shape:", output.shape
    )  # Expected: [batch_size, seq_length, output_dim]

    # Verify that the output is differentiable
    output.sum().backward()
    print("Backward pass successful.")

    @dataclass
    class Config:
        def __init__(self, num_dims, dropout):
            self.num_dims = num_dims
            self.dropout = dropout

    class AttentionKey(nn.Module):
        """Simple wrapper for attention key projection"""

        def __init__(self, hidden_size, num_heads, head_dim, multiplier=1):
            super().__init__()
            self.key = nn.Linear(
                hidden_size,
                num_heads * head_dim * multiplier,
                bias=False,
            )

        def forward(self, x):
            return self.key(x)

    # Test configuration
    hidden_size = 768
    num_heads = 12
    head_dim = 64
    num_experts = 3

    # Create expert modules
    experts = [
        AttentionKey(hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim)
        for _ in range(num_experts)
    ]

    # Create config
    config = Config(num_dims=hidden_size, dropout=0.1)

    smear = PraxisSMEAR(config, experts)

    # Create test input
    batch_size = 4
    seq_length = 32
    inputs = torch.randn(batch_size, seq_length, hidden_size)

    # Forward pass
    outputs = smear(inputs)

    # Print shapes for verification
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    # Expected output shape
    expected_output_size = (batch_size, seq_length, num_heads * head_dim)
    assert (
        outputs.shape == expected_output_size
    ), f"Expected shape {expected_output_size}, got {outputs.shape}"

    print("\nTest successful! Output shapes match expected dimensions.")

    # Verify routing probabilities
    with torch.no_grad():
        routing_probs = smear.router(inputs.mean(dim=1))
        prob_sums = routing_probs.sum(dim=-1)
        print(f"\nRouting probability sums (should be close to 1.0):")
        print(prob_sums)

    # Test backward pass
    outputs.sum().backward()
    print("\nBackward pass successful.")

    # Print expert weights
    with torch.no_grad():
        routing_probs = smear.router(inputs.mean(dim=1))
        expert_weights = routing_probs.mean(dim=0)
        print("\nExpert weights:")
        for i, weight in enumerate(expert_weights):
            print(f"Expert {i}: {weight.item():.3f}")
