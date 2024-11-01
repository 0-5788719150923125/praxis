import copy
from collections import defaultdict

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
                    merged_param += weighted_param

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
