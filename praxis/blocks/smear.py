import copy
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


@dataclass
class SMEARConfig:
    """Configuration class for SMEAR module"""

    num_dims: int
    num_experts: int
    dropout: float = 0.1


class PraxisSMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745
    """

    __version__ = "0.2.0"

    def __init__(self, config: SMEARConfig, experts: list[nn.Module], *args, **kwargs):
        super().__init__()

        self.num_experts = config.num_experts
        self.num_dims = config.num_dims
        self.experts = nn.ModuleList(experts)  # Properly register experts as submodules

        # Router network: simple linear -> softmax
        self.router = nn.Sequential(
            nn.Linear(self.num_dims, self.num_experts),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, *args, **kwargs):
        # Get routing probabilities
        routing_probs = self.router(inputs.mean(dim=1))  # [batch_size, num_experts]

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        # Use the first expert as the base module structure
        base_module = self.experts[0]
        # Apply the merged parameters using functional_call
        outputs = torch.func.functional_call(
            base_module, merged_state_dict, (inputs, *args), kwargs
        )

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
        self.parameter_names = self._collect_parameter_names(self.experts[0])
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
    print("Running SMEAR tests...")

    # Test 1: Simple MLP
    print("\n1. Testing MLP experts...")

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.activation = F.relu
            self.linear2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            return x

    # MLP test configuration
    input_dim = 16
    hidden_dim = 32
    output_dim = 16
    num_experts = 3

    experts = [SimpleMLP(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]

    config = SMEARConfig(num_dims=input_dim, num_experts=num_experts)
    smear = PraxisSMEAR(config, experts)

    batch_size = 4
    seq_length = 10
    dummy_input = torch.randn(batch_size, seq_length, input_dim)

    output = smear(dummy_input)
    print(f"MLP Output shape: {output.shape}")
    output.sum().backward()
    print("MLP test passed!")

    # Test 2: Attention Key
    print("\n2. Testing Attention Key experts...")

    class AttentionKey(nn.Module):
        def __init__(self, hidden_size, num_heads, head_dim):
            super().__init__()
            self.key = nn.Linear(
                hidden_size,
                num_heads * head_dim,
                bias=False,
            )

        def forward(self, x):
            return self.key(x)

    hidden_size = 768
    num_heads = 12
    head_dim = 64
    num_experts = 3

    attention_experts = [
        AttentionKey(hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim)
        for _ in range(num_experts)
    ]

    config = SMEARConfig(num_dims=hidden_size, num_experts=num_experts)
    attention_smear = PraxisSMEAR(config, attention_experts)

    batch_size = 4
    seq_length = 32
    inputs = torch.randn(batch_size, seq_length, hidden_size)
    outputs = attention_smear(inputs)

    print(f"Attention Input shape: {inputs.shape}")
    print(f"Attention Output shape: {outputs.shape}")
    outputs.sum().backward()
    print("Attention test passed!")

    # Test 3: LSTM
    print("\n3. Testing LSTM experts...")

    # LSTM test configuration
    input_size = 64
    hidden_size = 128
    num_experts = 3

    lstm_experts = [
        nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for _ in range(num_experts)
    ]

    config = SMEARConfig(num_dims=input_size, num_experts=num_experts)
    lstm_smear = PraxisSMEAR(config, lstm_experts)

    # Test LSTM with sequence data
    batch_size = 4
    seq_length = 20
    lstm_input = torch.randn(batch_size, seq_length, input_size)

    # Test both with and without initial hidden state
    lstm_output1, (h_n1, c_n1) = lstm_smear(lstm_input)

    initial_hidden = (
        torch.zeros(1, batch_size, hidden_size),
        torch.zeros(1, batch_size, hidden_size),
    )
    lstm_output2, (h_n2, c_n2) = lstm_smear(lstm_input, initial_hidden)

    print(f"LSTM Input shape: {lstm_input.shape}")
    print(f"LSTM Output shape: {lstm_output1.shape}")
    print(f"LSTM Hidden state shape: {h_n1.shape}")
    print(f"LSTM Cell state shape: {c_n1.shape}")

    # Test backprop
    lstm_output1.sum().backward()
    print("LSTM test passed!")

    print("\nAll tests completed successfully!")
