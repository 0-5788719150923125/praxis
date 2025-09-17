import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


@dataclass
class SMEARConfig:
    """Configuration class for SMEAR module"""

    hidden_size: int
    num_experts: int
    dropout: float = 0.1


class SMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745

    SMEAR dynamically merges expert parameters based on routing probabilities,
    rather than routing inputs to multiple experts. This enables more efficient
    parameter sharing and adaptation to input patterns.
    """

    __version__ = "0.3.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize the SMEAR module.

        Args:
            config: Configuration object
            layout: Layout type (not used by SMEAR, kept for interface compatibility)
            *args: Additional arguments
            **kwargs: Additional keyword arguments including 'experts' list
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.dropout_rate = getattr(
            config, "dropout", 0.1
        )  # Probability of dropping entire experts

        # Get experts from kwargs - required for SMEAR
        self.experts = kwargs.get("experts", None)
        if self.experts is None:
            raise ValueError(
                "SMEAR router requires 'experts' to be provided during initialization"
            )
        self.experts = nn.ModuleList(self.experts)

        self.parameter_names: List[str] = []

        # Router network with layer normalization as per paper
        # Use actual number of experts, not config.num_experts
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, len(self.experts))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_experts={len(self.experts)})"

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], float],  # Direct mode
        Tuple[
            torch.Tensor,
            Optional[Union[torch.Tensor, List, Dict]],
            Optional[torch.Tensor],
            float,
        ],  # Router mode
    ]:
        """
        Forward pass with SMEAR routing.

        Supports two modes:
        1. Direct mode: Used by RecurrentBlock with (inputs, current_state)
        2. Router mode: Used as a router in LocalLayer with full signature

        Returns:
            Appropriate tuple based on the mode
        """
        # Determine which mode we're in based on arguments
        if self._is_router_mode(args, kwargs):
            # Extract router mode arguments
            router_args = self._parse_router_args(args, kwargs)
            return self._router_forward(*router_args)
        else:
            # Extract direct mode arguments
            inputs, current_state = self._parse_direct_args(args, kwargs)
            return self._direct_forward(inputs, current_state)

    def _router_forward(
        self,
        layer: nn.Module,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Union[torch.Tensor, List, Dict]],
        current_state: Optional[torch.Tensor],
        current_depth: int,
        block_ids: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[Union[torch.Tensor, List, Dict]],
        Optional[torch.Tensor],
        float,
    ]:
        """Router mode forward pass."""
        # Get routing probabilities with proper normalization
        router_input = inputs.mean(dim=1)  # Average across sequence length
        router_input = self.router_norm(router_input)  # Layer norm on input

        # Get logits with normalized router weights
        # Apply weight normalization without modifying in-place
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Apply expert dropout during training (drop entire experts)
        if self.training and self.dropout_rate > 0:
            # Create dropout mask for experts
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            # Renormalize to ensure probabilities sum to 1
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        # Use the first expert as the base module structure
        base_module = self.experts[0]

        # Create args tuple for functional_call
        forward_args = (
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )

        # Apply the merged parameters using functional_call
        result = torch.func.functional_call(
            base_module, merged_state_dict, forward_args, {}
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 4:
            return result
        elif isinstance(result, tuple) and len(result) == 3:
            # Add zero aux loss if not provided
            return result[0], result[1], result[2], 0.0
        else:
            # Fallback for unexpected return format
            return result, past_key_values, current_state, 0.0

    def _collect_parameter_names(
        self, module: nn.Module, prefix: str = ""
    ) -> List[str]:
        """
        Recursively collect all parameter names from a module.

        Args:
            module: Module to collect parameter names from
            prefix: Prefix for nested parameter names

        Returns:
            List of parameter names with full path
        """
        parameter_names = []
        for name, submodule in module.named_children():
            parameter_names.extend(
                self._collect_parameter_names(submodule, prefix + name + ".")
            )
        for name, _ in module.named_parameters(recurse=False):
            parameter_names.append(prefix + name)
        return parameter_names

    def _merge_expert_parameters(
        self, routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Merge expert parameters based on routing probabilities.

        Args:
            routing_probs: Routing probabilities of shape [batch_size, num_experts]

        Returns:
            Dictionary of merged parameters

        Raises:
            ValueError: If a parameter is not found in an expert
        """
        # Initialize a dictionary to hold the merged parameters
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # Compute the mean routing probability across the batch for each expert
        expert_weights = routing_probs.mean(dim=0)  # [num_experts]

        # Iterate over all parameter names
        self.parameter_names = self._collect_parameter_names(self.experts[0])
        for param_name in self.parameter_names:
            # Initialize the merged parameter with zeros
            merged_param: Optional[torch.Tensor] = None

            for expert_idx, expert in enumerate(self.experts):
                # Get the parameter from the expert
                param = self._get_module_parameter(expert, param_name)

                if param is None:
                    raise ValueError(
                        f"Parameter '{param_name}' not found in expert {expert_idx}."
                    )

                # Ensure param is on the same device as expert_weights before multiplication
                if param.device != expert_weights.device:
                    param = param.to(expert_weights.device)

                # Weight the parameter by the expert's routing probability
                weighted_param = param * expert_weights[expert_idx]

                if merged_param is None:
                    merged_param = weighted_param
                else:
                    merged_param = merged_param + weighted_param

            assert merged_param is not None, "Merged parameter should not be None"
            merged_state_dict[param_name] = merged_param

        return merged_state_dict

    def _get_module_parameter(
        self, module: nn.Module, param_name: str
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a parameter from a module using a fully qualified parameter name.

        Args:
            module: Module to get parameter from
            param_name: Fully qualified parameter name (e.g., "layer1.weight")

        Returns:
            Parameter tensor if found, None otherwise
        """
        parts = param_name.split(".")
        submodule = module
        for part in parts[:-1]:
            if hasattr(submodule, part):
                submodule = getattr(submodule, part)
            else:
                return None
        return getattr(submodule, parts[-1], None)

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        """Check if we're in router mode based on arguments."""
        # Router mode if we have 7 positional args or 'layer' in kwargs
        return len(args) == 7 or "layer" in kwargs

    def _parse_router_args(self, args: tuple, kwargs: dict) -> tuple:
        """Parse arguments for router mode."""
        if len(args) == 7:
            # Positional arguments
            return args
        else:
            # Keyword arguments
            return (
                kwargs["layer"],
                kwargs["inputs"],
                kwargs.get("attention_mask"),
                kwargs.get("past_key_values"),
                kwargs.get("current_state"),
                kwargs.get("current_depth", 0),
                kwargs.get("block_ids"),
            )

    def _parse_direct_args(
        self, args: tuple, kwargs: dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parse arguments for direct mode."""
        if len(args) >= 2:
            # Positional args: (inputs, current_state)
            return args[0], args[1]
        elif len(args) == 1:
            # Only inputs provided
            return args[0], None
        else:
            # Try kwargs
            inputs = kwargs.get("inputs")
            if inputs is None:
                raise ValueError(f"No inputs provided. Args: {args}, Kwargs: {kwargs}")
            return inputs, kwargs.get("current_state")

    def _direct_forward(
        self,
        inputs: torch.Tensor,
        current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Direct mode forward pass for RecurrentBlock usage."""
        # Get routing probabilities with proper normalization
        router_input = inputs.mean(dim=1)  # Average across sequence length
        router_input = self.router_norm(router_input)  # Layer norm on input

        # Get logits with normalized router weights
        # Apply weight normalization without modifying in-place
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Apply expert dropout during training (drop entire experts)
        if self.training and self.dropout_rate > 0:
            # Create dropout mask for experts
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            # Renormalize to ensure probabilities sum to 1
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        # Use the first expert as the base module structure
        base_module = self.experts[0]

        # Apply the merged parameters using functional_call
        result = torch.func.functional_call(
            base_module, merged_state_dict, (inputs, current_state), {}
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            # Add zero aux loss if not provided
            return result[0], result[1], 0.0
        else:
            # Fallback
            return result, current_state, 0.0
