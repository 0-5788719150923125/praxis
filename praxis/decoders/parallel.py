from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
import torch.nn.parallel as parallel
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig

from praxis.decoders.checkpoint import create_forward, should_checkpoint
from praxis.stacks import PraxisStack


ConfigType = TypeVar("ConfigType", bound=PretrainedConfig)


class ParallelDecoder(nn.Module):
    def __init__(self, config: ConfigType, mode: Literal["mean", "variance", "weighted"] = "mean") -> None:
        super().__init__()
        self.stack = PraxisStack(config)
        self.mode = mode
        if self.mode == "weighted":
            self.contributions = nn.Parameter(
                torch.ones(config.depth, config.hidden_size)
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
        current_state: Optional[List[Any]] = None,
        block_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Union[List[Any], Dict[str, Any]]], Optional[List[Any]], Tensor]:
        """
        Forward pass through the parallel decoder.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/values for faster inference
            current_state: Optional current layer states
            block_ids: Optional block identification tensor
            
        Returns:
            Tuple containing:
                - Output hidden states
                - Updated past key values
                - Updated layer states
                - Combined auxiliary loss
        """
        sequential_experts: List[nn.Module] = list(self.stack.locals) + list(self.stack.remotes)
        ordered_experts: List[nn.Module] = self.stack.controller.sort_experts(sequential_experts.copy())
        new_states: List[Any] = []
        aux_losses: List[Tensor] = []

        # Create wrapper functions for each expert
        def create_expert_forward(idx: int) -> callable:
            def expert_forward(input_tensor: Tensor) -> Optional[Tuple[Tensor, Any, Any, Tensor]]:
                expert = ordered_experts[idx]
                layer_state = current_state[idx] if current_state is not None else None
                return create_forward(
                    expert,
                    self.stack,
                    input_tensor,
                    attention_mask,
                    past_key_values,
                    layer_state,
                    idx,
                    block_ids,
                    should_checkpoint(self.training, idx, self.stack.checkpoint_every),
                )

            return expert_forward

        # Create function list and replicate inputs
        expert_forwards = [create_expert_forward(i) for i in range(self.stack.depth)]
        inputs_list = [hidden_states] * self.stack.depth

        # Execute all expert forwards in parallel
        results = parallel.parallel_apply(expert_forwards, inputs_list)

        # Process results
        all_hidden_updates: List[Tensor] = []
        valid_expert_indices: List[int] = []

        for i, result in enumerate(results):
            if result is not None:
                hidden_update, pkv, layer_state, aux_loss = result

                new_states.append(layer_state)
                aux_losses.append(aux_loss)

                hidden_update = self.stack.post_layer(hidden_update, i)

                all_hidden_updates.append(hidden_update)
                valid_expert_indices.append(i)

        # Mean pooling of hidden updates to combine expert outputs
        hidden_states = self._combine_outputs(
            all_hidden_updates, valid_expert_indices, self.mode
        )

        # Apply post-decoding if defined
        hidden_states = self.stack.post_decoding(hidden_states)

        return hidden_states, past_key_values, current_state, sum(aux_losses)

    def _combine_outputs(
        self, 
        hidden_updates: List[Tensor], 
        valid_indices: List[int], 
        mode: Literal["mean", "variance", "weighted"] = "mean"
    ) -> Tensor:
        """
        Combine outputs from multiple experts using the specified combination mode.
        
        Args:
            hidden_updates: List of tensor outputs from each expert
            valid_indices: List of indices of valid experts
            mode: Combination mode ("mean", "variance", or "weighted")
            
        Returns:
            Combined tensor output
        """
        stacked_updates = torch.stack(hidden_updates)
        if mode == "mean":
            return torch.mean(stacked_updates, dim=0)
        elif mode == "variance":
            return self._compute_variance_weighted_sum(stacked_updates)
        elif mode == "weighted":
            return self._compute_feature_weighted_sum(stacked_updates, valid_indices)

    def _compute_variance_weighted_sum(self, stacked_updates: Tensor) -> Tensor:
        """
        Compute variance-weighted sum of expert outputs.
        
        This method weights expert outputs based on their variance, giving more
        weight to experts that produce more unique outputs for each feature.
        
        Args:
            stacked_updates: Stacked hidden state updates from each expert
            
        Returns:
            Variance-weighted sum of expert outputs
        """
        # Calculate the mean across experts
        mean_output = torch.mean(stacked_updates, dim=0)

        # Calculate squared difference from mean for each expert
        squared_diff = (stacked_updates - mean_output.unsqueeze(0)) ** 2

        # Calculate variance with small epsilon for numerical stability
        feature_variance = squared_diff.mean(dim=[1, 2]) + 1e-8

        # Apply log transform with a controllable temperature parameter
        # Higher temp = more aggressive weighting of high-variance features
        temperature = 1.0  # Can be made a learnable parameter if desired
        log_variance = torch.log(feature_variance) * temperature

        # Apply sigmoid
        weights = torch.sigmoid(log_variance)

        # Reshape for broadcasting
        weights = weights.view(weights.size(0), 1, 1, weights.size(1))

        # Apply weights to expert outputs
        weighted_updates = stacked_updates * weights

        # Sum the weighted updates
        return torch.sum(weighted_updates, dim=0)

    def _compute_feature_weighted_sum(self, stacked_updates: Tensor, valid_indices: List[int]) -> Tensor:
        """
        Compute feature-weighted sum of expert outputs.
        
        This method uses learned weights for each feature and expert to compute
        a weighted combination of expert outputs.
        
        Args:
            stacked_updates: Stacked hidden state updates from each expert
            valid_indices: List of indices of valid experts
            
        Returns:
            Feature-weighted sum of expert outputs
        """
        # Get weights for valid experts
        valid_weights = self.contributions[valid_indices]

        # Apply softmax along the experts dimension (0) for each feature independently
        weights = F.softmax(valid_weights, dim=0)  # Shape: [num_experts, hidden_size]

        # Reshape for broadcasting - we need to keep the expert dimension separate
        weights = weights.view(
            weights.size(0), 1, 1, weights.size(1)
        )  # [num_experts, 1, 1, hidden_size]

        # Apply weights to each feature dimension
        weighted_updates = stacked_updates * weights

        # Sum across the experts dimension
        return torch.sum(weighted_updates, dim=0)

    # def _compute_variance_weighted_sum(self, stacked_updates):
    #     # Calculate the mean across experts
    #     mean_output = torch.mean(stacked_updates, dim=0)

    #     # Calculate the variance from mean for each expert
    #     # Squared difference from mean for each expert's output
    #     squared_diff = (stacked_updates - mean_output.unsqueeze(0)) ** 2

    #     # Calculate variance for each feature across all expert outputs
    #     # (across the batch and sequence dimensions)
    #     feature_variance = squared_diff.mean(
    #         dim=[1, 2]
    #     )  # Shape: [num_experts, hidden_size]

    #     # Higher variance means the expert is more "specialized" or "unique"
    #     # We want to reward this uniqueness, so we use variance as weights
    #     # Apply softmax to get normalized weights that sum to 1
    #     weights = F.softmax(feature_variance, dim=0)

    #     # Reshape for broadcasting
    #     weights = weights.view(weights.size(0), 1, 1, weights.size(1))

    #     # Apply these variance-based weights
    #     weighted_updates = stacked_updates * weights

    #     # Sum across experts dimension
    #     return torch.sum(weighted_updates, dim=0)

    # def _compute_feature_weighted_sum(self, stacked_updates, valid_indices):
    #     # Get weights for valid experts
    #     valid_weights = self.contributions[valid_indices]

    #     # Apply sigmoid to each weight independently
    #     weights = torch.sigmoid(valid_weights)  # Shape: [num_experts, hidden_size]

    #     # Reshape for broadcasting
    #     weights = weights.view(weights.size(0), 1, 1, weights.size(1))

    #     # Apply weights to each feature dimension
    #     weighted_updates = stacked_updates * weights

    #     # Sum across the experts dimension
    #     combined = torch.sum(weighted_updates, dim=0)

    #     # Optional: Normalize by sum of weights to maintain consistent magnitude
    #     # This prevents the output from growing too large when many experts contribute
    #     weight_sum = weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    #     return combined / weight_sum
