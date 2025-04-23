import torch
import torch.nn.functional as F
import torch.nn.parallel as parallel
from torch import nn

from praxis.processors.checkpoint import create_forward, should_checkpoint


class ParallelProcessor(nn.Module):
    def __init__(self, config: "AutoConfig", mode="mean"):
        super().__init__()
        self.mode = mode
        if self.mode == "weighted":
            self.contributions = nn.Parameter(
                torch.ones(config.depth, config.hidden_size)
            )

    def forward(
        self,
        experts,
        stack,
        inputs,
        attention_mask,
        past_key_values,
        block_ids,
        current_state,
        original_order,
    ):
        hidden_states = inputs
        new_states = []
        aux_losses = []

        # Create wrapper functions for each expert
        def create_expert_forward(idx):
            def expert_forward(input_tensor):
                expert = experts[idx]
                layer_state = current_state[idx] if current_state is not None else None
                return create_forward(
                    expert,
                    stack,
                    input_tensor,
                    attention_mask,
                    past_key_values,
                    layer_state,
                    idx,
                    block_ids,
                    should_checkpoint(self.training, idx, stack.checkpoint_every),
                )

            return expert_forward

        # Create function list and replicate inputs
        expert_forwards = [create_expert_forward(i) for i in range(stack.depth)]
        inputs_list = [hidden_states] * stack.depth

        # Execute all expert forwards in parallel
        results = parallel.parallel_apply(expert_forwards, inputs_list)

        # Process results
        all_hidden_updates = []
        valid_expert_indices = []

        for i, result in enumerate(results):
            if result is not None:
                hidden_update, pkv, layer_state, aux_loss = result

                new_states.append(layer_state)
                aux_losses.append(aux_loss)

                # Apply post_layer transformation if defined
                if hasattr(stack, "post_layer"):
                    hidden_update = stack.post_layer(hidden_update, i)

                all_hidden_updates.append(hidden_update)
                valid_expert_indices.append(i)

        # Mean pooling of hidden updates to combine expert outputs
        hidden_states = self._combine_outputs(
            all_hidden_updates, valid_expert_indices, self.mode
        )

        # Apply post-decoding if defined
        if hasattr(stack, "post_decoding"):
            hidden_states = stack.post_decoding(hidden_states)

        return hidden_states, past_key_values, current_state, sum(aux_losses)

    def _combine_outputs(self, hidden_updates, valid_indices, mode="mean"):
        stacked_updates = torch.stack(hidden_updates)
        if mode == "mean":
            return torch.mean(stacked_updates, dim=0)
        elif mode == "variance":
            return self._compute_variance_weighted_sum(stacked_updates)
        elif mode == "weighted":
            return self._compute_feature_weighted_sum(stacked_updates, valid_indices)

    def _compute_variance_weighted_sum(self, stacked_updates):
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

    def _compute_feature_weighted_sum(self, stacked_updates, valid_indices):
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
