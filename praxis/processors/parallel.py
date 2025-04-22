from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torch.nn.parallel as parallel
from torch import nn

from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError
from praxis.processors.checkpoint import create_forward, should_checkpoint


class ParallelProcessor(nn.Module):
    def __init__(self, config: "AutoConfig", mode="mean"):
        super().__init__()
        self.mode = mode
        if self.mode == "weighted":
            self.contributions = nn.Parameter(torch.ones(config.depth))

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
        training: bool = False,
    ):
        hidden_states = inputs
        new_states = []
        aux_losses = []

        # Create wrapper functions for each expert
        def create_expert_forward(idx):
            def expert_forward(input_tensor):
                try:
                    expert = experts[idx]
                    layer_state = (
                        current_state[idx] if current_state is not None else None
                    )
                    return create_forward(
                        expert,
                        stack,
                        input_tensor,
                        attention_mask,
                        past_key_values,
                        layer_state,
                        idx,
                        block_ids,
                        should_checkpoint(training, idx, stack.checkpoint_every),
                    )
                except (P2PDaemonError, P2PHandlerError) as e:
                    if stack.debug:
                        print(e)
                    stack.manager.handle_failure(experts[idx])
                    return None

            return expert_forward

        # Create function list and replicate inputs
        expert_forwards = [create_expert_forward(i) for i in range(stack.depth)]
        inputs_list = [hidden_states] * stack.depth

        # Execute all expert forwards in parallel
        results = parallel.parallel_apply(expert_forwards, inputs_list)

        # Process results
        all_hidden_updates = []
        for i, result in enumerate(results):
            if result is not None:
                hidden_update, pkv, layer_state, aux_loss = result

                new_states.append(layer_state)
                aux_losses.append(aux_loss)

                # Apply post_layer transformation if defined
                if hasattr(stack, "post_layer"):
                    hidden_update = stack.post_layer(hidden_update, i)

                all_hidden_updates.append(hidden_update)

        # Mean pooling of hidden updates to combine expert outputs
        hidden_states = self._combine_outputs(all_hidden_updates, self.mode)

        # Apply post-decoding if defined
        if hasattr(stack, "post_decoding"):
            hidden_states = stack.post_decoding(hidden_states)

        return hidden_states, past_key_values, current_state, sum(aux_losses)

    def _combine_outputs(self, hidden_updates, mode="mean"):
        stacked_updates = torch.stack(hidden_updates)
        if mode == "mean":
            return torch.mean(stacked_updates, dim=0)
        elif mode == "weighted":
            return self._compute_weighted_sum(stacked_updates)

    def _compute_weighted_sum(self, stacked_updates):
        weights = F.softmax(self.contributions, dim=0)
        weighted_updates = stacked_updates * weights.view(-1, 1, 1, 1)
        return torch.sum(weighted_updates, dim=0)
