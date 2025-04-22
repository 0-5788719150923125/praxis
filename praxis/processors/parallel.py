import torch

from praxis.orchestration.hivemind import P2PDaemonError, P2PHandlerError
from praxis.processors.checkpoint import create_forward, should_checkpoint


def parallel_processor(
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

    # Process all experts in parallel
    all_hidden_updates = []

    for i in range(stack.depth):
        try:
            expert = experts[i]
            layer_state = current_state[i] if current_state is not None else None
            hidden_update, past_key_values, layer_state, aux_loss = create_forward(
                expert,
                stack,
                hidden_states,  # All experts get the same input
                attention_mask,
                past_key_values,
                layer_state,
                i,
                block_ids,
                should_checkpoint(training, i, stack.checkpoint_every),
            )

            new_states.append(layer_state)
            aux_losses.append(aux_loss)

            # Apply post_layer transformation if defined
            if hasattr(stack, "post_layer"):
                hidden_update = stack.post_layer(hidden_update, i)

            all_hidden_updates.append(hidden_update)

        except (P2PDaemonError, P2PHandlerError) as e:
            # Prune dead peers
            if stack.debug:
                print(e)
            stack.manager.handle_failure(expert)
            continue

    # Mean pooling of hidden updates to combine expert outputs
    hidden_states = torch.mean(torch.stack(all_hidden_updates), dim=0)

    # Apply post-decoding if defined
    if hasattr(stack, "post_decoding"):
        hidden_states = stack.post_decoding(hidden_states)

    return hidden_states, past_key_values, current_state, sum(aux_losses)
