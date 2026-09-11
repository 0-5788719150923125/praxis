import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.utils import (
    generate_alternating_values,
    generate_decay_values,
    generate_u_shape_values,
)

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

MOD_LAYOUT: Dict[str, Callable[[int], List[float]]] = {
    "standard": lambda depth: generate_alternating_values(
        size=depth, interval=1, capacity=0.125
    ),
    # generate_decay_values rises with depth unless reversed.
    "decayed": lambda depth: generate_decay_values(
        depth, reverse=True, center=0.5, lower_bound=0.125
    ),
    "ramped": lambda depth: generate_decay_values(depth, center=0.5, lower_bound=0.125),
    "u": lambda depth: generate_u_shape_values(
        depth,
        decay_point=0.1,
        ramp_point=0.9,
        lower_bound=0.125,
        steepness=2.0,
    ),
    "skip_2": lambda depth: generate_alternating_values(
        size=depth, interval=2, capacity=0.125
    ),
}


class MixtureOfDepths(nn.Linear):
    """At each layer, route only a fraction of tokens through the heavy
    computation; the rest pass through via the residual. Uses expert-choice
    routing (the layer picks its top-k tokens by score) rather than
    token-choice, per the original paper's recommendation.

    The ``layout`` controls how per-layer capacity varies with depth - flat,
    decayed, U-shaped, ramped, or skip-every-N. See
    https://arxiv.org/abs/2404.02258.

    CAUSALITY follows the paper (Sec. 3.5). Expert-choice top-k is non-causal -
    whether a token is among the top-k depends on the tokens after it - so the
    paper trains with it and SAMPLES with a causal rule instead: a token is
    routed iff its own router output clears 0.5 after the sigmoid, which the
    auxiliary BCE loss (``aux_loss``) trains the router to predict. Training here
    uses top-k and inference uses that rule, so every eval forward - validation
    and generation - is causal; training keeps the paper's non-causal selection.
    """

    def __init__(
        self, config: ConfigType, layout: str = "standard", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(in_features=config.hidden_size, out_features=1)
        self.capacities: List[float] = self._build_capacities(config, layout)
        if config.debug:
            print(self.capacities)

    def _build_capacities(self, config: ConfigType, layout: str) -> List[float]:
        """Per-step capacity schedule, indexed by the global flattened depth.

        Subclasses (e.g. ``ArcMixture``) override to key the schedule to a
        different axis, such as the physical layer index.
        """
        if layout not in MOD_LAYOUT:
            raise ValueError(
                f"Unknown mixture-of-depths layout {layout!r}; known: {sorted(MOD_LAYOUT)}"
            )
        return MOD_LAYOUT[layout](config.depth)

    def forward(
        self,
        layer: nn.Module,
        inputs: Tensor,
        attention_mask: Optional[Tensor],
        past_key_values: Optional[Union[Tensor, List, Dict]],
        current_state: Optional[Tensor],
        current_depth: int,
        block_ids: Optional[Tensor],
        positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Union[Tensor, List, Dict]], Optional[Tensor], float]:
        """
        Forward pass with selective token routing based on capacity.

        Args:
            layer: The layer module to process selected tokens
            inputs: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask tensor
            past_key_values: Optional cached key/value tensors for attention
            current_state: Optional current layer state
            current_depth: Current layer depth
            block_ids: Optional block identifiers for tokens

        Returns:
            Tuple containing:
                - Processed outputs tensor
                - Updated key/value cache
                - Updated layer state
                - Combined loss value
        """
        router_loss = 0.0
        capacity = self._capacity_for(current_depth)

        b, s, d = inputs.shape
        k = int(s * capacity)

        # if capacity is 1, then we should process all tokens normally
        if capacity == 1:
            layer_outputs, layer_kv, state_update, aux_loss = layer(
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            )
            return layer_outputs, layer_kv, state_update, aux_loss

        # if k is 0, then no tokens will be selected and we should skip this layer
        if k == 0:
            return inputs, past_key_values, current_state, router_loss

        # emit scalar weights for each token
        router_logits = self._compute_router_logits(
            inputs, current_depth
        )  # -> batch, seq_len, 1

        valid = None
        if self.training:
            #  𝑟𝑙> 𝑃𝛽 (R) - equation 1
            token_weights, token_indices = torch.topk(
                router_logits,
                k,
                dim=1,
                sorted=False,
            )

            # Sort indices by position and get the sorting indices
            token_indices, sort_indices = torch.sort(token_indices, dim=1)

            # Re-order the weights to match the sorted indices
            token_weights = torch.gather(token_weights, dim=1, index=sort_indices)

            # The paper's auxiliary BCE: teaches the router's own output to
            # predict the top-k choice, which is the rule inference routes on.
            router_loss = self.aux_loss(router_logits, token_indices)
        else:
            token_indices, token_weights, valid = self._causal_selection(router_logits)
            if token_indices is None:
                return inputs, past_key_values, current_state, router_loss

        # expand router predictions to match input dimensions
        indices_expanded = token_indices.expand(-1, -1, d)

        # pull top-k tokens from the original inputs
        filtered_inputs = torch.gather(
            input=inputs, dim=1, index=indices_expanded
        )  # -> batch, capacity, 1

        # slice an attention mask that matches the top-k selections
        squeezed_indices = token_indices.squeeze(-1)
        filtered_attention_mask = None
        if attention_mask is not None:
            filtered_attention_mask = torch.gather(
                input=attention_mask,
                dim=1,
                index=squeezed_indices,
            )

        filtered_block_ids = None
        if block_ids is not None:
            filtered_block_ids = torch.gather(
                input=block_ids,
                dim=1,
                index=squeezed_indices,
            )  # [batch, k]

        # pass the selected tokens through a transformer block
        layer_outputs, layer_kv, state_update, aux_loss = layer(
            filtered_inputs,
            filtered_attention_mask,
            past_key_values,
            current_state,
            current_depth,
            filtered_block_ids,
            token_weights,
        )

        if valid is not None:
            # Padding slots carry unselected tokens; they sit after every real
            # selection, so causal attention never lets them reach one, and their
            # outputs are dropped here.
            layer_outputs = torch.where(valid, layer_outputs, filtered_inputs)

        # reintegrate the processed tokens with our residual stream
        outputs = torch.scatter(
            input=inputs,
            dim=1,
            index=indices_expanded,
            src=layer_outputs,
        )

        return outputs, layer_kv, state_update, aux_loss + router_loss

    def _causal_selection(self, router_logits: Tensor):
        """Inference routing: each token routes iff its own logit is positive.

        Returns ``(indices, weights, valid)`` with every row's selected tokens
        first, in order, padded to the batch's largest selection; ``valid`` marks
        the real slots. ``(None, None, None)`` when no token routes.
        """
        selected = router_logits.squeeze(-1) > 0  # [batch, seq]
        counts = selected.sum(dim=1)
        widest = int(counts.max())
        if widest == 0:
            return None, None, None
        seq = selected.shape[1]
        position = torch.arange(seq, device=selected.device)
        order = torch.argsort((~selected).long() * seq + position, dim=1)[:, :widest]
        indices = order.unsqueeze(-1)  # [batch, widest, 1]
        slot = torch.arange(widest, device=selected.device)
        valid = (slot.unsqueeze(0) < counts.unsqueeze(1)).unsqueeze(-1)
        weights = torch.gather(router_logits, dim=1, index=indices)
        return indices, weights, valid

    def _capacity_for(self, current_depth: int) -> float:
        """Token capacity for this step.

        Base MoD keys capacity to the global flattened depth index. Subclasses
        (e.g. ``ArcMixture``) may instead key it to the physical layer index so
        a layer's sparsity is fixed across every recurrent pass.
        """
        return self.capacities[current_depth]

    def _compute_router_logits(self, inputs: Tensor, current_depth: int) -> Tensor:
        """Per-token routing logits, shape [batch, seq_len, 1].

        Subclasses (e.g. ``ArcMixture``) override this to inject
        depth-conditioned terms before the top-k selection.
        """
        return F.linear(inputs, self.weight, self.bias)

    def aux_loss(self, router_logits: Tensor, selected_indices: Tensor) -> Tensor:
        """
        Compute auxiliary loss to center sigmoid outputs around 0.5.

        Args:
            router_logits: Router output logits
            selected_indices: Indices of selected tokens

        Returns:
            Binary cross-entropy loss
        """
        router_targets = torch.zeros_like(router_logits)
        router_targets.scatter_(1, selected_indices, 1.0)
        # page 7: the aux loss centers the sigmoid of the router's outputs around 0.5
        return F.binary_cross_entropy_with_logits(
            router_logits.view(-1), router_targets.view(-1)
        )
