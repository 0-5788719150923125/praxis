"""Goodness-based Mono-Forward Decoder implementing the original algorithm."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from praxis.containers import LossContainer
from praxis.decoders.sequential import SequentialDecoder
from praxis.modules.layer_with_optimizer import LayerWithOptimizer

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class MonoForwardDecoder(SequentialDecoder):
    """
    Simplified Mono-Forward decoder using LayerWithOptimizer.

    Each layer:
    1. Receives detached inputs (no gradients from previous layers)
    2. Computes goodness scores G = activations @ M.T
    3. Updates weights immediately using layer-local loss
    4. Returns detached output for O(1) memory complexity
    """

    def __init__(self, config: ConfigType) -> None:
        # Initialize parent to get the base layers
        super().__init__(config)

        # Configuration
        self.config = config

        # Replace each layer with LayerWithOptimizer wrapper
        wrapped_layers = nn.ModuleList()

        for i, layer in enumerate(self.locals):
            # Create goodness projection for this layer using nn.Linear
            projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            # Initialize similar to the paper (optional - could use default Kaiming)
            nn.init.normal_(projection.weight, std=(2.0 / config.hidden_size) ** 0.5)

            # Get optimizer configuration from config or use defaults
            optimizer_class, optimizer_kwargs = self._get_optimizer_config()

            # Wrap with LayerWithOptimizer
            wrapped = LayerWithOptimizer(
                layer=layer,
                optimizer_class=optimizer_class,
                optimizer_kwargs=optimizer_kwargs,
                loss_fn=F.cross_entropy,  # Standard cross-entropy
                projection_layer=projection,
            )

            wrapped_layers.append(wrapped)

        # Replace the original layers
        self.locals = wrapped_layers

    def _get_optimizer_config(self) -> Tuple[Type[Optimizer], Dict[str, Any]]:
        """
        Get optimizer configuration from the model config.

        Expects config to have:
        - optimizer_config: dict with all optimizer parameters
        - optimizer_wrappers: dict with flags for trac, ortho, lookahead, schedule_free
        """
        # Check if optimizer config is provided in the model config
        if hasattr(self.config, "optimizer_config"):
            from pytorch_optimizer import create_optimizer

            from praxis.optimizers import get_optimizer

            # Get the optimizer config dict
            optimizer_config = self.config.optimizer_config

            # Create a dummy module to get the optimizer class
            class DummyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.param = nn.Parameter(torch.zeros(1))

            dummy = DummyModule()

            # Check if we have wrapper flags
            wrappers = getattr(self.config, "optimizer_wrappers", {})

            # If we have wrappers, use get_optimizer to handle them
            if any(wrappers.values()):
                optimizer = get_optimizer(
                    dummy,
                    trac=wrappers.get("trac", False),
                    ortho=wrappers.get("ortho", False),
                    lookahead=wrappers.get("lookahead", False),
                    schedule_free=wrappers.get("schedule_free", False),
                    **optimizer_config,
                )
            else:
                # Otherwise just create the base optimizer
                optimizer = create_optimizer(dummy, **optimizer_config)

            optimizer_class = type(optimizer)

            # For wrapped optimizers, we want the base class for LayerWithOptimizer
            # since it will create its own instance
            if optimizer_class.__name__ in [
                "TRAC",
                "OrthoGrad",
                "Lookahead",
                "ScheduleFreeWrapper",
            ]:
                # Get the base optimizer class
                base_optimizer = (
                    optimizer.base_optimizer
                    if hasattr(optimizer, "base_optimizer")
                    else optimizer.optimizer
                )
                optimizer_class = type(base_optimizer)

            # Remove optimizer_name from kwargs as it's not a valid parameter
            optimizer_kwargs = {k: v for k, v in optimizer_config.items() if k != "optimizer_name"}
            return optimizer_class, optimizer_kwargs
        else:
            # Fallback to default
            print("MonoForwardDecoder: No optimizer config provided, using Adam")
            return torch.optim.Adam, {"lr": 1e-3}

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[Union[List[Any], Dict[str, Any]]] = None,
        current_state: Optional[List[Any]] = None,
        block_ids: Optional[Tensor] = None,
        losses: LossContainer = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[
        Tensor,
        Optional[Union[List[Any], Dict[str, Any]]],
        Optional[List[Any]],
        LossContainer,
    ]:
        """
        Forward pass using LayerWithOptimizer for true O(1) memory.
        
        Always returns accumulated goodness scores for consistency between
        training and inference.
        """
        _, seq_len, _ = hidden_states.shape

        # Always accumulate goodness scores
        total_goodness = torch.zeros(
            hidden_states.size(0),
            seq_len,
            self.config.vocab_size,
            device=hidden_states.device,
        )

        current_route: List[int] = []
        controller_state = None

        # Process through layers
        for layer_idx in range(self.num_experts):
            # Controller routing
            hidden_states, controller_state, controller_loss, next_expert_idx = (
                self.controller.get_next_expert(
                    hidden_states,
                    controller_state,
                    self.locals,
                    self.locals,
                    current_route,
                    layer_idx,
                )
            )

            losses.add_loss_container(controller_loss)

            if next_expert_idx is None:
                break

            # Get the LayerWithOptimizer wrapped layer
            wrapped_layer = self.locals[next_expert_idx]

            # Forward through layer - it handles everything!
            output, layer_loss = wrapped_layer(
                hidden_states,
                labels=labels,
                attention_mask=attention_mask,
                current_state=current_state[layer_idx] if current_state else None,
                past_key_values=past_key_values,
                current_depth=layer_idx,
                block_ids=block_ids,
            )

            # Extract hidden states from output
            if isinstance(output, tuple):
                hidden_states = output[0]
                if len(output) > 1:
                    past_key_values = output[1]
            else:
                hidden_states = output

            # Record loss value only for monitoring (no gradients)
            if layer_loss is not None:
                # Use .item() to ensure we only store the scalar value
                losses.add_loss(
                    f"goodness_layer_{layer_idx}",
                    (
                        layer_loss.item()
                        if hasattr(layer_loss, "item")
                        else float(layer_loss)
                    ),
                )

            # Apply compression and post-processing
            hidden_states = self.compressor.reduce_sequence(hidden_states)
            hidden_states = self.post_layer(hidden_states, layer_idx)

            # Always accumulate goodness for this layer
            # Compute goodness for this layer
            goodness = wrapped_layer.projection(hidden_states)

            # Accumulate all layers' goodness
            min_len = min(goodness.size(1), total_goodness.size(1))
            if self.training:
                # During training, accumulate with gradients for the final loss
                total_goodness[:, :min_len] = total_goodness[:, :min_len] + goodness[:, :min_len]
            else:
                # During inference, no gradients needed
                with torch.no_grad():
                    total_goodness[:, :min_len] += goodness[:, :min_len]

            # Update route
            current_route = self.controller.update_route(
                hidden_states, current_route, layer_idx, next_expert_idx
            )

        # Final processing
        hidden_states = self.compressor.expand_sequence(hidden_states, seq_len)
        hidden_states = self.post_decoding(hidden_states)
        self.controller.post_forward(hidden_states, current_route)
        hidden_states = self.order(hidden_states)

        # Add marker to signal layer-wise training is complete
        if self.training and labels is not None:
            losses.add_loss("_layer_wise_complete", torch.tensor(0.0, requires_grad=True))

        # Always return accumulated goodness scores
        return total_goodness, past_key_values, current_state, losses

    def parameters(self, recurse: bool = True):
        """Override to exclude layer parameters (they have their own optimizers)."""
        # Only return non-layer parameters (e.g., controller parameters)
        for name, param in self.named_parameters(recurse=recurse):
            # Skip parameters from wrapped layers
            if not name.startswith("locals."):
                yield param
