"""Goodness-based Mono-Forward Decoder implementing the original algorithm."""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_optimizer import Lion

from praxis.containers import LossContainer
from praxis.decoders.sequential import SequentialDecoder
from praxis.modules.layer_with_optimizer import LayerWithOptimizer

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class GoodnessProjection(nn.Module):
    """Projection matrix for computing goodness scores."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        # M matrix: [num_classes, hidden_size]
        self.M = nn.Parameter(
            torch.randn(num_classes, hidden_size) * (2.0 / hidden_size) ** 0.5
        )

    def forward(self, activations: Tensor) -> Tensor:
        """
        Compute goodness scores.

        Args:
            activations: [batch, seq_len, hidden_size]

        Returns:
            goodness: [batch, seq_len, num_classes]
        """
        return activations @ self.M.T


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
        self.learning_rate = getattr(config, "mono_forward_lr", 1e-3)
        
        
        # Replace each layer with LayerWithOptimizer wrapper
        wrapped_layers = nn.ModuleList()
        
        for i, layer in enumerate(self.locals):
            # Create goodness projection for this layer
            projection = GoodnessProjection(config.hidden_size, config.vocab_size)
            
            # Wrap with LayerWithOptimizer using Lion
            wrapped = LayerWithOptimizer(
                layer=layer,
                optimizer_class=Lion,
                optimizer_kwargs={
                    "lr": self.learning_rate * 0.333,  # Lion uses ~1/3 the learning rate of Adam
                    "weight_decay": 0.1,
                    "betas": (0.9, 0.95),
                    "r": 0.98,
                    "use_gc": True,
                    "adanorm": True,
                    "cautious": True,
                },
                loss_fn=F.cross_entropy,  # Standard cross-entropy
                projection_layer=projection,
            )
            
            wrapped_layers.append(wrapped)
        
        # Replace the original layers
        self.locals = wrapped_layers

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
        """
        _, seq_len, _ = hidden_states.shape

        # For inference, prepare goodness accumulator
        if not self.training:
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
                losses.add_loss(f"goodness_layer_{layer_idx}", layer_loss.item() if hasattr(layer_loss, 'item') else float(layer_loss))
            
            # Apply compression and post-processing
            hidden_states = self.compressor.reduce_sequence(hidden_states)
            hidden_states = self.post_layer(hidden_states, layer_idx)
            
            # For inference, accumulate goodness
            if not self.training:
                with torch.no_grad():
                    # Compute goodness for this layer
                    goodness = wrapped_layer.projection(hidden_states)
                    
                    # Accumulate all layers' goodness
                    min_len = min(goodness.size(1), total_goodness.size(1))
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

        # Return appropriate output
        if self.training:
            # During training, we need to provide a path for embedding/head training
            # Since all our hidden states are detached, we'll signal that a separate
            # forward pass is needed for embeddings/head
            losses.add_loss("_layer_wise_complete", 0.0)
            
            # Return the final detached hidden states
            # The model will need to do a separate forward for embeddings/head
            return hidden_states, past_key_values, current_state, losses
        else:
            # During inference, return goodness scores for classification
            return total_goodness, past_key_values, current_state, losses
    
    def parameters(self, recurse: bool = True):
        """Override to exclude layer parameters (they have their own optimizers)."""
        # Only return non-layer parameters (e.g., controller parameters)
        for name, param in self.named_parameters(recurse=recurse):
            # Skip parameters from wrapped layers
            if not name.startswith('locals.'):
                yield param