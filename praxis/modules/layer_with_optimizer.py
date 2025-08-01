"""LayerWithOptimizer: A module wrapper that includes its own optimizer for immediate updates."""

from typing import Dict, Any, Optional, Callable, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class LayerWithOptimizer(nn.Module):
    """
    Wraps a layer with its own optimizer for immediate weight updates.
    
    This enables true Mono-Forward training where each layer:
    1. Receives detached inputs (no gradients from previous layers)
    2. Computes its output
    3. Computes its loss (if training)
    4. Updates weights immediately
    5. Returns detached output
    
    The optimizer state is saved/loaded with the module for checkpointing.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        optimizer_class: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable] = None,
        projection_layer: Optional[nn.Module] = None,
    ):
        """
        Initialize LayerWithOptimizer.
        
        Args:
            layer: The actual layer/module to wrap
            optimizer_class: Optimizer class to use (any torch.optim.Optimizer subclass)
            optimizer_kwargs: Keyword arguments for optimizer
            loss_fn: Optional custom loss function. If None, uses cross entropy
            projection_layer: Optional projection for computing predictions
        """
        super().__init__()
        
        self.layer = layer
        self.projection = projection_layer
        self.loss_fn = loss_fn or F.cross_entropy
        
        # Create optimizer for this layer's parameters
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        params_to_optimize = list(self.layer.parameters())
        if self.projection is not None:
            params_to_optimize.extend(self.projection.parameters())
        
        self.optimizer = optimizer_class(params_to_optimize, **optimizer_kwargs)
        
        # Statistics
        self.last_loss = None
        self.update_count = 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional immediate weight update.
        
        Args:
            hidden_states: Input tensor (will be detached)
            labels: Optional labels for computing loss
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for the layer
            
        Returns:
            Tuple of (output, loss) where output is detached
        """
        # CRITICAL: Detach inputs to prevent gradient flow from previous layers
        hidden_states = hidden_states.detach()
        
        # Enable gradients for this layer's computation
        hidden_states.requires_grad_(True)
        
        # Forward through the wrapped layer
        # Handle LocalExpert which expects positional arguments
        if hasattr(self.layer, '_forward'):
            # This is likely a LocalExpert
            current_state = kwargs.get('current_state', None)
            past_key_values = kwargs.get('past_key_values', None)
            current_depth = kwargs.get('current_depth', 0)
            block_ids = kwargs.get('block_ids', None)
            
            output = self.layer(
                hidden_states,
                current_state,
                attention_mask,
                past_key_values,
                current_depth,
                block_ids
            )
        elif attention_mask is not None:
            output = self.layer(hidden_states, attention_mask=attention_mask, **kwargs)
        else:
            output = self.layer(hidden_states, **kwargs)
        
        # Handle different output types
        if isinstance(output, tuple):
            hidden_output = output[0]
            other_outputs = output[1:]
        else:
            hidden_output = output
            other_outputs = ()
        
        loss = None
        
        # If training and labels provided, compute loss and update immediately
        if self.training and labels is not None:
            # Compute predictions
            if self.projection is not None:
                logits = self.projection(hidden_output)
            else:
                logits = hidden_output
            
            # Ensure shapes match for loss computation
            if logits.dim() == 3 and labels.dim() == 2:
                # Typical case: [batch, seq, vocab] vs [batch, seq]
                batch_size, seq_len = labels.shape
                if logits.size(1) > seq_len:
                    logits = logits[:, :seq_len, :]
                elif logits.size(1) < seq_len:
                    labels = labels[:, :logits.size(1)]
                
                # Flatten for loss computation
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)
            else:
                logits_flat = logits
                labels_flat = labels
            
            # Compute loss
            loss = self.loss_fn(logits_flat, labels_flat, ignore_index=-100)
            
            # IMMEDIATE WEIGHT UPDATE
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            self.last_loss = loss.detach().item()
            self.update_count += 1
            
            # Detach loss to prevent further gradient accumulation
            loss = loss.detach()
        
        # CRITICAL: Detach output to prevent gradient flow to next layers
        hidden_output = hidden_output.detach()
        
        # Reconstruct output format
        if other_outputs:
            output = (hidden_output,) + other_outputs
        else:
            output = hidden_output
        
        return output, loss
    
    def state_dict(self, *args, **kwargs):
        """
        Override to include optimizer state in checkpoint.
        """
        state = super().state_dict(*args, **kwargs)
        state['_optimizer_state_dict'] = self.optimizer.state_dict()
        state['_update_count'] = self.update_count
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override to load optimizer state from checkpoint.
        """
        # Extract our custom entries
        optimizer_state = state_dict.pop('_optimizer_state_dict', None)
        update_count = state_dict.pop('_update_count', 0)
        
        # Load the regular state
        super().load_state_dict(state_dict, strict=strict)
        
        # Load optimizer state if available
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        
        self.update_count = update_count
    
    def __repr__(self):
        """Custom representation showing optimizer info."""
        optimizer_name = type(self.optimizer).__name__
        optimizer_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 'N/A'
        
        # Build the representation
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  optimizer={optimizer_name}(lr={optimizer_lr}),")
        lines.append(f"  (layer): {repr(self.layer)},")
        if self.projection is not None:
            lines.append(f"  (projection): {repr(self.projection)}")
        lines.append(")")
        
        return "\n".join(lines)
    
