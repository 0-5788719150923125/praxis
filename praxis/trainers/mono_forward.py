"""MonoForward trainer implementing layer-wise training with immediate updates."""

from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.optim import Optimizer

from praxis.trainers.backpropagation import BackpropagationTrainer
from praxis.trainers.layer_wise import LayerWithOptimizer


class MonoForwardTrainer(BackpropagationTrainer):
    """
    Trainer that implements MonoForward training strategy.
    
    MonoForward trains each layer independently with its own optimizer,
    enabling O(1) memory complexity and immediate weight updates.
    """

    def __init__(
        self,
        model: nn.Module,
        cache_dir: str,
        ckpt_path: Optional[str] = None,
        precision: str = "bf16-mixed",
        **kwargs
    ):
        """
        Initialize MonoForward trainer.
        
        Args:
            model: The model to train
            cache_dir: Directory for caching checkpoints
            ckpt_path: Optional checkpoint path to resume from
            precision: Training precision (default: bf16-mixed)
            **kwargs: Additional arguments passed to parent
        """
        # Wrap decoder layers with LayerWithOptimizer if not already done
        self._wrap_model_layers(model)
        
        # Call parent init with the wrapped model
        super().__init__(model, cache_dir, ckpt_path, precision, **kwargs)
        
        # Override the training mode to use layer-wise updates
        self.layer_wise_training = True

    def _wrap_model_layers(self, model: nn.Module) -> None:
        """
        Wrap decoder layers with LayerWithOptimizer for immediate updates.
        
        Args:
            model: The model whose layers to wrap
        """
        # Check if model has a decoder with layers
        if not hasattr(model, 'decoder'):
            return
            
        decoder = model.decoder
        if not hasattr(decoder, 'locals'):
            return
            
        # Get optimizer config from model config
        config = getattr(model, 'config', None)
        if not config:
            return
            
        optimizer_config = getattr(config, 'optimizer_config', {})
        optimizer_class_name = optimizer_config.get('optimizer_class', 'Adam')
        optimizer_kwargs = optimizer_config.get('optimizer_kwargs', {'lr': 1e-3})
        
        # Get the optimizer class
        optimizer_class = self._get_optimizer_class(optimizer_class_name)
        
        # Wrap each layer if not already wrapped
        wrapped_layers = []
        for i, layer in enumerate(decoder.locals):
            if isinstance(layer, LayerWithOptimizer):
                # Already wrapped
                wrapped_layers.append(layer)
            else:
                # Create projection layer for this decoder layer
                projection = None
                if hasattr(config, 'hidden_size'):
                    vocab_size = getattr(config, 'vocab_size', 50257)
                    projection = nn.Linear(config.hidden_size, vocab_size, bias=False)
                
                # Wrap the layer
                wrapped_layer = LayerWithOptimizer(
                    layer=layer,
                    optimizer_class=optimizer_class,
                    optimizer_kwargs=optimizer_kwargs,
                    projection_layer=projection
                )
                wrapped_layers.append(wrapped_layer)
        
        # Replace the layers
        decoder.locals = nn.ModuleList(wrapped_layers)

    def _get_optimizer_class(self, name: str) -> Type[Optimizer]:
        """
        Get optimizer class from string name.
        
        Args:
            name: Name of the optimizer
            
        Returns:
            Optimizer class
        """
        # Try torch.optim first
        if hasattr(torch.optim, name):
            return getattr(torch.optim, name)
        
        # Try custom optimizers
        try:
            from praxis.optimizers import OPTIMIZER_REGISTRY
            if name in OPTIMIZER_REGISTRY:
                return OPTIMIZER_REGISTRY[name]
        except ImportError:
            pass
        
        # Default to Adam
        print(f"[MonoForward] Unknown optimizer '{name}', using Adam")
        return torch.optim.Adam

    def configure_optimizers(self):
        """
        Configure optimizers for MonoForward training.
        
        In MonoForward, each layer has its own optimizer, but we still
        need a main optimizer for non-layer parameters (embeddings, head, etc.)
        """
        # Get all parameters that are NOT in LayerWithOptimizer wrappers
        main_params = []
        
        # Collect parameters from non-wrapped modules
        for name, module in self.model.named_modules():
            # Skip LayerWithOptimizer modules and their children
            if isinstance(module, LayerWithOptimizer):
                continue
            
            # Check if this module is a child of LayerWithOptimizer
            is_child_of_wrapped = False
            for parent_name, parent_module in self.model.named_modules():
                if isinstance(parent_module, LayerWithOptimizer) and name.startswith(parent_name + '.'):
                    is_child_of_wrapped = True
                    break
            
            if not is_child_of_wrapped:
                # Add parameters from this module
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        main_params.append(param)
        
        if main_params:
            # Create optimizer for main parameters
            optimizer_config = getattr(self.model.config, 'optimizer_config', {})
            optimizer_class_name = optimizer_config.get('optimizer_class', 'Adam')
            optimizer_kwargs = optimizer_config.get('optimizer_kwargs', {'lr': 1e-3})
            
            optimizer_class = self._get_optimizer_class(optimizer_class_name)
            optimizer = optimizer_class(main_params, **optimizer_kwargs)
            
            return optimizer
        else:
            # No main parameters, create dummy optimizer
            # This is needed for Lightning compatibility
            return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-3)

    def training_step(self, batch, batch_idx):
        """
        Training step for MonoForward.
        
        The actual layer-wise updates happen inside the forward pass
        of LayerWithOptimizer modules. This just handles the overall loss.
        """
        # Run normal training step
        loss = super().training_step(batch, batch_idx)
        
        # The loss returned here is mainly for logging
        # The actual layer updates already happened
        return loss

    def on_train_start(self):
        """Hook called at the beginning of training."""
        super().on_train_start()
        print("[MonoForward] Starting layer-wise training with immediate updates")
        
        # Count wrapped layers
        wrapped_count = 0
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'locals'):
            for layer in self.model.decoder.locals:
                if isinstance(layer, LayerWithOptimizer):
                    wrapped_count += 1
        
        if wrapped_count > 0:
            print(f"[MonoForward] {wrapped_count} layers using independent optimizers")
        else:
            print("[MonoForward] Warning: No layers wrapped with LayerWithOptimizer")

    def on_train_epoch_end(self):
        """Hook called at the end of each training epoch."""
        super().on_train_epoch_end()
        
        # Log layer-wise statistics if available
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'locals'):
            total_updates = 0
            for i, layer in enumerate(self.model.decoder.locals):
                if isinstance(layer, LayerWithOptimizer) and hasattr(layer, 'update_count'):
                    total_updates += layer.update_count
            
            if total_updates > 0:
                print(f"[MonoForward] Total layer updates this epoch: {total_updates}")

    def __repr__(self):
        """String representation."""
        return f"MonoForwardTrainer(layer_wise={self.layer_wise_training})"