"""Parameter counting utilities for models and optimizers."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def count_model_parameters(model: nn.Module) -> int:
    """Count total parameters in a model.

    Args:
        model: The PyTorch model to count parameters for

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_optimizer_parameters(
    optimizer: torch.optim.Optimizer, model: Optional[nn.Module] = None
) -> int:
    """Count total parameters in optimizer state.

    Args:
        optimizer: The main optimizer
        model: Optional model to check for layer-wise optimizers (e.g., MonoForward decoder)

    Returns:
        Total number of parameters in optimizer state
    """
    optimizer_params = 0

    # Count parameters in main optimizer
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for k, v in state.items():
                if torch.is_tensor(v):
                    optimizer_params += v.numel()

    # Check for layer-wise optimizers in MonoForward decoder
    if model and hasattr(model, "decoder") and hasattr(model.decoder, "locals"):
        decoder = model.decoder
        # Check if layers have their own optimizers (LayerWithOptimizer wrapper)
        for layer in decoder.locals:
            if hasattr(layer, "optimizer") and layer.optimizer is not None:
                opt = layer.optimizer
                for group in opt.param_groups:
                    for p in group["params"]:
                        state = opt.state.get(p, {})
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                optimizer_params += v.numel()

    return optimizer_params


def get_parameter_stats(
    model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, int]:
    """Get simplified parameter statistics.

    Args:
        model: The model to count parameters for
        optimizer: Optional optimizer to count state parameters

    Returns:
        Dictionary with 'model_parameters' and 'optimizer_parameters' counts
    """
    stats = {
        "model_parameters": count_model_parameters(model),
        "optimizer_parameters": 0,
    }

    if optimizer is not None:
        stats["optimizer_parameters"] = count_optimizer_parameters(optimizer, model)

    return stats
