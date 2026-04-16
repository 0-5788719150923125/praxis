"""Per-layer gradient dynamics extraction shared across trainer backends."""

from typing import Dict, Optional

import torch.nn as nn


def extract_layer_dynamics(
    layer: nn.Module, lr: float
) -> Optional[Dict[str, float]]:
    """Compute gradient norm, variance, and update ratio for a single layer.

    Returns a dict with keys ``grad_norm``, ``grad_var``, and
    ``update_ratio``, or ``None`` if no parameter in *layer* has a
    populated ``.grad``.

    Args:
        layer: The module whose parameters to inspect (gradients must
            already be populated, i.e. call after backward but before
            optimizer.zero_grad).
        lr: Current learning rate, used to compute the update-to-weight
            ratio ``||grad|| * lr / ||weight||``.
    """
    grad_norms_sq = []
    grad_vars = []
    weight_norms_sq = []

    for param in layer.parameters():
        if param.grad is None:
            continue
        grad_norms_sq.append(param.grad.norm().item() ** 2)
        grad_vars.append(param.grad.var().item())
        weight_norms_sq.append(param.norm().item() ** 2)

    if not grad_norms_sq:
        return None

    grad_norm = sum(grad_norms_sq) ** 0.5
    weight_norm = sum(weight_norms_sq) ** 0.5
    grad_var = sum(grad_vars) / len(grad_vars)
    update_ratio = (grad_norm * lr) / max(weight_norm, 1e-8)

    return {
        "grad_norm": float(grad_norm),
        "grad_var": float(grad_var),
        "update_ratio": float(update_ratio),
    }
