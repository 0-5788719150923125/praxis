"""Deep device-transfer for nn.Modules with non-registered tensor attrs."""

import torch
import torch.nn as nn


def _move_value(value, device: torch.device):
    """Recursively move tensors inside arbitrary containers to *device*."""
    if isinstance(value, torch.Tensor):
        if value.device != device:
            return value.to(device)
        return value
    if isinstance(value, dict):
        return {k: _move_value(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_value(v, device) for v in value)
    return value


def deep_to(module: nn.Module, device: torch.device) -> nn.Module:
    """Move a module to *device*, including non-registered tensor attrs.

    ``nn.Module.to(device)`` only moves parameters and buffers
    registered via ``register_parameter`` / ``register_buffer``. Some
    submodules store tensors as plain instance attributes (RoPE
    ``inv_freq``, cached cos/sin tables, pre-computed masks, etc.)
    that survive ``.to()`` on whatever device they were created on.

    This helper does ``.to(device)`` first (covers the common case),
    then sweeps every sub-module's ``__dict__`` for stray tensors -
    including tensors nested inside dicts, lists, and tuples - and
    moves them to the target device.
    """
    module = module.to(device)
    for submodule in module.modules():
        for attr_name, attr_value in list(vars(submodule).items()):
            moved = _move_value(attr_value, device)
            if moved is not attr_value:
                setattr(submodule, attr_name, moved)
    return module


def force_cpu(module: nn.Module) -> nn.Module:
    """Shorthand for ``deep_to(module, torch.device("cpu"))``."""
    return deep_to(module, torch.device("cpu"))
