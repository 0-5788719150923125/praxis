"""Builds the memory network whose weights are updated at test time.

The network is just a dense module from ``praxis.dense``, so the memory can
reuse any feedforward variant (MLP today, PEER/KAN/... later) by name. The
profile picks the ``dense`` type, ``activation``, and number of ``layers``;
dropout is forced off so the surprise gradient stays deterministic.
"""

import copy
import inspect

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from praxis.dense import DENSE_REGISTRY


def _materialize_lazy(model: nn.Module, dim: int) -> None:
    """Force any lazy parameters (e.g. Serpent's per-feature frequencies) to
    materialize, so the memory harness can enumerate ``named_parameters()`` and
    drive the net through ``functional_call`` - both need concrete tensors.

    A single dim->dim dummy forward triggers ``LazyModuleMixin`` initialization;
    dropout is already off, and plain MLP memory nets carry no forward-updated
    buffers, so this leaves no state behind. On CPU/fp32 at construction; the
    materialized params move with the later ``.to(device)``.
    """
    if not any(isinstance(p, UninitializedParameter) for p in model.parameters()):
        return
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, dim))
    model.train(was_training)


def build_memory_model(config, spec: dict) -> nn.Module:
    """Construct the test-time memory network (a ``dim -> dim`` map) for a
    profile spec. Reuses ``praxis.dense`` so memory and the FFN share variants.
    """
    cfg = copy.copy(config)
    # Default to a parameter-free activation (gelu): the memory net's parameter
    # set is then just the fast-weight matrices updated at test time. A profile
    # may name a *learnable* activation instead (e.g. serpent) - its per-feature
    # params then join the fast weights, so the test-time surprise update tunes
    # the activation's own frequencies online, not only the linear maps. Lazy
    # params from such activations are materialized below before the memory
    # harness reads them.
    cfg.activation = spec.get("activation", "gelu")
    cfg.dropout = 0.0  # surprise gradient must be deterministic

    dense_cls = DENSE_REGISTRY[spec.get("dense", "mlp")]

    # Pass num_layers/hidden_dim only to variants that accept them explicitly,
    # so dense modules that don't (PEER, KAN) aren't handed args they'd misroute.
    # The memory net stays small by default (hidden = 1x dim): its weights carry
    # a batch x num_chunks dimension, so the FFN's 4x width would blow up VRAM.
    params = inspect.signature(dense_cls.__init__).parameters
    kwargs = {}
    if "num_layers" in params:
        kwargs["num_layers"] = spec.get("layers", 2)
    if "hidden_dim" in params:
        kwargs["hidden_dim"] = int(config.hidden_size * spec.get("expansion", 1.0))
    model = dense_cls(cfg, **kwargs)
    _materialize_lazy(model, config.hidden_size)
    return model
