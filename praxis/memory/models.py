"""Builds the memory network whose weights are updated at test time.

The network is just a dense module from ``praxis.dense``, so the memory can
reuse any feedforward variant (MLP today, PEER/KAN/... later) by name. The
profile picks the ``dense`` type, ``activation``, and number of ``layers``;
dropout is forced off so the surprise gradient stays deterministic.
"""

import copy
import inspect

import torch.nn as nn

from praxis.dense import DENSE_REGISTRY


def build_memory_model(config, spec: dict) -> nn.Module:
    """Construct the test-time memory network (a ``dim -> dim`` map) for a
    profile spec. Reuses ``praxis.dense`` so memory and the FFN share variants.
    """
    cfg = copy.copy(config)
    # A parameter-free activation by default: the memory net's whole parameter
    # set is then just the fast-weight matrices we update at test time, and we
    # avoid pulling lazy/learnable activation params (e.g. serpent) into it.
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
    return dense_cls(cfg, **kwargs)
