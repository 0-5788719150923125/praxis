"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest

from praxis import PraxisConfig


def test_memory_net_has_no_lazy_params():
    """The memory net exposes only concrete parameters, whether the profile
    uses the parameter-free default (gelu) or opts into a learnable/lazy
    activation (serpent). A lazy UninitializedParameter would crash the
    per-sequence weight expansion in init_state, so build_memory_model
    materializes them up front."""
    from praxis.memory import build_memory_model

    cfg = PraxisConfig(hidden_size=64, activation="serpent")

    # Default (no activation in spec) -> gelu, no lazy params.
    default_net = build_memory_model(cfg, {"dense": "mlp", "layers": 2})
    assert {type(p).__name__ for p in default_net.parameters()} == {"Parameter"}

    # Opt into serpent -> lazy per-feature freqs are materialized to concrete
    # Parameters, and they are present (they become fast weights).
    serpent_net = build_memory_model(
        cfg, {"dense": "mlp", "layers": 2, "activation": "serpent"}
    )
    assert {type(p).__name__ for p in serpent_net.parameters()} == {"Parameter"}
    from praxis.activations.serpent import Serpent

    assert any(isinstance(m, Serpent) for m in serpent_net.modules())
