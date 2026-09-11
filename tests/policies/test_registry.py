"""Sweep of the ``rl_policies`` and ``rl_profiles`` registries: every policy
builds, and the data it declares a binding to exists."""

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.data.config import DATASET_COLLECTIONS, DATASETS
from praxis.policies import (
    needs_rl_datasets,
    resolves_to_weight_controller,
    rl_dataset_collections,
    rl_dataset_weights,
)


@pytest.mark.parametrize("name", sorted(registry.namespace("rl_policies")))
def test_policy_builds_and_binds_to_existing_data(name):
    cls = registry.lookup("rl_policies", name)
    assert isinstance(cls(PraxisConfig(hidden_size=64, dropout=0.0)), torch.nn.Module)

    collections, weights = rl_dataset_collections(name), rl_dataset_weights(name)
    assert all(c in DATASET_COLLECTIONS for c in collections), collections
    for dataset in weights:
        # A restricted dataset may only be injected by a policy it names.
        assert name in DATASETS[dataset].get("requires_rl_type", (name,))
    if getattr(cls, "is_weight_controller", False):
        # Weight controllers reward from a callback and pull no data.
        assert (collections, weights, needs_rl_datasets(name)) == ((), {}, False)


@pytest.mark.parametrize("name", sorted(registry.namespace("rl_profiles")))
def test_rl_profile_resolves_to_a_weight_controller(name):
    assert resolves_to_weight_controller(name)
    assert rl_dataset_collections(name) == () and not needs_rl_datasets(name)


@pytest.mark.parametrize(
    "name,collections,weights,needs_rl",
    [
        ("engagement", ("print",), {}, False),
        ("joke", ("joke",), {}, False),
        # hh-rlhf is bound dataset-level (restricted), not via a collection.
        ("preference", (), {"hh-rlhf": 1.0}, False),
        ("reinforce", ("rl",), {}, True),
        ("grpo", ("rl",), {}, True),
        ("cot", ("cot",), {}, True),
        ("harmonic_weight_wave", (), {}, False),
        # An unregistered legacy name fails open rather than loading nothing.
        ("cot-reinforce", ("cot",), {}, True),
    ],
)
def test_declared_data_bindings(name, collections, weights, needs_rl):
    assert rl_dataset_collections(name) == collections
    assert rl_dataset_weights(name) == weights
    assert needs_rl_datasets(name) is needs_rl
