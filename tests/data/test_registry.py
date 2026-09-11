"""Sweep of the ``samplers`` registry: every weighting mode builds a manager
that serves a batch and keeps its mode's invariant."""

import pytest

from praxis import registry
from praxis.data.datasets.manager import WEIGHTING_MODES, InterleaveDataManager
from praxis.data.datasets.novelty import NoveltyTracker

SAMPLERS = sorted(registry.namespace("samplers"))


def test_samplers_registry_matches_the_manager_modes():
    assert set(SAMPLERS) == set(WEIGHTING_MODES)


@pytest.mark.parametrize("mode", SAMPLERS)
def test_sampler_mode_serves_a_batch(mode, default_tokenizer, make_sampler):
    manager = InterleaveDataManager(
        samplers=[make_sampler("a", task_type=0), make_sampler("b", task_type=1)],
        weights=[0.8, 0.2],
        tokenizer=default_tokenizer,
        block_size=128,
        weighting_mode=registry.lookup("samplers", mode),
    )
    batch = manager.get_batch(batch_size=1)
    assert batch["batch"] and batch["batch"][0].numel() > 0

    assert manager._adaptive is (mode != "static")
    assert hasattr(manager, "novelty_tracker") is (mode == "novelty")
    if mode in ("static", "uniform"):  # neither adapts the configured weights
        assert manager.weights == [0.8, 0.2]
    else:
        assert sum(manager.weights) == pytest.approx(1.0)
    if mode == "novelty":
        assert isinstance(manager.novelty_tracker, NoveltyTracker)
    if mode == "loss":
        assert InterleaveDataManager.shared_losses == {}
    if mode == "tasker":
        assert InterleaveDataManager.shared_task_weights == [1.0, 1.0]
