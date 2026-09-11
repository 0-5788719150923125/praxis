"""Tests for praxis/data/utils.py: ``rl_type`` pulls in the datasets its
policies are bound to, and nothing else."""

import pytest


class TestRlDatasetBinding:
    """`rl_type` must be sufficient on its own to configure a working RL task.

    These policies filter hard on task tags, so a mix missing the dataset that
    emits the tag makes them score zero positions - no loss, no metric, nothing
    that distinguishes it from the policy being off.
    """

    def _ids(self, train, rl_type):
        from praxis.data.utils import get_dataset_configs

        cfg = get_dataset_configs(train, [], rl_type=rl_type)
        return [e["_id"] for e in cfg["primary"]]

    def test_rl_type_alone_pulls_its_data(self):
        ids = self._ids(["focused"], ["engagement", "joke"])
        assert "synthetic-print" in ids
        assert "rated-jokes" in ids
        assert "hh-rlhf" in self._ids(["print"], "preference")

    def test_restricted_dataset_needs_its_policy(self):
        """hh-rlhf's card permits preference modeling only, so no collection
        may carry it and it cannot be requested without the policy."""
        from praxis.data.config import DATASET_COLLECTIONS, DATASETS
        from praxis.data.utils import add_datasets

        assert DATASETS["hh-rlhf"]["requires_rl_type"] == ("preference",)
        for name, members in DATASET_COLLECTIONS.items():
            assert "hh-rlhf" not in members, f"collection '{name}' leaks hh-rlhf"

        # Anything but the owning policy is refused, loudly.
        with pytest.raises(ValueError, match="reserved for rl_type"):
            add_datasets({"primary": []}, {"hh-rlhf": 1.0}, "primary")
        assert "hh-rlhf" not in self._ids(["focused"], ["engagement"])

    def test_no_rl_type_adds_nothing(self):
        assert self._ids(["joke"], None) == ["rated-jokes"]

    def test_auto_include_never_duplicates(self):
        """A duplicate entry does not error - it doubles that dataset's
        sampling weight, so the mix silently changes under the user."""
        from collections import Counter

        # Same collection listed explicitly...
        ids = self._ids(["focused", "print", "joke"], ["engagement", "joke"])
        assert not [k for k, v in Counter(ids).items() if v > 1]
        # ...and the harder case: overlap at the DATASET level, where a policy
        # injects a dataset a requested collection already carries.
        ids = self._ids(["print", "joke"], ["engagement", "joke"])
        assert not [k for k, v in Counter(ids).items() if v > 1]
        assert self._ids(["print"], "preference").count("hh-rlhf") == 1

    def test_weight_controller_leaves_the_mix_alone(self):
        assert self._ids(["print"], None) == self._ids(
            ["print"], "harmonic_weight_wave"
        )
