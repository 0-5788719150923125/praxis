"""FullWidth: the ``none`` policy masks nothing and has no profile."""

import torch

from praxis import registry


def test_full_width_is_noop_and_has_no_profile(glu, active_channels):
    policy = registry.lookup("width", "none")()
    assert policy.profile(8) is None
    block, x = glu(), torch.randn(2, 3, 16)
    assert len(active_channels(block, policy, 0, 8, x)) == 24  # nothing masked
