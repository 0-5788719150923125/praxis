"""Sweeps over every ``transforms`` profile."""

import pytest
import torch

from praxis import registry
from praxis.transforms import apply_transform


@pytest.mark.parametrize("profile", sorted(registry.namespace("transforms")))
def test_every_profile_builds_and_runs(toy_model, profile):
    model = toy_model()
    stats = apply_transform(model, profile)
    assert stats.after < stats.before
    if "_mtp_" in profile:
        assert len(stats.targets) == 3
        assert all(n.endswith(".projection") for n, _, _, _ in stats.targets)
        y = model.mtp.bank.depths[0].projection(torch.randn(2, 128))
    elif "_conv_" in profile:
        assert len(stats.targets) == 6
        y = model.encoder.encoder.layers[0].conv(torch.randn(2, 32, 10))
    else:
        # Broad profiles reach every host type, not just one.
        names = [n for n, _, _, _ in stats.targets]
        assert "embeds" in names and "bag" in names
        assert any(n.endswith(".conv") for n in names)
        assert any(n.endswith(".projection") for n in names)
        y = model.embeds(torch.randint(0, 64, (2, 5)))
    y.sum().backward()


def test_every_unrestricted_profile_requests_alignment():
    """The guard on adding a profile. A spec that matches any name reaches the
    auto-sized modules, so it is one of the profiles they should be asking about;
    forgetting the flag would silently leave the largest banks indivisible."""
    for name, entry in registry.namespace("transforms").items():
        unrestricted = entry.spec.matches("decoder.0.ffn.down") and entry.spec.matches(
            "encoder.encoder.layers.0.conv"
        )
        assert unrestricted == entry.request_alignment, name
