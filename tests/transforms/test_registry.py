"""Ghost features: the algebra, the in-place parametrization, and the walker.

The properties pinned here are the ones the experiment's interpretation rests on.
If the complex product is wrong the arm is not testing the paper's mechanism; if
the init does not inherit the host module's scale the arms differ by an init as
well as a mechanism; if the walker misses a target the run is not the experiment.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.transforms import MIN_TARGET_NUMEL, apply_transform

# --- the walker -------------------------------------------------------------


class _Toy(nn.Module):
    """Mimics the real qualified names closely enough to exercise the profile
    regexes, with decoys each profile must NOT match. Every tensor is sized over
    MIN_TARGET_NUMEL so the floor is not what the test measures."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=100)
        self.encoder = nn.Module()
        for half in ("encoder", "decoder"):
            stack = nn.Module()
            stack.layers = nn.ModuleList()
            for _ in range(3):
                block = nn.Module()
                block.conv = nn.Conv1d(32, 64, kernel_size=3)
                block.proj = nn.Linear(64, 64, bias=False)
                stack.layers.append(block)
            setattr(self.encoder, half, stack)
        self.decoder = nn.Module()
        self.decoder.conv = nn.Conv1d(32, 64, kernel_size=3)  # conv decoy
        self.mtp = nn.Module()
        self.mtp.bank = nn.Module()
        self.mtp.bank.depths = nn.ModuleList()
        for _ in range(3):
            d = nn.Module()
            d.projection = nn.Linear(128, 64)
            d.norm = nn.Linear(64, 64)  # mtp decoy
            self.mtp.bank.depths.append(d)
        self.embeds = nn.Embedding(64, 128)
        self.bag = nn.EmbeddingBag(64, 128, mode="sum")
        self.lm_head = nn.Linear(64, 100)  # vocab-dimensioned
        self.tiny = nn.Linear(8, 8)  # under MIN_TARGET_NUMEL
        self.odd = nn.Linear(128, 63)  # indivisible on the output axis


@pytest.mark.parametrize("profile", sorted(registry.namespace("transforms")))
def test_every_profile_builds_and_runs(profile):
    model = _Toy()
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
