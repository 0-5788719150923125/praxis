"""A toy model whose qualified names exercise the transform profiles."""

from types import SimpleNamespace

import pytest
import torch.nn as nn


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


@pytest.fixture
def toy_model():
    """``toy_model()`` builds a fresh toy; tests that need two call it twice."""
    return _Toy
