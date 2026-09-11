"""Sweep over the routers registry.

Every entry that builds against ONE shared block (``LAYER_LAYOUT == "shared"``,
the SMEAR family) is constructed from its key and trained for a step on the
toy block. The per-position routers need a real layer stack; their model-level
coverage is the causality sweep in tests/test_modeling.py.
"""

import pytest
import torch

from praxis import registry
from praxis.decoders.base import _router_layout
from tests.routers.toy_block import Block, Cfg, router_args

SHARED_KEYS = [
    key for key in registry.namespace("routers") if _router_layout(key) == "shared"
]


def test_the_smear_family_is_enumerated():
    assert {"smear", "vear", "smear_batch", "smear_token", "distance"} <= set(
        SHARED_KEYS
    )


@pytest.mark.parametrize("key", SHARED_KEYS)
def test_shared_layout_entries_build_and_train(key):
    cfg = Cfg()
    cfg.num_experts = 4
    block = Block(cfg.hidden_size)
    router = registry.lookup("routers", key)(cfg, block=block, verbose=False)
    x = torch.randn(3, 5, cfg.hidden_size)
    out = router(*router_args(block, x))[0]
    assert out.shape == x.shape
    out.sum().backward()
    assert block.attn.qkv.weight.grad.abs().sum() > 0
    assert router.wrappers["attn_qkv"].lora_b.grad is not None
