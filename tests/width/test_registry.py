"""Every entry of the ``width`` registry: a scoped step keeps the residual stream at
full width at every depth, and leaving the scope restores the block exactly."""

from types import MethodType

import pytest
import torch

from praxis import registry


@pytest.mark.parametrize("key", sorted(registry.namespace("width")))
def test_scope_preserves_shape_and_restores_on_exit(key, glu):
    policy = registry.lookup("width", key)()
    block, x = glu(), torch.randn(2, 4, 16)
    with torch.no_grad():
        before = block(x)
        for depth in range(6):
            with policy.scope([block], current_depth=depth, max_depth=6):
                assert block(x).shape == before.shape
        assert torch.equal(block(x), before)
    assert block.up.weight.shape == (48, 16)  # full 2*inner
    for module in block.modules():
        assert not module._forward_pre_hooks and not module._forward_hooks
        # No patched forward left behind (restoring the bound original is fine).
        assert module.forward == MethodType(type(module).forward, module)
