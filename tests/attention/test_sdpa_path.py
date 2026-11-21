"""Eager GPU attention runs on SDPA; it must match the FlexAttention path."""

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.attention.self_attention import SelfAttention


def test_score_mods_keep_flex_attention():
    assert SelfAttention._prefers_sdpa(None)
    assert not SelfAttention._prefers_sdpa(lambda score, b, h, q, k: score)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention is GPU-only")
@pytest.mark.parametrize(
    "key, encoding, window",
    [
        ("self_attention", "rope", None),
        ("self_attention", "rope", 8),
        ("kuramoto", "rope_learned_rates", None),
        ("kuramoto_capsules", "rope_learned_rates", None),
    ],
)
def test_sdpa_matches_flex(monkeypatch, key, encoding, window):
    def run(sdpa):
        if not sdpa:
            monkeypatch.setattr(
                SelfAttention, "_prefers_sdpa", staticmethod(lambda score_mod: False)
            )
        torch.manual_seed(0)
        config = PraxisConfig(
            hidden_size=64, num_heads=4, num_queries=1, dropout=0.0, encoding=encoding
        )
        if window:
            config.window_size = window
        module = registry.lookup("attention", key)(config).cuda().train()
        torch.manual_seed(1)
        x = torch.randn(2, 50, 64, device="cuda", requires_grad=True)
        block_ids = torch.tensor([[1] * 20 + [2] * 30, [1] * 50], device="cuda")
        out, _, _ = module(x, block_ids=block_ids)
        out.square().sum().backward()
        return out.detach(), x.grad

    out, grad = run(sdpa=True)
    flex_out, flex_grad = run(sdpa=False)
    torch.testing.assert_close(out, flex_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(grad, flex_grad, atol=1e-5, rtol=1e-4)
