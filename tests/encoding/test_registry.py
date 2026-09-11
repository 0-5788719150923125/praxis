"""Every entry of the ``encoding`` registry honours the interface the attention
modules call: ``before_scores`` keeps Q/K shapes and passes V through, and only
ALiBi builds a FlexAttention score_mod (the rest work through ``before_scores``)."""

import pytest
import torch

from praxis import PraxisConfig, registry

ENCODINGS = sorted(registry.namespace("encoding"))


def _encoding(key):
    config = PraxisConfig(
        hidden_size=64, num_heads=4, num_queries=1, block_size=512, dropout=0.0
    )
    config.encoding = key
    return registry.lookup("encoding", key)(config)


@pytest.mark.parametrize("key", ENCODINGS)
def test_before_scores_keeps_shapes_and_passes_values_through(key):
    enc = _encoding(key)
    q, k, v = (torch.randn(2, 4, 32, 16) for _ in range(3))
    q2, k2, v2 = enc.before_scores(q, k, v)
    assert q2.shape == q.shape and k2.shape == k.shape
    assert v2 is v


@pytest.mark.parametrize("key", ENCODINGS)
def test_only_alibi_builds_a_score_mod(key):
    mod = _encoding(key).build_score_mod(num_heads=4, device=torch.device("cpu"))
    assert (mod is not None) == (key == "alibi")
