"""SyntaxesAttention: every query attends to one shared window of the most recent
``syntaxes_context_size`` tokens."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.attention.syntaxes import SyntaxesAttention

CONTEXT = 16
HIDDEN = 64


def _attention(**fields):
    config = PraxisConfig(
        hidden_size=HIDDEN,
        num_heads=4,
        num_queries=1,
        dropout=0.0,
        encoding="nope",
        syntaxes_context_size=CONTEXT,
    )
    for name, value in fields.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    return SyntaxesAttention(config)


@pytest.mark.parametrize("seq_len", [4, CONTEXT, 3 * CONTEXT])
def test_forward_shape(seq_len):
    """Shorter than, equal to, and past the context window."""
    attention = _attention()
    assert attention.context_size == CONTEXT
    inputs = torch.randn(2, seq_len, HIDDEN)
    with torch.no_grad():
        output, past_kv, aux_loss = attention(inputs=inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert aux_loss == 0
    assert past_kv is None


def test_queries_before_the_window_attend_to_nothing():
    """A query older than the shared window has no key in its past, so its output
    is zero - not NaN, and not a read of the window's first key, which is in its
    future."""
    attention = _attention()
    seq_len = 3 * CONTEXT
    start = seq_len - CONTEXT
    inputs = torch.randn(2, seq_len, HIDDEN, requires_grad=True)
    output, _, _ = attention(inputs=inputs)
    assert torch.equal(output[:, :start], torch.zeros_like(output[:, :start]))
    assert output[:, start:].abs().amax() > 0
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()


def test_causal_past_the_window():
    """Editing a window token moves no earlier output, including the edit that
    lands on the window's first position."""
    attention = _attention(encoding="rope")
    seq_len = 3 * CONTEXT
    inputs = torch.randn(2, seq_len, HIDDEN)
    with torch.no_grad():
        base, _, _ = attention(inputs=inputs)
        for position in (seq_len - CONTEXT, seq_len - CONTEXT // 2, seq_len - 3):
            edited = inputs.clone()
            edited[:, position] += 1.0
            moved, _, _ = attention(inputs=edited)
            assert torch.equal(moved[:, :position], base[:, :position]), position
            assert not torch.equal(moved[:, position:], base[:, position:]), position


def test_attention_mask_hides_a_context_key():
    """A masked key contributes nothing: changing that token moves only its own
    position's output, through its query."""
    attention = _attention()
    seq_len = 2 * CONTEXT
    hidden = seq_len - CONTEXT // 2
    mask = torch.ones(2, seq_len)
    mask[:, hidden] = 0
    inputs = torch.randn(2, seq_len, HIDDEN)
    edited = inputs.clone()
    edited[:, hidden] += 1.0
    with torch.no_grad():
        base, _, _ = attention(inputs=inputs, attention_mask=mask)
        moved, _, _ = attention(inputs=edited, attention_mask=mask)
    others = torch.arange(seq_len) != hidden
    assert torch.equal(moved[:, others], base[:, others])
    assert not torch.equal(moved[:, hidden], base[:, hidden])


@pytest.mark.xfail(
    strict=True,
    reason="syntaxes.py unpacks before_scores(k, k, v) as `k, v, _`, so the values "
    "are the keys and v_proj never runs",
)
def test_every_parameter_gets_gradient():
    attention = _attention()
    inputs = torch.randn(2, 2 * CONTEXT, HIDDEN, requires_grad=True)
    output, _, _ = attention(inputs=inputs)
    output.mean().backward()
    assert inputs.grad.norm() > 0
    for name, param in attention.named_parameters():
        assert param.grad is not None and param.grad.norm() > 0, name


def test_kv_caching_not_implemented():
    attention = _attention()
    inputs = torch.randn(2, CONTEXT, HIDDEN)
    past_key_values = torch.randn(2, CONTEXT, HIDDEN)
    with pytest.raises(NotImplementedError, match="KV caching not yet supported"):
        attention(inputs=inputs, past_key_values=past_key_values)
