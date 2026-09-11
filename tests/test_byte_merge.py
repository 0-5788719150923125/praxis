"""The local decoder's byte/trunk merge, and the in-context copy probe that
says whether the trunk's long-range signal reaches the output at all."""

import pytest
import torch
import torch.nn as nn

from praxis.encoders.byte_latent.merge import MERGES, PatchMerge
from praxis.metrics.copy_probe import copy_gain

DIM = 16


def test_add_is_the_blt_sum():
    merge = PatchMerge(DIM, "add")
    byte, trunk = torch.randn(2, 5, DIM), torch.randn(2, 5, DIM)
    torch.testing.assert_close(merge(byte, trunk), byte + trunk)


def test_gated_starts_at_an_even_split_of_normalized_streams():
    merge = PatchMerge(DIM, "gated").train()
    byte, trunk = torch.randn(2, 5, DIM) * 300, torch.randn(2, 5, DIM)
    out = merge(byte, trunk)
    norm = lambda x: torch.nn.functional.rms_norm(x, (DIM,), eps=1e-6)
    torch.testing.assert_close(out, 0.5 * norm(byte) + 0.5 * norm(trunk))
    metrics = merge.training_metrics()
    assert metrics["merge_gate_trunk"] == pytest.approx(0.5)
    assert metrics["merge_trunk_ratio"] == pytest.approx(1.0, rel=1e-4)


def test_gated_share_cannot_be_won_by_scale():
    """Gated, scaling either stream changes nothing."""
    merge = PatchMerge(DIM, "gated")
    with torch.no_grad():
        nn.init.normal_(merge.gate.weight)
    byte, trunk = torch.randn(2, 5, DIM), torch.randn(2, 5, DIM)
    torch.testing.assert_close(merge(byte * 100, trunk), merge(byte, trunk))
    torch.testing.assert_close(merge(byte, trunk * 100), merge(byte, trunk))


def test_gated_merge_reads_only_its_own_position():
    merge = PatchMerge(DIM, "gated")
    with torch.no_grad():
        nn.init.normal_(merge.gate.weight)
    byte, trunk = torch.randn(1, 6, DIM), torch.randn(1, 6, DIM)
    edited = trunk.clone()
    edited[0, 4] += torch.randn(DIM)
    a, b = merge(byte, trunk), merge(byte, edited)
    torch.testing.assert_close(a[0, :4], b[0, :4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(a[0, 5:], b[0, 5:], rtol=0.0, atol=0.0)


def test_both_modes_report_the_magnitudes_and_only_gated_reports_a_gate():
    for mode in MERGES:
        merge = PatchMerge(DIM, mode).train()
        merge(torch.randn(2, 5, DIM), torch.randn(2, 5, DIM))
        keys = set(merge.training_metrics())
        assert {"merge_trunk_ratio", "merge_trunk_content_ratio"} <= keys
        assert ("merge_gate_trunk" in keys) == (mode == "gated")
        assert keys <= set(PatchMerge.metric_descriptions)


def test_an_unknown_merge_is_refused():
    with pytest.raises(ValueError, match="Unknown merge"):
        PatchMerge(DIM, "concat")


def test_the_unimplemented_cross_attention_decoder_is_refused():
    from praxis import PraxisConfig
    from praxis.encoders.byte_latent.encoder import ByteLatentEncoder

    config = PraxisConfig(hidden_size=32, embed_size=32, vocab_size=1024)
    with pytest.raises(NotImplementedError, match="cross_attn_decoder"):
        ByteLatentEncoder(config, cross_attn_decoder=True)


# --- the copy probe ------------------------------------------------------------


class _Copier(nn.Module):
    """Predicts the token that followed the previous occurrence of the current
    one, when there is one - an ideal in-context copier - and uniform otherwise."""

    def __init__(self, vocab=64):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        for row in range(b):
            for i in range(t):
                seen = (input_ids[row, :i] == input_ids[row, i]).nonzero().flatten()
                if len(seen):
                    logits[row, i, input_ids[row, seen[-1] + 1]] = 30.0
        return type("Out", (), {"logits": logits})()


class _Amnesiac(_Copier):
    """Same interface, no use of context at all."""

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        return type("Out", (), {"logits": torch.zeros(b, t, self.vocab)})()


def _distinct_rows(rows=2, t=40, vocab=64):
    """Rows of distinct tokens, so the only way to predict a repeat is to copy."""
    torch.manual_seed(0)
    return torch.stack([torch.randperm(vocab)[:t] for _ in range(rows)])


def test_copy_gain_separates_a_copier_from_a_model_without_context():
    ids = _distinct_rows()
    assert float(copy_gain(_Amnesiac(), ids, length=20)) == pytest.approx(0.0, abs=1e-6)
    gain = float(copy_gain(_Copier(), ids, length=20))
    assert gain > 5.0  # log2(64) = 6 bits per token, all but free the second time


def test_copy_gain_runs_the_uncompiled_module():
    wrapper = nn.Module()
    wrapper._orig_mod = _Copier()
    wrapper.forward = lambda **kwargs: pytest.fail("the compiled module was called")
    assert float(copy_gain(wrapper, _distinct_rows(), length=20)) > 5.0


def test_copy_gain_skips_what_it_cannot_read():
    assert copy_gain(_Copier(), torch.arange(6).view(1, -1)) is None  # too short
    codec = _Copier()
    codec.encoder = type("Encoder", (), {"handles_loss": True})()
    assert copy_gain(codec, _distinct_rows(), length=20) is None
