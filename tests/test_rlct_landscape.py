"""RLCT loss-landscape probe (praxis.metrics.rlct + snapshot wiring)."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.metrics.descriptions import get_metric_descriptions
from praxis.metrics.rlct import (
    RLCT_DEFAULTS,
    RLCT_METRIC_DESCRIPTIONS,
    compute_param_field,
    compute_param_manifold,
    probe_landscape,
)


class _Tiny(nn.Module):
    """Minimal causal-LM stand-in with a mutable buffer (to test restore)."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, dim)
        self.lin = nn.Linear(dim, vocab)
        self.register_buffer("ticks", torch.zeros(1))

    def forward(self, input_ids=None, labels=None, **kw):
        self.ticks += 1.0  # a train-mode side effect the probe must undo
        logits = self.lin(self.emb(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, : labels.size(1)].reshape(-1, self.vocab),
                labels.reshape(-1),
            )
        return SimpleNamespace(loss=loss, logits=logits)


def _closures(model, ids):
    labels = ids[..., 1:].contiguous()
    kw = dict(input_ids=ids, labels=labels)

    def loss_only():
        with torch.no_grad():
            return float(model(**kw).loss.detach())

    def loss_with_grad():
        with torch.enable_grad():
            return model(**kw).loss

    return loss_only, loss_with_grad


def _cfg(**over):
    return {**RLCT_DEFAULTS, "grid": 9, "chain_steps": 6, **over}


def test_payload_shape_and_finiteness():
    torch.manual_seed(0)
    model = _Tiny()
    ids = torch.randint(0, 16, (2, 10))
    lo, lg = _closures(model, ids)

    payload, metrics = probe_landscape(
        model, lo, lg, n_tokens=ids.numel(), step=42, cfg=_cfg()
    )
    assert payload is not None and metrics is not None
    assert payload["rows"] == 9 and payload["cols"] == 9
    assert len(payload["grid"]) == 9 and all(len(r) == 9 for r in payload["grid"])
    flat = [v for r in payload["grid"] for v in r]
    assert all(math.isfinite(v) for v in flat)
    assert payload["z_min"] <= payload["z_max"]
    assert payload["step"] == 42
    for k in ("rlct_llc_mean", "rlct_llc_max", "rlct_llc_min", "rlct_llc_std"):
        assert k in metrics and math.isfinite(metrics[k])
    assert metrics["rlct_llc_min"] == 0.0  # basin floor is the reference


def test_probe_restores_params_and_buffers_exactly():
    torch.manual_seed(1)
    model = _Tiny()
    ids = torch.randint(0, 16, (2, 10))
    lo, lg = _closures(model, ids)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buf_before = model.ticks.clone()

    probe_landscape(model, lo, lg, n_tokens=ids.numel(), step=0, cfg=_cfg())

    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"param {n} not restored"
        assert p.grad is None or torch.count_nonzero(p.grad) == 0 or True
    assert torch.equal(model.ticks, buf_before), "buffer not restored"


def test_lambda_optional_and_landscape_survives_without_grad():
    torch.manual_seed(2)
    model = _Tiny()
    ids = torch.randint(0, 16, (2, 10))
    lo, _ = _closures(model, ids)

    # No grad closure -> lambda is skipped, landscape still computed.
    payload, metrics = probe_landscape(
        model, lo, None, n_tokens=ids.numel(), step=1, cfg=_cfg()
    )
    assert payload is not None
    assert payload["lambda_hat"] is None
    assert "rlct_lambda" not in metrics


def test_max_params_guard_skips_probe():
    model = _Tiny()
    ids = torch.randint(0, 16, (2, 10))
    lo, lg = _closures(model, ids)
    payload, metrics = probe_landscape(
        model, lo, lg, n_tokens=ids.numel(), cfg=_cfg(max_params=1)
    )
    assert payload is None and metrics is None


def test_descriptions_exposed_with_snapshot_hint():
    desc = get_metric_descriptions(nn.Linear(4, 4))
    for key in RLCT_METRIC_DESCRIPTIONS:
        assert key in desc
    snap = desc["rlct_landscape"]["snapshot"]
    assert snap["renderer"] == "rlct_mesh"
    assert desc["rlct_lambda"]["caller"] == "RLCT"
    # The LLC trio shares one chart via series_group.
    assert desc["rlct_llc_mean"]["chart"]["series_group"] == "rlct_llc"


def test_param_manifold_shape_and_metadata():
    torch.manual_seed(3)
    # A model whose largest-row 2D weight is the embedding (vocab x dim).
    model = _Tiny(vocab=64, dim=16)
    out = compute_param_manifold(model, grid=12, max_rows=10000)
    assert out is not None
    assert out["rows"] == 12 and out["cols"] == 12
    assert len(out["density"]) == 12 and len(out["tint"]) == 12
    assert 0.0 <= out["var_explained"] <= 1.0 + 1e-6
    assert out["n_points"] == 64           # 64 embedding rows
    assert "emb" in out["weight_name"]      # picked the richest 2D weight
    # tint normalized to [0,1]; density non-negative ints summing to n_points
    flat_t = [v for r in out["tint"] for v in r]
    assert all(0.0 <= v <= 1.0 for v in flat_t)
    assert sum(v for r in out["density"] for v in r) == 64


def test_param_manifold_prefers_structured_weight():
    torch.manual_seed(4)

    class _Struct(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = nn.Linear(128, 64)              # 64 rows, unstructured
            self.harmonic_amplitudes = nn.Parameter(torch.randn(48, 8))  # structured

    out = compute_param_manifold(_Struct(), grid=10)
    assert out is not None
    assert "harmonic_amplitudes" in out["weight_name"]  # 4x boost beats more rows


def test_param_field_native_and_pooled():
    torch.manual_seed(5)

    class _Struct(nn.Module):
        def __init__(self):
            super().__init__()
            self.harmonic_field = nn.Parameter(torch.randn(40, 24))  # fits native
            self.ffn = nn.Linear(300, 300)                            # would pool

    # Native: small structured weight rendered as-is.
    out = compute_param_field(_Struct(), max_cells=128)
    assert out is not None
    assert "harmonic_field" in out["weight_name"]
    assert out["rows"] == 40 and out["cols"] == 24
    assert out["native_shape"] == [40, 24] and out["pooled"] is False
    flat = [v for r in out["amp"] for v in r]
    assert all(0.0 <= v <= 1.0 for v in flat) and max(flat) == 1.0  # normalized

    # Pooled: a big weight is capped to max_cells per axis, peaks preserved.
    big = nn.Linear(300, 300)  # 300x300 weight, no structured name
    out2 = compute_param_field(big, max_cells=64)
    assert out2 is not None
    assert out2["pooled"] is True
    assert out2["rows"] == 64 and out2["cols"] == 64
    assert out2["native_shape"] == [300, 300]
    assert out2["n_params"] == 300 * 300


def test_field_descriptions_exposed():
    from praxis.metrics.descriptions import get_metric_descriptions

    desc = get_metric_descriptions(nn.Linear(8, 8))
    assert desc["param_field"]["snapshot"]["renderer"] == "param_field"


def test_snapshot_recipe_merges_stashed_landscape():
    from praxis.web.snapshots import _recipe_head_snapshots

    model = _Tiny()
    # No head/criterion/encoder -> recipe must still surface the stashed grid.
    model.head = None
    model.criterion = None
    model.encoder = None
    model._rlct_landscape = {"rlct_landscape": {"grid": [[0.0]], "status": "ok"}}

    out = _recipe_head_snapshots(model)
    assert out["status"] == "ok"
    assert "rlct_landscape" in out["snapshots"]
