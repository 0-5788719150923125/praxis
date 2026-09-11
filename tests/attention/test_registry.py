"""Every entry of the ``attention`` registry: it builds, runs, is causal, and its
metrics reach the dashboard.

The grid is deterministic: every entry at one small shape, every encoding only on
the entries that read ``config.encoding``, and ModularAttention's own mode axes
only on ModularAttention.
"""

import copy
import itertools
import math

import pytest
import torch
import torch.nn as nn

from praxis import PraxisConfig, registry
from praxis.attention.causal import CausalAttention
from praxis.attention.infini import _DEFAULT_SEGMENT_SIZE
from praxis.attention.modular import ModularAttention
from praxis.attention.syntaxes import SyntaxesAttention

ATTENTION = sorted(registry.namespace("attention"))
ENCODINGS = sorted(registry.namespace("encoding"))
HIDDEN = 64


def _class(key):
    entry = registry.lookup("attention", key)
    return getattr(entry, "func", entry)  # profiles are functools.partial


# The kaleidoscope and SSOG fields have no Q/K to encode, and ignore it.
READS_ENCODING = [
    key
    for key in ATTENTION
    if issubclass(_class(key), (ModularAttention, SyntaxesAttention, CausalAttention))
]
HAS_METRICS = [key for key in ATTENTION if hasattr(_class(key), "metric_descriptions")]


def _build(key, **fields):
    config = PraxisConfig(hidden_size=HIDDEN, num_heads=2, num_queries=1, dropout=0.0)
    config.causal = True  # modeling.py sets this at assembly; the bare config is False
    for name, value in fields.items():
        setattr(config, name, value)
    torch.manual_seed(0)
    return registry.lookup("attention", key)(config)


def _run(module, batch=2, seq_len=16):
    x = torch.randn(batch, seq_len, HIDDEN)
    out, _, _ = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("key", ATTENTION)
def test_forward(key):
    _run(_build(key))


@pytest.mark.parametrize("key", ATTENTION)
def test_no_zero_dim_parameters(key):
    """Schedule-free optimizers swap parameters through ``x.view(torch.uint8)``,
    which a 0-dim tensor cannot do. This crashed a run at step 1."""
    zero_dim = [n for n, p in _build(key).named_parameters() if p.dim() == 0]
    assert zero_dim == [], zero_dim


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("key", READS_ENCODING)
def test_forward_encoding(key, encoding):
    _run(_build(key, encoding=encoding))


@pytest.mark.parametrize("encoding", ["rope", "hope", "arc"])
@pytest.mark.parametrize("key", READS_ENCODING)
def test_forward_odd_head_dim(key, encoding):
    """64 / 3 heads gives an odd head_dim, which the rotary encodings must pass
    through on their unpaired last dimension."""
    _run(_build(key, encoding=encoding, num_heads=3))


MODULAR_GRID = list(
    itertools.product(
        [None, "differential", "stickbreaking", "mla"],  # mode
        [None, 2],  # kv_rank
        [None, 2],  # k_heads
        [False, True],  # mega
    )
)


@pytest.mark.parametrize(
    "mode, kv_rank, k_heads, mega",
    MODULAR_GRID,
    ids=[f"{m}-kv{r}-k{k}-mega{g}" for m, r, k, g in MODULAR_GRID],
)
def test_modular_grid(mode, kv_rank, k_heads, mega):
    modes = {m: m == mode for m in ("differential", "stickbreaking", "mla")}
    module = _build(
        "modular", num_queries=2, kv_rank=kv_rank, k_heads=k_heads, mega=mega, **modes
    )
    _run(module)


# ------------------------------------------------------------------------------
# causality
# ------------------------------------------------------------------------------
# One token changes in one row; no output at an earlier position of that row, and
# none in the other row, may move - in training and in inference. The method is
# tests/test_modeling.py's: every forward runs on a fresh copy under the same RNG
# (training forwards mutate state), and every parameter is first moved off its
# initialization so zero-initialized paths cannot hide a leak.

# Longer than every window default, so the windowed entries run past their windows:
# SyntaxesAttention keeps the last 128 tokens, Infini folds 256-token segments.
SYNTAXES_CONTEXT = 128
SEQ = _DEFAULT_SEGMENT_SIZE + 16
# Edits land where a window opens - a leak into a window's past shows first at
# its first position - plus one inside a window.
EDITS = (SEQ - SYNTAXES_CONTEXT, SEQ - SYNTAXES_CONTEXT // 2, _DEFAULT_SEGMENT_SIZE)
TOLERANCE = 1e-6


def _movement(key, train):
    """Worst change before an edited position and in the other row, over all
    edits, and the largest change the edits made where they may."""
    pristine = _build(key)
    with torch.no_grad():
        for p in pristine.parameters():
            if p.is_floating_point():
                p.add_(0.05 * torch.randn_like(p))
    torch.manual_seed(1)
    x = torch.randn(2, SEQ, HIDDEN)

    def outputs(inputs):
        module = copy.deepcopy(pristine).train(train)
        torch.manual_seed(0)
        with torch.no_grad():
            return module(inputs)[0]

    base = outputs(x)
    before = other_row = reached = 0.0
    for position in EDITS:
        edited = x.clone()
        edited[0, position] += 1.0
        delta = (outputs(edited) - base).abs().amax(dim=-1)
        before = max(before, delta[0, :position].max().item())
        other_row = max(other_row, delta[1].max().item())
        reached = max(reached, delta[0, position:].max().item())
    return before, other_row, reached


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("key", ATTENTION)
def test_causal(key, train):
    before, other_row, reached = _movement(key, train)
    assert reached > 0, "no edit moved anything, so the check cannot see a leak"
    assert before <= TOLERANCE, f"an output before the edit moved by {before:.3e}"
    assert other_row <= TOLERANCE, f"an output in another row moved by {other_row:.3e}"


# ------------------------------------------------------------------------------
# dashboard
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("key", HAS_METRICS)
def test_metrics_reach_the_dashboard(key):
    """Attention modules have no loss hook and are not an attribute of the model
    the way the head and encoder are, so a module walk is the only way anything
    of theirs is reachable. Three walks must each find the module: values
    (dynamics.db), declarations (a logged key with no ``metric_descriptions``
    entry is written and then dropped, since the manifest is built from
    declarations), and live snapshots - through the precompute recipe too, as
    the /api/head_snapshots route is only the cold-start fallback. Arc has a walk
    of its own, so each value is counted once."""
    from praxis.metrics.descriptions import get_metric_descriptions
    from praxis.metrics.specialization import (
        collect_arc_metrics,
        collect_attention_metrics,
        collect_attention_snapshots,
    )
    from praxis.web.snapshots import _recipe_head_snapshots

    module = _build(key).train()
    module(torch.randn(2, 16, HIDDEN))
    model = nn.Sequential(nn.Identity(), module)
    metrics = {k: v for k, v in module.training_metrics().items() if v is not None}
    snapshots = (
        module.dashboard_snapshots() if hasattr(module, "dashboard_snapshots") else {}
    )

    assert all(math.isfinite(v) for v in metrics.values())
    collected = {**collect_attention_metrics(model), **collect_arc_metrics(model)}
    assert collected.keys() == metrics.keys()
    assert collect_attention_snapshots(model).keys() == snapshots.keys()

    descriptions = get_metric_descriptions(model)
    for name in metrics:
        assert descriptions.get(name, {}).get("chart"), name
    for name in snapshots:
        assert descriptions.get(name, {}).get("snapshot"), name
    # The dashboard labels each card with the class that raised it.
    for name in list(metrics)[:1]:
        assert descriptions[name]["caller"] == type(module).__name__

    if snapshots:
        payload = _recipe_head_snapshots(model)
        assert payload["status"] == "ok"
        assert set(snapshots) <= set(payload["snapshots"])
