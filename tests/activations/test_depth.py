"""Per-depth activation specialization (praxis/activations/depth.py)."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.activations.depth import (
    DepthActivation,
    base_activation,
    build_depth_activation,
    depth_passes,
)


def _cfg(**kwargs):
    base = dict(depth=6, num_layers=1, hidden_size=64, embed_size=64)
    base.update(kwargs)
    return PraxisConfig(**base)


def test_one_instance_per_recurrent_pass():
    """A depth-6 stack of single-layer blocks gets six independent activations,
    each a real module with its own parameters."""
    act = build_depth_activation("serpent", _cfg())
    assert isinstance(act, DepthActivation)
    assert len(act.passes) == 6
    assert all(a is not act.passes[0] for a in act.passes[1:])


def test_pass_index_cycles_by_num_layers():
    """A block sees current_depth {i, i+num_layers, ...}, so the pass index is
    current_depth // num_layers - and wraps rather than indexing past the end."""
    act = build_depth_activation("serpent", _cfg(depth=6, num_layers=2))
    assert len(act.passes) == 3
    assert [act.pass_index(d) for d in range(8)] == [0, 0, 1, 1, 2, 2, 0, 0]


def test_each_pass_computes_its_own_shape():
    """Different depths route to different instances, so perturbing one pass
    moves that depth's output and leaves the others alone."""
    torch.manual_seed(0)
    act = build_depth_activation("serpent", _cfg())
    x = torch.randn(2, 4, 64)
    before = [act(x, current_depth=d) for d in range(6)]

    with torch.no_grad():
        act.passes[2].a.add_(1.0)
    after = [act(x, current_depth=d) for d in range(6)]

    assert not torch.allclose(before[2], after[2])
    for d in (0, 1, 3, 4, 5):
        assert torch.allclose(before[d], after[d])


def test_parameter_free_activations_collapse_to_one():
    """SiLU is the same function at every depth, so N copies would be N
    identical modules and N curves saying one thing."""
    act = build_depth_activation("swish", _cfg())
    assert act.shared and len(act.passes) == 1
    assert act.training_metrics() == {}


def test_disabled_and_non_recurrent_carry_no_wrapper():
    """Off, or a stack that never revisits a block, returns a bare activation -
    no wrapper, no renamed state-dict keys."""
    off = build_depth_activation("serpent", _cfg(depth_activations=False))
    flat = build_depth_activation("serpent", _cfg(depth=1, num_layers=1))
    assert not isinstance(off, DepthActivation)
    assert not isinstance(flat, DepthActivation)


def test_specialization_metrics_track_divergence():
    """The diagnostic answers 'are the passes still the same function?'.
    Forcing every pass to identical parameters collapses specialization to 0
    and similarity to 1."""
    act = build_depth_activation("serpent", _cfg())
    act(torch.randn(2, 4, 64))  # materialize the lazy params

    with torch.no_grad():
        for a in act.passes[1:]:
            for name in ("a", "b", "g"):
                getattr(a, name).copy_(getattr(act.passes[0], name))

    metrics = act.training_metrics()
    assert metrics["depth_act_specialization"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["depth_act_similarity"] == pytest.approx(1.0, abs=1e-5)


def test_legacy_checkpoints_broadcast_to_every_pass():
    """A checkpoint written before the wrapper stored params at the activation's
    own prefix. Every pass starts from that one learned shape."""
    import torch.nn as nn

    holder = nn.Module()
    holder.act = build_depth_activation("serpent", _cfg())
    holder.act(torch.randn(2, 4, 64))

    features = holder.act.passes[0].a.shape
    legacy = {
        "act.a": torch.full(features, 0.25),
        "act.b": torch.full(features, 0.5),
        "act.g": torch.full(features, 0.75),
    }
    holder.load_state_dict(legacy)

    for a in holder.act.passes:
        assert torch.allclose(a.a, torch.full(features, 0.25))
        assert torch.allclose(a.g, torch.full(features, 0.75))


def test_base_activation_unwraps_for_introspection():
    """Introspection asks about the function CLASS, which is pass 0's answer."""
    act = build_depth_activation("serpent", _cfg())
    assert base_activation(act) is act.passes[0]
    bare = build_depth_activation("serpent", _cfg(depth_activations=False))
    assert base_activation(bare) is bare


def test_depth_passes_matches_arc_glu():
    """The pass count is the same ceil(depth / num_layers) ArcGLU uses, so the
    two mechanisms cannot disagree about how many passes exist."""
    assert depth_passes(_cfg(depth=6, num_layers=1)) == 6
    assert depth_passes(_cfg(depth=6, num_layers=4)) == 2
    assert depth_passes(_cfg(depth=3, num_layers=3)) == 1
