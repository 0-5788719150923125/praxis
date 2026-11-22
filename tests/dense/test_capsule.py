"""HarmonicCapsules: routing symmetry, gradients, and the dense metrics walk."""

import math

import torch
from torch import nn

from praxis.dense.capsule import HarmonicCapsules


class Cfg:
    hidden_size = 32
    activation = "gelu"


def _module(**kwargs):
    torch.manual_seed(0)
    return HarmonicCapsules(Cfg(), atoms=4, **kwargs)


def test_shapes_finite_and_every_parameter_gets_gradient():
    net = _module()
    x = torch.randn(2, 7, 32)
    y = net(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_common_phase_shift_rotates_poses_and_leaves_routing_unchanged():
    """Shifting every phase by phi multiplies harmonic n by exp(i n phi): the
    coupling must not move and the output poses must rotate exactly so."""
    net = _module(harmonics=3).eval()
    with torch.no_grad():
        net.frequency.uniform_(0.5, 2.0)
    x = torch.randn(3, 5, 32, dtype=torch.float64)
    net = net.double()
    phi = 0.7
    shifted = x + phi / net.frequency.detach()

    poses, coupling = net.route(x)
    poses2, coupling2 = net.route(shifted)
    torch.testing.assert_close(coupling2, coupling)

    a = net.atoms
    z = torch.complex(poses[..., :a], poses[..., a:])
    z2 = torch.complex(poses2[..., :a], poses2[..., a:])
    n = torch.arange(1, net.harmonics + 1, dtype=torch.float64)
    rot = torch.polar(torch.ones_like(n), n * phi).view(-1, 1)
    torch.testing.assert_close(z2, z * rot)


def test_routing_moves_the_coupling_off_uniform():
    net = _module(iterations=3)
    _, coupling = net.route(torch.randn(4, 32))
    uniform = torch.full_like(coupling, 1.0 / net.outputs)
    assert not torch.allclose(coupling, uniform)
    _, flat = _module(iterations=1).route(torch.randn(4, 32))
    torch.testing.assert_close(flat, torch.full_like(flat, 1.0 / net.outputs))


def test_capsule_lengths_stay_below_one():
    net = _module()
    poses, _ = net.route(torch.randn(16, 32) * 10)
    lengths = poses.pow(2).sum(dim=(-2, -1)).sqrt()
    assert (lengths < 1).all() and (lengths > 0).all()


def test_dense_walk_logs_and_declares_each_layer():
    from praxis.metrics.specialization import (
        collect_dense_descriptions,
        collect_dense_metrics,
    )

    stack = nn.ModuleList([_module(), _module()]).train()
    x = torch.randn(2, 6, 32)
    for layer in stack:
        x = x + layer(x)

    metrics = collect_dense_metrics(stack)
    descriptions = collect_dense_descriptions(stack)
    for key in HarmonicCapsules.metric_descriptions:
        for layer in (0, 1):
            name = f"{key}_layer{layer}"
            assert name in metrics and math.isfinite(metrics[name]), name
            assert descriptions[name]["chart"]["series_label"] == f"layer {layer}"
    entropy = metrics["capsule_routing_entropy_layer0"]
    assert 0 < entropy < 1


def test_activation_chart_samples_the_phase_read():
    """The model walker finds no activation module in this FFN, so the
    activation charts were empty; the phase read stands in for one."""
    from praxis.web.routes.dynamics import _compute_activation_curves

    net = _module()
    with torch.no_grad():
        net.frequency.uniform_(0.3, 1.5)
    curves, kind = _compute_activation_curves(nn.Sequential(net), -6.0, 6.0, 64)
    assert kind == "HarmonicCapsules"
    assert [c["name"] for c in curves] == ["0.cos", "0.sin"]

    x = torch.tensor(curves[0]["x"])
    alpha = net.frequency.detach().sort().values[net.frequency.numel() // 2]
    for curve, fn in zip(curves, (torch.cos, torch.sin)):
        torch.testing.assert_close(torch.tensor(curve["forward"]), fn(alpha * x))
        assert len(curve["forward_low"]) == len(curve["backward_high"]) == 64
    torch.testing.assert_close(
        torch.tensor(curves[1]["backward"]),
        alpha * torch.cos(alpha * x),
        atol=0.02,
        rtol=0.0,
    )
