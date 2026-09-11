"""LowRankSecondMoment: factored second-moment telemetry, passthrough update."""

import torch
import torch.nn as nn

from praxis.optimization.low_rank_moment import LowRankSecondMoment


def _problem():
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def _step(opt, model, X, Y):
    opt.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_update_is_passthrough():
    # Tracking must not change the trajectory: wrapped == bare base optimizer.
    m1, X, Y = _problem()
    m2, _, _ = _problem()  # identical seed => identical init
    base = torch.optim.SGD(m1.parameters(), lr=0.1, momentum=0.9)
    wrapped = LowRankSecondMoment(
        torch.optim.SGD(m2.parameters(), lr=0.1, momentum=0.9)
    )
    for _ in range(50):
        _step(base, m1, X, Y)
        _step(wrapped, m2, X, Y)
    assert torch.allclose(m1.weight, m2.weight, atol=1e-6)


def test_factored_storage_for_2d():
    model, X, Y = _problem()  # weight is [4, 8]
    opt = LowRankSecondMoment(torch.optim.SGD(model.parameters(), lr=0.1))
    _step(opt, model, X, Y)
    p = next(iter(model.parameters()))
    st = opt.state[p]
    assert st["vr"].shape == (4,) and st["vc"].shape == (8,)  # O(out+in)
    assert "v" not in st  # never materializes the full [4, 8] second moment


def test_full_vector_for_non_2d():
    model = nn.Sequential(nn.Linear(4, 4))  # bias is 1D
    opt = LowRankSecondMoment(torch.optim.SGD(model.parameters(), lr=0.1))
    X = torch.randn(16, 4)
    opt.zero_grad()
    (model(X) ** 2).mean().backward()
    opt.step()
    bias = model[0].bias
    assert "v" in opt.state[bias] and opt.state[bias]["v"].shape == bias.shape


def test_reconstruction_matches_rank1_gradient():
    # If g**2 is exactly separable (a outer b) and constant, the factored
    # estimate reconstructs it (Adafactor is exact at rank 1).
    torch.manual_seed(1)
    w = nn.Parameter(torch.randn(5, 3))
    a, b = torch.rand(5) + 0.5, torch.rand(3) + 0.5
    g = torch.outer(a.sqrt(), b.sqrt())  # so g**2 = a outer b
    opt = LowRankSecondMoment(torch.optim.SGD([w], lr=0.0))
    for _ in range(2000):  # converge the EMA past bias correction
        w.grad = g.clone()
        opt.step()
    v = opt.get_second_moment(w)
    assert torch.allclose(v, torch.outer(a, b), rtol=1e-3, atol=1e-3)
