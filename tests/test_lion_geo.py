"""LionGeo: the SMEAR-of-norm-geometries optimizer (praxis/optimizers/lion_geo.py)
and its composite/profile/metrics wiring."""

import copy
import importlib.util
import types
from pathlib import Path

import torch
import torch.nn as nn

from praxis.optimizers import get_optimizer, get_optimizer_profile
from praxis.optimizers.composite import CompositeOptimizer
from praxis.optimizers.lion_geo import ADAPT_RATE, LOGIT_CLAMP, LionGeo

SHARE_FLOOR = float(torch.sigmoid(torch.tensor(-LOGIT_CLAMP)))
SHARE_CEIL = float(torch.sigmoid(torch.tensor(LOGIT_CLAMP)))


def _quadratic_problem():
    # Minimize ||W x - y||^2 with a learnable target, so the optimum is ~0 and
    # a real optimizer should drive the loss down sharply.
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def _train(optimizer, model, X, Y, steps=200):
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_reduces_loss_on_rectangular_matrix():
    model, X, Y = _quadratic_problem()
    opt = LionGeo(model.parameters(), lr=0.01)
    losses = _train(opt, model, X, Y)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
    assert model.weight.shape == (4, 8)  # NS branch preserved the shape


def test_state_and_share_floor():
    model, X, Y = _quadratic_problem()
    opt = LionGeo(model.parameters(), lr=0.01)
    _train(opt, model, X, Y, steps=50)
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            assert state["exp_avg"].shape == p.shape
            assert state["geo_diff"].shape == p.shape
            assert float(state["geo_logit"].abs()) <= LOGIT_CLAMP + 1e-6
    shares = opt.get_smear_shares()
    assert shares and all(SHARE_FLOOR - 1e-6 <= s <= SHARE_CEIL + 1e-6 for s in shares)


def test_hypergradient_moves_logit_toward_aligned_branch():
    """Planting geo_diff aligned with the next gradient must raise the spectral
    logit; anti-aligned must lower it (the hypergradient's sign convention)."""
    p = nn.Parameter(torch.randn(4, 4))
    opt = LionGeo([p], lr=0.01)
    g = torch.randn(4, 4)
    p.grad = g.clone()
    opt.step()  # creates state; no geo_diff existed yet, logit still 0
    assert float(opt.state[p]["geo_logit"]) == 0.0

    opt.state[p]["geo_diff"] = g.clone()  # perfectly aligned with next grad
    p.grad = g.clone()
    opt.step()
    up = float(opt.state[p]["geo_logit"])
    assert up > 0.0
    # Jacobian at w=0.5 is 1.0, cosine is 1.0: the nudge is exactly ADAPT_RATE.
    assert abs(up - ADAPT_RATE) < 1e-5

    opt.state[p]["geo_diff"] = -g.clone()  # anti-aligned: pushes back down
    p.grad = g.clone()
    opt.step()
    assert float(opt.state[p]["geo_logit"]) < up


def test_state_dict_roundtrip():
    model, X, Y = _quadratic_problem()
    opt = LionGeo(model.parameters(), lr=0.01)
    _train(opt, model, X, Y, steps=5)
    # state_dict shares tensor references; snapshot it like a checkpoint would.
    saved = copy.deepcopy(opt.state_dict())
    logit_before = float(opt.state[model.weight]["geo_logit"])
    _train(opt, model, X, Y, steps=5)
    opt.load_state_dict(saved)
    assert float(opt.state[model.weight]["geo_logit"]) == logit_before
    _train(opt, model, X, Y, steps=1)  # still steps after restore


class _TinyLM(nn.Module):
    """Embedding + interior matrices + vocab head, with the config attr the
    Muon-style split reads."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=vocab)
        self.emb = nn.Embedding(vocab, dim)
        self.body = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        return self.head(self.body(self.emb(ids)))


def test_composite_build_split_and_metrics():
    torch.manual_seed(0)
    model = _TinyLM()
    profile, _ = get_optimizer_profile("LionGeo")
    opt = get_optimizer(model, wrappers=(), **profile)
    assert isinstance(opt, CompositeOptimizer)
    assert isinstance(opt.primary, LionGeo)

    primary_ids = {id(p) for g in opt.primary.param_groups for p in g["params"]}
    # Interior matrix on the smear; embedding and vocab head on the secondary.
    assert id(model.body.weight) in primary_ids
    assert id(model.emb.weight) not in primary_ids
    assert id(model.head.weight) not in primary_ids

    ids = torch.randint(0, 16, (4, 6))
    loss = model(ids).sum()
    loss.backward()
    opt.step()

    from praxis.metrics.optimizer import extract_optimizer_dynamics

    opt.zero_grad()
    model(ids).sum().backward()  # grads present, as in on_before_optimizer_step
    out = extract_optimizer_dynamics(opt)
    assert 0.0 < out["opt_geo_share"] < 1.0
    assert out["opt_geo_share_spread"] >= 0.0
    assert "opt_momentum_rms" in out  # exp_avg naming feeds the default suite


def test_abstractinator_e_resolves():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "praxis"
        / "cli"
        / "loaders"
        / "experiments.py"
    )
    spec = importlib.util.spec_from_file_location("_experiments_loader_e", module_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    cfg = loader.load_rendered_config(
        Path(__file__).resolve().parents[1] / "experiments" / "abstractinator-e.yml"
    )
    assert cfg["optimizer"] == "LionGeo"
    assert cfg["loss_func"] == "mode_cross_entropy"
    assert cfg["mtp_type"] == "serpent_rnn"  # -d inheritance intact
    assert cfg["residual_type"] == "smear"
