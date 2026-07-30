"""LionGeo: the SMEAR-of-norm-geometries optimizer (praxis/optimization/lion_geo.py)
and its composite/profile/metrics wiring."""

import copy
import importlib.util
import math
import types
from pathlib import Path

import torch
import torch.nn as nn

from praxis.optimization import get_optimizer, get_optimizer_profile
from praxis.optimization.composite import CompositeOptimizer
from praxis.optimization.lion_geo import (
    ADAPT_RATE,
    DIFF_DTYPE,
    GEOMETRIES,
    LOGIT_CLAMP,
    LionGeo,
)

N_ARMS = len(GEOMETRIES)
# The clamp alone bounds every share, whatever the centring does: with all
# logits in [-C, C], w_i lies in [e^-2C / N, 1 / (1 + (N-1) e^-2C)].
_SPREAD = math.exp(-2.0 * LOGIT_CLAMP)
SHARE_FLOOR = _SPREAD / N_ARMS
SHARE_CEIL = 1.0 / (1.0 + (N_ARMS - 1) * _SPREAD)


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
            assert state["geo_diffs"].shape == (N_ARMS, *p.shape)
            # Deviations feed a cosine only, so they are stored at half
            # precision: that is what keeps the third arm off the memory bill.
            assert state["geo_diffs"].dtype == DIFF_DTYPE
            assert state["geo_logits"].shape == (N_ARMS,)
            assert float(state["geo_logits"].abs().max()) <= LOGIT_CLAMP + 1e-6
    by_arm = opt.get_geometry_shares()
    assert set(by_arm) == set(GEOMETRIES)
    for shares in by_arm.values():
        assert shares  # no arm is extinguished, whatever the mixture chose
        assert all(SHARE_FLOOR - 1e-6 <= s <= SHARE_CEIL + 1e-6 for s in shares)
    # The mixture is a distribution: the arms sum to 1 at every matrix.
    for per_matrix in zip(*by_arm.values()):
        assert abs(sum(per_matrix) - 1.0) < 1e-5
    assert opt.get_smear_shares() == by_arm["spectral"]


def test_hypergradient_moves_logit_toward_aligned_arm():
    """Planting a deviation aligned with the next gradient must raise that
    arm's logit and lower the others (the hypergradient's sign convention,
    plus the centring that keeps the softmax's common shift pinned)."""
    p = nn.Parameter(torch.randn(4, 4))
    opt = LionGeo([p], lr=0.01)
    # Half-precision-exact, so storing it as a deviation round-trips losslessly
    # and the planted cosine is exactly 1 - which is what makes the arithmetic
    # below checkable to the last digit.
    g = torch.randn(4, 4).to(DIFF_DTYPE).float()
    p.grad = g.clone()
    opt.step()  # creates state; no geo_diffs existed yet, logits still 0
    assert torch.equal(opt.state[p]["geo_logits"], torch.zeros(N_ARMS))

    spectral = GEOMETRIES.index("spectral")
    aligned = torch.zeros(N_ARMS, *p.shape)  # zero rows contribute no cosine
    aligned[spectral] = g.clone()
    opt.state[p]["geo_diffs"] = aligned
    p.grad = g.clone()
    opt.step()
    logits = opt.state[p]["geo_logits"]
    up = float(logits[spectral])
    assert up > 0.0
    assert all(float(logits[i]) < 0.0 for i in range(N_ARMS) if i != spectral)
    assert abs(float(logits.mean())) < 1e-6  # centred
    # Uniform init: Jacobian is 4*(1/N)*(1-1/N), cosine is 1, and centring
    # keeps (N-1)/N of the nudge on the arm that earned it.
    w = 1.0 / N_ARMS
    expected = ADAPT_RATE * 4.0 * w * (1.0 - w) * (1.0 - w)
    assert abs(up - expected) < 1e-5

    anti = torch.zeros(N_ARMS, *p.shape)
    anti[spectral] = -g.clone()  # anti-aligned: pushes back down
    opt.state[p]["geo_diffs"] = anti
    p.grad = g.clone()
    opt.step()
    assert float(opt.state[p]["geo_logits"][spectral]) < up


def test_state_dict_roundtrip():
    model, X, Y = _quadratic_problem()
    opt = LionGeo(model.parameters(), lr=0.01)
    _train(opt, model, X, Y, steps=5)
    # state_dict shares tensor references; snapshot it like a checkpoint would.
    saved = copy.deepcopy(opt.state_dict())
    logits_before = opt.state[model.weight]["geo_logits"].clone()
    _train(opt, model, X, Y, steps=5)
    opt.load_state_dict(saved)
    assert torch.equal(opt.state[model.weight]["geo_logits"], logits_before)
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
    keys = ["opt_geo_share"] + [
        f"opt_geo_share_{name}" for name in GEOMETRIES if name != "spectral"
    ]
    assert all(0.0 < out[k] < 1.0 for k in keys)
    assert abs(sum(out[k] for k in keys) - 1.0) < 1e-5  # one matrix: a distribution
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
    assert cfg["optimizer"] == "LionGeo"  # the swap this experiment exists for
    # Nothing pins loss_func/head_type here: -e sets those directly and they
    # move with the live experiment. The inheritance below is the invariant.
    assert cfg["mtp_type"] == "serpent_rnn"  # -d inheritance intact
    assert cfg["residual_type"] == "smear"
