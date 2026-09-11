"""Muon param-splitting + CompositeOptimizer (Muon body / secondary head).

The invariants here guard the two things that historically broke Muon: it
must never orthogonalize an embedding or the LM head (the vocab-facing
params route to the secondary/AdamW), and the secondary's learning rate must
survive the cosine scheduler's per-group flattening as a fixed ratio.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.optimization import (
    CompositeOptimizer,
    _create_muon,
    get_optimizer,
    get_optimizer_profile,
)
from praxis.schedulers import get_scheduler_func


class TinyLM(nn.Module):
    """Minimal LM-shaped model: an embedding and head share a vocab dimension;
    the two interior linears are the only Muon-eligible matrices."""

    def __init__(self, vocab=50, hidden=16, tie=False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=128)
        self.embed = nn.Embedding(vocab, hidden)
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def forward(self, x):
        return self.lm_head(self.norm(self.h2(self.h1(self.embed(x)))))


def _find_composite(opt):
    """Walk the wrapper chain (HalfLion -> ... -> CompositeOptimizer)."""
    while opt is not None and not isinstance(opt, CompositeOptimizer):
        opt = getattr(opt, "optimizer", None)
    return opt


def _ce_step(model, opt):
    x = torch.randint(0, model.config.vocab_size, (8,))
    y = torch.randint(0, model.config.vocab_size, (8,))
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return float(loss.detach())


# --------------------------------------------------------------------------
# LR ratio survives the flattening cosine scheduler
# --------------------------------------------------------------------------


def test_composite_lr_ratio_survives_cosine_scheduler():
    model = TinyLM()
    profile, ds = get_optimizer_profile("Muon")
    opt = get_optimizer(model, wrappers=["low_rank_moment", "half_lion"], **profile)
    comp = _find_composite(opt)
    assert comp is not None

    expected = float(get_optimizer_profile("Lion")[0]["lr"]) / float(profile["lr"])
    sched = get_scheduler_func(
        optimizer_config=profile, disable_schedule=ds, warmup_steps=10
    )(opt)

    for _ in range(12):
        _ce_step(model, opt)
        # the rates each sub-optimizer ACTUALLY used this step, before the
        # scheduler re-flattens them:
        body = comp.primary.param_groups[0]["lr"]
        head = comp.secondary.param_groups[0]["lr"]
        sched.step()
    assert body > 0
    assert abs(head / body - expected) < 1e-6


# --------------------------------------------------------------------------
# It actually optimizes, checkpoints, and survives the wrapper stack
# --------------------------------------------------------------------------


def test_composite_reduces_loss():
    torch.manual_seed(0)
    model = TinyLM()
    profile, _ = get_optimizer_profile("Muon")
    opt = _create_muon(model, **profile)
    first = _ce_step(model, opt)
    for _ in range(60):
        last = _ce_step(model, opt)
    assert last < first
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_composite_state_dict_round_trip():
    profile, _ = get_optimizer_profile("Muon")
    model = TinyLM()
    opt = _create_muon(model, **profile)
    for _ in range(3):
        _ce_step(model, opt)
    sd = opt.state_dict()
    assert set(sd) == {"primary", "secondary", "secondary_lr_ratio"}

    opt2 = _create_muon(TinyLM(), **profile)
    opt2.load_state_dict(sd)  # must not raise
    assert opt2.secondary_lr_ratio == opt.secondary_lr_ratio


def test_composite_runs_under_wrapper_stack_with_rl_hook():
    model = TinyLM()
    profile, _ = get_optimizer_profile("Muon")
    opt = get_optimizer(model, wrappers=["low_rank_moment", "half_lion"], **profile)
    # half_lion exposes the RL controller's wave hook; it must survive wrapping
    # a composite base.
    assert hasattr(opt, "set_wave")
    for _ in range(5):
        _ce_step(model, opt)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_state_dict_survives_orphaned_state():
    """An optimizer state key not in any param_group (a param orphaned after its
    state was created) must not crash checkpointing - torch's state_dict would
    raise KeyError. The composite prunes it instead."""
    profile, _ = get_optimizer_profile("Muon")
    model = TinyLM()
    opt = _create_muon(model, **profile)
    for _ in range(3):
        _ce_step(model, opt)

    # Inject an orphan: state for a tensor that is in no param_group.
    orphan = torch.randn(4, 4, requires_grad=True)
    opt.primary.state[orphan] = {"momentum_buffer": torch.zeros(4, 4)}
    import pytest

    with pytest.raises(KeyError):
        opt.primary.state_dict()  # confirm the raw failure mode

    sd = opt.state_dict()  # composite prunes the orphan -> no raise
    assert set(sd) == {"primary", "secondary", "secondary_lr_ratio"}
    assert orphan not in opt.primary.state  # pruned in place
    opt2 = _create_muon(TinyLM(), **profile)
    opt2.load_state_dict(sd)  # and the result loads cleanly
