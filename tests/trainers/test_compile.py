"""try_compile (praxis/trainers/compile.py): skip paths, the eager fallback, and
compiling an optimizer's innermost step. ``torch.compile`` is faked throughout -
these test the wiring, not Inductor."""

import pytest
import torch
import torch.nn as nn

from praxis.environments import EnvironmentFeatures
from praxis.optimization.wrappers import SequentialWrapper
from praxis.trainers.compile import RECOMPILE_LIMIT, try_compile


@pytest.fixture(autouse=True)
def _compilation_allowed(monkeypatch):
    """An environment with skip_compilation set would short-circuit every path."""
    monkeypatch.setattr(
        EnvironmentFeatures, "is_enabled", classmethod(lambda c, f: False)
    )
    # try_compile_model sets this process-global before compiling.
    monkeypatch.setattr(
        torch._dynamo.config, "recompile_limit", torch._dynamo.config.recompile_limit
    )


def _refuse(*args, **kwargs):
    raise AssertionError("torch.compile must not be reached")


@pytest.mark.parametrize(
    "hparams", [{"device": "cpu"}, {"device": "cuda", "no_compile": True}]
)
def test_skip_paths_return_the_model_untouched(monkeypatch, hparams):
    monkeypatch.setattr(torch, "compile", _refuse)
    model = nn.Linear(4, 4)
    assert try_compile(model, hparams) is model


def test_compiles_off_cpu_and_raises_the_recompile_limit(monkeypatch):
    compiled = object()
    monkeypatch.setattr(torch, "compile", lambda model, **kw: compiled)
    assert try_compile(nn.Linear(4, 4), {"device": "cuda"}) is compiled
    assert torch._dynamo.config.recompile_limit == RECOMPILE_LIMIT


def test_try_compile_handles_exceptions(monkeypatch):
    """A compile failure falls back to the eager model rather than killing the run."""

    def fail(*args, **kwargs):
        raise RuntimeError("Compilation failed")

    monkeypatch.setattr(torch, "compile", fail)
    model = nn.Linear(4, 4)
    assert try_compile(model, {"device": "cuda"}) is model


def test_try_compile_optimizer_compiles_the_innermost_step(monkeypatch):
    """Wrappers delegate step() to the base, so the base's step is what compiles."""
    seen = []

    def fake_compile(fn, **kwargs):
        seen.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    base = torch.optim.SGD(nn.Linear(4, 4).parameters(), lr=0.1)
    wrapped = SequentialWrapper(["ortho"])(base)
    assert try_compile(wrapped, {"device": "cuda"}) is wrapped
    assert len(seen) == 1 and seen[0].__self__ is base
    assert "step" in vars(base)  # replaced on the instance
