"""``ModelBackend``: compile gating, ``eval_mode``, warmup, the deadline and bucketing.

``torch.compile`` is stubbed throughout - compiling for real would make these minutes
long and would test Inductor rather than this wiring. The measured payoff (1.7x on a
128-byte generation, byte-identical output) needs a GPU, so it is not asserted here.
"""

import contextlib
import time
from types import SimpleNamespace

import pytest
import torch

from praxis.environments import EnvironmentFeatures
from praxis.generation.bucketing import active_buckets
from praxis.generation.decode_backend import ModelBackend
from tests.stubs import build_memory_model, neural_memories


@pytest.fixture
def clean_features(monkeypatch):
    """An empty EnvironmentFeatures for the test, the real one restored after."""
    monkeypatch.setattr(EnvironmentFeatures, "_features", {})
    monkeypatch.setattr(EnvironmentFeatures, "_active_environment", None)


@pytest.fixture
def feature_on(clean_features):
    """The decode compile is opt-in per environment; most tests want it on."""
    EnvironmentFeatures.set_from_environment({"compile_decode_memory": True})


@pytest.fixture
def stub_compile(monkeypatch):
    """Replace torch.compile with a counting pass-through."""
    calls = {"n": 0}

    def fake(fn, **kwargs):
        calls["n"] += 1

        def wrapper(*args, **kw):
            calls.setdefault("invoked", 0)
            calls["invoked"] += 1
            return fn(*args, **kw)

        return wrapper

    monkeypatch.setattr(torch, "compile", fake)
    return calls


def test_backend_honors_no_compile(feature_on):
    assert ModelBackend(build_memory_model(), tokenizer=None)._compile_memory is True
    off = ModelBackend(build_memory_model(no_compile=True), tokenizer=None)
    assert off._compile_memory is False


def test_backend_unwraps_a_compiled_model():
    """Whole-model compile is ruinous at decode, so the backend must decode on
    the original module even when handed a wrapper."""
    model = build_memory_model()

    class FakeOptimizedModule:
        def __init__(self, mod):
            self._orig_mod = mod

    backend = ModelBackend(FakeOptimizedModule(model), tokenizer=None)
    assert backend.model is model


def test_eval_mode_restores_training_flag(feature_on, stub_compile):
    model = build_memory_model()
    model.train()
    backend = ModelBackend(model, tokenizer=None)
    with backend.eval_mode():
        assert not model.training
        assert all(m._decode_forward is not None for m in neural_memories(model))
    assert model.training
    assert all(m._decode_forward is None for m in neural_memories(model))


def test_eval_mode_does_not_force_eager(feature_on, stub_compile, monkeypatch):
    """The global ``force_eager`` stance would defeat the compiled body it
    installs - measured at 1.01x, i.e. nothing. If a stance ever comes back,
    this fails."""
    model = build_memory_model()
    backend = ModelBackend(model, tokenizer=None)
    seen = []

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.compiler, "set_stance", spy)
    with backend.eval_mode():
        pass
    assert seen == []


def test_warmup_compiles_before_any_request(feature_on, stub_compile):
    """The point of warmup: after it, no caller pays for Inductor."""
    model = build_memory_model()
    model.train()
    backend = ModelBackend(model, tokenizer=None)
    backend.warmup()
    assert stub_compile["n"] == len(neural_memories(model))
    assert stub_compile.get("invoked", 0) > 0
    # ... and it handed the model back exactly as it found it.
    assert all(m._decode_forward is None for m in neural_memories(model))
    assert model.training


def test_warmup_is_a_no_op_when_compilation_is_off(feature_on, stub_compile):
    backend = ModelBackend(build_memory_model(no_compile=True), tokenizer=None)
    backend.warmup()
    assert stub_compile["n"] == 0


def test_warmup_never_raises(feature_on, stub_compile, monkeypatch):
    """A warmup is an optimization; it must not be able to end a run."""
    model = build_memory_model()
    backend = ModelBackend(model, tokenizer=None)
    calls = []

    def explode(*a, **k):
        calls.append(1)
        raise RuntimeError("cuda is having a day")

    monkeypatch.setattr(type(model), "forward", explode)
    backend.warmup()  # must not raise
    assert calls, "warmup returned before the forward it has to survive"


def test_decode_compile_is_off_by_default(clean_features):
    """A rolling context grows, so static-shape decode graphs multiply without
    bound and keep Inductor's worker pool resident - measured at 2839MB mean
    child memory against 599MB with this off, and a swap-exhaustion kill two
    hours into abstractinator-u. Off unless an environment asks for it."""
    assert ModelBackend(build_memory_model(), tokenizer=None)._compile_memory is False


# ---------------------------------------------------------------------------
# the deadline: an absolute wall-clock time becomes a per-step relative budget
# ---------------------------------------------------------------------------


class _CapturingModel:
    """Records the GenerationConfig ``ModelBackend`` builds, then no-ops."""

    def __init__(self):
        self.configs = []
        self.config = None

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, tokens, generation_config=None, **kwargs):
        self.configs.append(generation_config)
        return SimpleNamespace(sequences=tokens)


def test_the_deadline_reaches_transformers_as_max_time(prose_tokenizer):
    """``MaxTimeCriteria`` is what transformers evaluates inside its own loop.

    The translation matters: the request carries an ABSOLUTE wall-clock
    deadline, while ``max_time`` is a budget measured from the moment
    ``generate`` builds its criteria. Recomputing it per call is what stops a
    turn that halts and resumes from handing each step the full timeout again.
    """
    model = _CapturingModel()
    backend = ModelBackend(model, prose_tokenizer)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    backend.generate_until_halt(ids, {"max_new_tokens": 4}, deadline=time.time() + 5.0)
    assert 4.0 < model.configs[-1].max_time <= 5.0

    # Second step of the same turn, a moment later: a SMALLER budget, not a
    # fresh 5 seconds.
    time.sleep(0.05)
    backend.generate_until_halt(ids, {"max_new_tokens": 4}, deadline=time.time() + 0.5)
    assert 0.0 < model.configs[-1].max_time <= 0.5


def test_an_already_expired_deadline_becomes_a_zero_budget(prose_tokenizer):
    """Clamped rather than skipped: a zero budget halts before the first token,
    which is what an expired request should cost. Passing a negative through
    would be silently ignored by MaxTimeCriteria's `elapsed > max_time`."""
    model = _CapturingModel()
    backend = ModelBackend(model, prose_tokenizer)
    backend.generate_until_halt(
        torch.tensor([[1]], dtype=torch.long),
        {"max_new_tokens": 4},
        deadline=time.time() - 10.0,
    )
    assert model.configs[-1].max_time == 0.0


def test_no_deadline_leaves_max_time_unset(prose_tokenizer):
    """``/input`` polls forever and passes no deadline, so no time criterion
    should be built at all."""
    model = _CapturingModel()
    backend = ModelBackend(model, prose_tokenizer)
    backend.generate_until_halt(torch.tensor([[1]], dtype=torch.long), {})
    assert model.configs[-1].max_time is None


# ---------------------------------------------------------------------------
# bucketing follows compilation
# ---------------------------------------------------------------------------


def test_backend_buckets_when_something_specializes_on_shape():
    """Bucketing follows compilation: on its own it is a small cost, and its
    only value is bounding the shape set for whatever compiles against it."""
    model = build_memory_model()
    backend = ModelBackend(model, tokenizer=None)
    backend._bucket_decode = True
    backend._compile_memory = False  # scope only; no Inductor in a unit test
    assert active_buckets() is None
    with backend.eval_mode():
        ladder = active_buckets()
        assert ladder, "eval_mode must open the scope when bucketing is on"
        cap = model.config.max_position_embeddings
        assert all(r <= cap for r in ladder), "rungs above the cap must be dropped"
    assert active_buckets() is None


def test_backend_leaves_lengths_alone_when_nothing_compiles():
    backend = ModelBackend(build_memory_model(), tokenizer=None)
    backend._bucket_decode = False
    with backend.eval_mode():
        assert active_buckets() is None
