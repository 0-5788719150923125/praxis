"""Decode-time compilation of NeuralMemory: plumbing, scoping, and fallback.

The measured payoff (1.7x on a 128-byte generation, byte-identical output)
needs a GPU and several minutes of Inductor, so it is not asserted here. What
IS asserted is everything that could silently break it or, worse, leak a
compiled body into training: installation, dispatch, restoration, the
``no_compile`` gate, and degrading to eager when compilation raises.

``torch.compile`` is stubbed throughout - compiling for real would make this
test minutes long and would test Inductor rather than this wiring.
"""

import contextlib

import pytest
import torch

from praxis.environments import EnvironmentFeatures

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.generation.decode_backend import ModelBackend
from praxis.memory.neural_memory import NeuralMemory, decode_compiled


def build_model(**overrides):
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        attention_type="causal",
        encoding="rope",
        memory_type="mal_energy",
        **overrides,
    )
    return PraxisForCausalLM(cfg).eval()


def memories(model):
    return [m for m in model.modules() if isinstance(m, NeuralMemory)]


@pytest.fixture
def feature_on():
    """The decode compile is opt-in per environment; most tests want it on."""
    EnvironmentFeatures.set_from_environment({"compile_decode_memory": True})
    try:
        yield
    finally:
        EnvironmentFeatures.clear()


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


def test_model_has_neural_memory():
    """Guard the fixture itself: a config that quietly built the no-op memory
    would make every other test in this file vacuously pass."""
    assert memories(build_model())


def test_installs_inside_and_restores_outside(stub_compile):
    model = build_model()
    mems = memories(model)
    assert all(m._decode_forward is None for m in mems)

    with decode_compiled(model):
        assert all(m._decode_forward is not None for m in mems)

    assert all(m._decode_forward is None for m in mems)
    assert stub_compile["n"] == len(mems)


def test_compiled_body_is_cached_across_windows(stub_compile):
    """Only the first generation of a run may pay for Inductor."""
    model = build_model()
    for _ in range(3):
        with decode_compiled(model):
            pass
    assert stub_compile["n"] == len(memories(model))


def test_restores_when_the_body_raises(stub_compile):
    model = build_model()
    with pytest.raises(RuntimeError):
        with decode_compiled(model):
            raise RuntimeError("boom")
    assert all(m._decode_forward is None for m in memories(model))


def test_disabled_is_a_no_op(stub_compile):
    model = build_model()
    with decode_compiled(model, enabled=False):
        assert all(m._decode_forward is None for m in memories(model))
    assert stub_compile["n"] == 0


def test_compile_failure_degrades_to_eager(monkeypatch):
    def explode(fn, **kwargs):
        raise RuntimeError("inductor said no")

    monkeypatch.setattr(torch, "compile", explode)
    model = build_model()
    ids = torch.randint(0, 200, (1, 8))
    with decode_compiled(model):
        assert all(m._decode_forward is None for m in memories(model))
        with torch.no_grad():
            model(input_ids=ids)  # still runs, on the eager body


def test_forward_dispatches_to_the_installed_body(stub_compile):
    model = build_model()
    ids = torch.randint(0, 200, (1, 8))
    with torch.no_grad():
        with decode_compiled(model):
            model(input_ids=ids)
    assert stub_compile.get("invoked", 0) > 0


def test_output_is_unchanged_by_the_dispatch_hop(stub_compile):
    """The dispatcher must be a pure hop. A real compile is byte-identical too
    (verified on abstractinator-t); this pins the plumbing that surrounds it."""
    model = build_model()
    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        eager = model(input_ids=ids).logits
        with decode_compiled(model):
            hopped = model(input_ids=ids).logits
    assert torch.equal(eager, hopped)


def test_backend_honors_no_compile(feature_on):
    assert ModelBackend(build_model(), tokenizer=None)._compile_memory is True
    off = ModelBackend(build_model(no_compile=True), tokenizer=None)
    assert off._compile_memory is False


def test_backend_unwraps_a_compiled_model():
    """Whole-model compile is ruinous at decode, so the backend must decode on
    the original module even when handed a wrapper."""
    model = build_model()

    class FakeOptimizedModule:
        def __init__(self, mod):
            self._orig_mod = mod

    backend = ModelBackend(FakeOptimizedModule(model), tokenizer=None)
    assert backend.model is model


def test_eval_mode_restores_training_flag(feature_on, stub_compile):
    model = build_model()
    model.train()
    backend = ModelBackend(model, tokenizer=None)
    with backend.eval_mode():
        assert not model.training
        assert all(m._decode_forward is not None for m in memories(model))
    assert model.training
    assert all(m._decode_forward is None for m in memories(model))


def test_eval_mode_does_not_force_eager(feature_on, stub_compile):
    """The global ``force_eager`` stance would defeat the compiled body it
    installs - measured at 1.01x, i.e. nothing. If a stance ever comes back,
    this fails."""
    model = build_model()
    backend = ModelBackend(model, tokenizer=None)
    seen = []
    real = torch.compiler.set_stance

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return contextlib.nullcontext()

    torch.compiler.set_stance = spy
    try:
        with backend.eval_mode():
            pass
    finally:
        torch.compiler.set_stance = real
    assert seen == []


def test_warmup_compiles_before_any_request(feature_on, stub_compile):
    """The point of warmup: after it, no caller pays for Inductor."""
    model = build_model()
    model.train()
    backend = ModelBackend(model, tokenizer=None)
    backend.warmup()
    assert stub_compile["n"] == len(memories(model))
    assert stub_compile.get("invoked", 0) > 0
    # ... and it handed the model back exactly as it found it.
    assert all(m._decode_forward is None for m in memories(model))
    assert model.training


def test_warmup_is_a_no_op_when_compilation_is_off(feature_on, stub_compile):
    backend = ModelBackend(build_model(no_compile=True), tokenizer=None)
    backend.warmup()
    assert stub_compile["n"] == 0


def test_warmup_never_raises(monkeypatch):
    """A warmup is an optimization; it must not be able to end a run."""
    model = build_model()
    backend = ModelBackend(model, tokenizer=None)

    def explode(*a, **k):
        raise RuntimeError("cuda is having a day")

    monkeypatch.setattr(type(model), "forward", explode)
    backend.warmup()  # must not raise


def test_queue_callback_warms_the_backend(feature_on, stub_compile):
    from praxis.callbacks.lightning.generation_queue import GenerationQueueCallback

    class FakeGenerator:
        def __init__(self, backend):
            self.backend = backend

    model = build_model()
    backend = ModelBackend(model, tokenizer=None)
    GenerationQueueCallback(FakeGenerator(backend)).on_train_start(None, None)
    assert stub_compile["n"] == len(memories(model))


def test_queue_callback_tolerates_a_backend_without_warmup():
    from praxis.callbacks.lightning.generation_queue import GenerationQueueCallback

    class Bare:
        backend = object()

    GenerationQueueCallback(Bare()).on_train_start(None, None)


def test_decode_compile_is_off_by_default():
    """A rolling context grows, so static-shape decode graphs multiply without
    bound and keep Inductor's worker pool resident - measured at 2839MB mean
    child memory against 599MB with this off, and a swap-exhaustion kill two
    hours into abstractinator-u. Off unless an environment asks for it."""
    EnvironmentFeatures.clear()
    assert ModelBackend(build_model(), tokenizer=None)._compile_memory is False


def test_feature_alone_does_not_override_no_compile(feature_on):
    off = ModelBackend(build_model(no_compile=True), tokenizer=None)
    assert off._compile_memory is False
