import contextlib
import time

import pytest
import torch
from transformers import GenerationConfig

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.environments import EnvironmentFeatures
from praxis.generation.bucketing import active_buckets
from praxis.generation.decode_backend import ModelBackend
from praxis.memory.neural_memory import NeuralMemory
from praxis.modeling import PraxisForCausalLM

# ------------------------------------------------------------------------------
# decode_compile
# ------------------------------------------------------------------------------
# Decode-time compilation of NeuralMemory: plumbing, scoping, and fallback.
#
# The measured payoff (1.7x on a 128-byte generation, byte-identical output) needs a GPU
# and several minutes of Inductor, so it is not asserted here. What IS asserted is
# everything that could silently break it or, worse, leak a compiled body into training:
# installation, dispatch, restoration, the ``no_compile`` gate, and degrading to eager
# when compilation raises.
#
# ``torch.compile`` is stubbed throughout - compiling for real would make this test
# minutes long and would test Inductor rather than this wiring.


def build_memory_model(**overrides):
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


def neural_memories(model):
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


def test_eval_mode_does_not_force_eager(feature_on, stub_compile):
    """The global ``force_eager`` stance would defeat the compiled body it
    installs - measured at 1.01x, i.e. nothing. If a stance ever comes back,
    this fails."""
    model = build_memory_model()
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


def test_warmup_never_raises(monkeypatch):
    """A warmup is an optimization; it must not be able to end a run."""
    model = build_memory_model()
    backend = ModelBackend(model, tokenizer=None)

    def explode(*a, **k):
        raise RuntimeError("cuda is having a day")

    monkeypatch.setattr(type(model), "forward", explode)
    backend.warmup()  # must not raise


def test_decode_compile_is_off_by_default():
    """A rolling context grows, so static-shape decode graphs multiply without
    bound and keep Inductor's worker pool resident - measured at 2839MB mean
    child memory against 599MB with this off, and a swap-exhaustion kill two
    hours into abstractinator-u. Off unless an environment asks for it."""
    EnvironmentFeatures.clear()
    assert ModelBackend(build_memory_model(), tokenizer=None)._compile_memory is False


# ------------------------------------------------------------------------------
# generation_deadline
# ------------------------------------------------------------------------------
# The request deadline, which is what bounds a chat request's cost to the run.
#
# Queued generations are served by ``GenerationQueueCallback`` from inside
# ``on_train_batch_end``, so they hold the training loop's turn. The wait in
# ``generate_from_messages`` is client-side only: before the deadline existed, giving up
# after 60s stopped us listening but did not stop the loop from decoding the whole turn.
# Measured on ``abstractinator-r``, where the encoder stack cannot cache and every
# decode step is a full forward: one 512-byte Discord turn stalled training for 208
# seconds, ~148 of them after the client had already timed out and thrown the eventual
# reply away.
#
# Two things have to hold, and neither is visible from the caller's side:
#
# - a request that expires BEFORE it is served must never run, - a request served just
# under the wire must stop decoding when it expires, rather than running to
# ``max_new_tokens``.


# ---------------------------------------------------------------------------
# the wiring itself: an absolute deadline becomes a per-step relative budget
# ---------------------------------------------------------------------------


class _CapturingModel:
    """Records the GenerationConfig ``ModelBackend`` builds, then no-ops."""

    def __init__(self):
        self.configs = []
        self.config = None

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, tokens, generation_config=None, **kwargs):
        from types import SimpleNamespace

        self.configs.append(generation_config)
        return SimpleNamespace(sequences=tokens)


def test_the_deadline_reaches_transformers_as_max_time(tokenizer):
    """``DeadlineCriteria`` is gone; ``MaxTimeCriteria`` was always the same
    class, and it is the one transformers evaluates inside its own loop.

    The translation matters: the request carries an ABSOLUTE wall-clock
    deadline, while ``max_time`` is a budget measured from the moment
    ``generate`` builds its criteria. Recomputing it per call is what stops a
    turn that halts and resumes from handing each step the full timeout again.
    """
    from praxis.generation.decode_backend import ModelBackend

    model = _CapturingModel()
    backend = ModelBackend(model, tokenizer)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    backend.generate_until_halt(ids, {"max_new_tokens": 4}, deadline=time.time() + 5.0)
    assert 4.0 < model.configs[-1].max_time <= 5.0

    # Second step of the same turn, a moment later: a SMALLER budget, not a
    # fresh 5 seconds.
    time.sleep(0.05)
    backend.generate_until_halt(ids, {"max_new_tokens": 4}, deadline=time.time() + 0.5)
    assert 0.0 < model.configs[-1].max_time <= 0.5


def test_an_already_expired_deadline_becomes_a_zero_budget(tokenizer):
    """Clamped rather than skipped: a zero budget halts before the first token,
    which is what an expired request should cost. Passing a negative through
    would be silently ignored by MaxTimeCriteria's `elapsed > max_time`."""
    from praxis.generation.decode_backend import ModelBackend

    model = _CapturingModel()
    backend = ModelBackend(model, tokenizer)
    backend.generate_until_halt(
        torch.tensor([[1]], dtype=torch.long),
        {"max_new_tokens": 4},
        deadline=time.time() - 10.0,
    )
    assert model.configs[-1].max_time == 0.0


def test_no_deadline_leaves_max_time_unset(tokenizer):
    """``/input`` polls forever and passes no deadline, so no time criterion
    should be built at all."""
    from praxis.generation.decode_backend import ModelBackend

    model = _CapturingModel()
    backend = ModelBackend(model, tokenizer)
    backend.generate_until_halt(torch.tensor([[1]], dtype=torch.long), {})
    assert model.configs[-1].max_time is None


# ------------------------------------------------------------------------------
# decode_bucketing
# ------------------------------------------------------------------------------
# Decode-length bucketing: the ladder, the padding, and what it must not change.
#
# The point of bucketing is that a turn stops minting a new shape per step, which is
# what makes anything shape-sensitive (compiled decode bodies today) safe to run over a
# growing context. It is only allowed to buy that if the model writes the same bytes, so
# most of this file is about equivalence rather than shapes.


# ---------------------------------------------------------------------------
# equivalence on a real byte-latent model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def byte_model():
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        decoder_type="sequential",
        head_type="forward",
        encoder_type="abstractinator_v1",
        tokenizer_type="byte_level",
        byte_offset=0,
        byte_vocab_size=256,
        codebook_size=256,
        max_position_embeddings=1024,
        mtp_depth=3,
        mtp_type="per_depth",
    )
    torch.manual_seed(0)
    return PraxisForCausalLM(cfg).eval()


def test_backend_buckets_when_something_specializes_on_shape(byte_model):
    """Bucketing follows compilation: on its own it is a small cost, and its
    only value is bounding the shape set for whatever compiles against it."""
    from praxis.generation import ModelBackend

    backend = ModelBackend(byte_model, tokenizer=None)
    backend._bucket_decode = True
    backend._compile_memory = False  # scope only; no Inductor in a unit test
    assert active_buckets() is None
    with backend.eval_mode():
        ladder = active_buckets()
        assert ladder, "eval_mode must open the scope when bucketing is on"
        cap = byte_model.config.max_position_embeddings
        assert all(r <= cap for r in ladder), "rungs above the cap must be dropped"
    assert active_buckets() is None


def test_backend_leaves_lengths_alone_when_nothing_compiles(byte_model):
    from praxis.generation import ModelBackend

    backend = ModelBackend(byte_model, tokenizer=None)
    backend._bucket_decode = False
    with backend.eval_mode():
        assert active_buckets() is None
