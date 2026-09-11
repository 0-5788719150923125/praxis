"""Stubs and small builders shared across test namespaces.

Import them by name (``from tests.stubs import Cfg, Enc``). Fixtures belong in a
conftest.py instead; a stub used by one namespace only belongs in that
namespace's own files.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.inference.decoding import first_halt
from praxis.memory.neural_memory import NeuralMemory
from praxis.tokenizers import create_tokenizer
from praxis.trainers.mono_forward import MonoForwardTrainer  # noqa: F401 (re-exported)

# ------------------------------------------------------------------------------
# models and configs
# ------------------------------------------------------------------------------


def build_memory_model(**overrides):
    """A tiny eval-mode transformer carrying a ``mal_energy`` NeuralMemory."""
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


def materialize(cls, x, **kwargs):
    """``cls(**kwargs)`` after one forward on ``x``, which sizes lazy parameters."""
    module = cls(**kwargs)
    module(x)
    return module


def neural_memories(model):
    return [m for m in model.modules() if isinstance(m, NeuralMemory)]


class Cfg:
    """The config fields a classifier and its criterion read, at toy sizes."""

    hidden_size = 48
    embed_size = 48
    vocab_size = 32
    loss_func = "cross_entropy"
    tie_word_embeddings = False
    crystal_n = None
    crystal_label_smoothing = 0.0
    embedding_rms_lambda = 0.0
    causal = True
    debug = False


class Enc(nn.Module):
    """Minimal encoder declaring an output layout."""

    def __init__(self, d=48, v=32):
        super().__init__()
        self.output_dim = d
        self.output_vocab_size = v


# ------------------------------------------------------------------------------
# Mono-Forward
# ------------------------------------------------------------------------------

try:
    import ray  # noqa: F401

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

requires_ray = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")


def _mf_config(num_layers: int = 4) -> PraxisConfig:
    """Tiny CPU-only Mono-Forward config.

    The pipeline-overlap assertion needs at least 4 layers for
    ``pipeline_in_flight_max >= num_layers - 1`` to prove anything; the math
    and single-fit tests run faster at depth 2.
    """
    return PraxisConfig(
        vocab_size=256,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=num_layers,
        num_layers=num_layers,
        max_length=64,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )


class _FixedBatchDataset(IterableDataset):
    """Yields the same batch forever: a memorization workload whose loss must fall."""

    def __init__(self, vocab_size: int, batch_size: int, seq_len: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.batch = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)

    def __iter__(self):
        while True:
            yield {"input_ids": self.batch}


class _SyntheticDataModule:
    """The one DataModule method ``MonoForwardTrainer.fit`` calls."""

    def __init__(self, dataset: _FixedBatchDataset) -> None:
        self._dataset = dataset

    def train_dataloader(self):
        return iter(self._dataset)


class _ToyTokenizer:
    """Character-level tokenizer covering the surface ``Generator`` touches.

    ``encode``, ``decode``, ``apply_chat_template`` and the named-token lookup
    ``ChatFormat.suppressed_token_ids`` needs, without the training-time
    tokenization machinery.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.sep_token_id = 3
        self.bos_token = "<s>"
        self.eos_token = "</s>"

    def convert_tokens_to_ids(self, token):
        """Named control tokens only; anything else is unknown (None)."""
        named = {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        if isinstance(token, (list, tuple)):
            return [named.get(t) for t in token]
        return named.get(token)

    def encode(self, text: str) -> list:
        if not text:
            return [self.bos_token_id]
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list, skip_special_tokens: bool = False) -> str:
        special = {0, 1, 2, 3}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(chr(int(i) % 128))
        return "".join(chars)

    def apply_chat_template(
        self, messages: list, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        return "\n".join(parts)


class _StubTrainer:
    """The whole surface :class:`MonoForwardLM` needs: a config and one forward.

    Mono-Forward differs from in-process decoding only in where the forward
    runs, so everything above it (sampling, halting, the request queue) is the
    ordinary path and needs no stub.
    """

    def __init__(self, token: int = 65, num_layers: int = 2):
        self._config = _mf_config(num_layers=num_layers)
        self.token = token
        self.calls = 0

    def infer_logits(self, input_ids):
        self.calls += 1
        b, t = input_ids.shape
        logits = torch.full((b, t, self._config.vocab_size), -10.0)
        logits[:, :, self.token] = 10.0
        return logits


def _stub_generator(trainer=None):
    from praxis.inference import MonoForwardGenerator

    return MonoForwardGenerator(
        trainer=trainer or _StubTrainer(), tokenizer=_ToyTokenizer()
    )


# ------------------------------------------------------------------------------
# generation backends
# ------------------------------------------------------------------------------


class _ScriptedBackend:
    """Emits a fixed byte script, halting exactly as the real decode loops do.

    Drives the real ``Generator``, so a test exercises the whole endpoint path
    - prompt construction, the halt contract, the tool state machine, and reply
    extraction - with the model's output pinned instead of sampled.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.max_positions = None
        self.pending = list(tokenizer.encode(script, add_special_tokens=False))

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def _criteria(self, step_kwargs):
        """The criteria transformers would build for these kwargs."""
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
        )

        criteria = StoppingCriteriaList()
        stops = step_kwargs.get("stop_strings")
        if stops:
            criteria.append(
                StopStringCriteria(stop_strings=list(stops), tokenizer=self.tokenizer)
            )
        eos = step_kwargs.get("eos_token_id")
        if eos:
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            criteria.append(EosTokenCriteria(eos_token_id=torch.tensor(eos)))
        return criteria

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        criteria = self._criteria(step_kwargs)
        budget = int(step_kwargs.get("max_new_tokens", 100))
        start = tokens.shape[1]
        if streamer is not None:
            streamer.put(tokens)
        ids = tokens[0].tolist()
        produced = 0
        while self.pending and produced < budget:
            nxt = self.pending.pop(0)
            ids.append(nxt)
            produced += 1
            if streamer is not None:
                streamer.put(torch.tensor([nxt]))
            if first_halt(torch.tensor([ids]), criteria, start) is not None:
                break
        if streamer is not None:
            streamer.end()
        return torch.tensor([ids], dtype=torch.long)


class _SlowBackend:
    """Spends ``delay`` seconds per token and never halts on a boundary.

    Never halting isolates the request deadline as the only thing that can end
    the decode. The per-token deadline check mirrors the ``MaxTimeCriteria``
    transformers builds from ``GenerationConfig.max_time`` for the real backend.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, delay=0.01):
        self.delay = delay
        self.device = "cpu"
        self.max_positions = None
        self.calls = 0
        self.tokens_emitted = 0

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        self.calls += 1
        budget = int(step_kwargs.get("max_new_tokens", 100))
        for _ in range(budget):
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.delay)
            # One ordinary byte ('a'), which is not a boundary under prose.
            nxt = torch.tensor([[ord("a")]], dtype=torch.long)
            tokens = torch.cat([tokens, nxt], dim=-1)
            self.tokens_emitted += 1
        return tokens


class MockGenerator:
    """Records generation requests and answers each with a canned reply."""

    def __init__(self):
        self.model = Mock()
        self.request_counter = 0
        self.last_deadline = None

    def request_generation(self, prompt: str, kwargs: dict, deadline=None, **_) -> str:
        """``**_`` absorbs the streaming callbacks (``on_text``/``on_reset``) the
        real ``Generator`` takes, so interface additions don't break the double."""
        self.request_counter += 1
        self.last_deadline = deadline
        return f"request_{self.request_counter}"

    def get_result(self, request_id: str) -> str:
        if "request_" in request_id:
            return f"Generated response for {request_id}"
        return None


# ------------------------------------------------------------------------------
# tokenizers
# ------------------------------------------------------------------------------


def tokenizer_for(chat_format):
    """The byte-level tokenizer under ``chat_format``."""
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format=chat_format
    )


class _ForeignTokenizer:
    """The shape of an off-the-hub (ChatML) tokenizer, minus everything irrelevant."""

    chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}<|im_end|>{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    eos_token = "<|im_end|>"
    eos_token_id = 7
    bos_token = "<|begin|>"
    bos_token_id = 1
    sep_token = None
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<|im_end|>" if i == 7 else "?" for i in ids)


class MockTokenizer:
    """Special-token strings plus a ``[BOS]role`` chat template."""

    def __init__(self):
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        result = ""
        for msg in messages:
            result += f"{self.bos_token}{msg.get('role', 'user')}\n{msg.get('content', '')}{self.sep_token}\n"
        if add_generation_prompt:
            result += f"{self.bos_token}assistant\n"
        return result


class _ToolTokenizer:
    """Renders every conversation as one fixed user turn awaiting the assistant."""

    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def apply_chat_template(self, messages, **kwargs):
        return "[BOS]user\nhi[SEP]\n[BOS]assistant\n"

    def convert_tokens_to_ids(self, token):
        return None


class _FakeTok:
    """Renders every conversation as a fixed system turn."""

    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "[BOS]system\n..."


# ------------------------------------------------------------------------------
# terminal
# ------------------------------------------------------------------------------


class _Tty(io.StringIO):
    """A stand-in terminal that records everything written to it."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.chunks = []

    def write(self, s):
        with self.lock:
            self.chunks.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    @property
    def text(self):
        with self.lock:
            return "".join(self.chunks)
