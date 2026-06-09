"""Decode backends: the one thing that genuinely differs between
inference paths, factored out so a single :class:`Generator` owns the
request queue, tool-call loop, prompt handling, and sampling defaults.

A backend answers "extend this token sequence until the next halt token"
plus a little metadata (device, positional capacity, eval-mode context,
preferred sampling temperature). Everything else is shared.

- :class:`ModelBackend` wraps ``model.generate`` (halt-and-resume native:
  the boundary tokens sit in ``eos_token_id``).
- :class:`MonoForwardBackend` drives ``MonoForwardTrainer.generate``,
  whose streaming token iterator hops activations through Ray actors. It
  implements the same halt-and-resume contract by stopping its yield loop
  when a produced token lands in the stop set.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import torch
from transformers import GenerationConfig


class DecodeBackend(ABC):
    """How next-tokens are produced, plus the model-shaped metadata the
    shared Generator needs."""

    tokenizer: Any
    device: Any

    @property
    def default_sampling_temperature(self) -> Optional[float]:
        """Preferred temperature when the caller omits one (None = no
        preference; the transformers default applies)."""
        return None

    @property
    def max_positions(self) -> Optional[int]:
        """Positional capacity; the context must never exceed it."""
        return None

    @contextlib.contextmanager
    def eval_mode(self):
        """Scope generation in inference mode. Default is a no-op."""
        yield

    @abstractmethod
    def generate_until_halt(
        self, tokens: torch.Tensor, step_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Extend ``tokens`` until a halt token (any id in
        ``step_kwargs['eos_token_id']``) or ``max_new_tokens``. Return the
        full ``[1, L]`` sequence, including the halt token. Returns the
        input unchanged when nothing was produced."""


class ModelBackend(DecodeBackend):
    """Standard in-process ``model.generate`` backend."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def default_sampling_temperature(self) -> Optional[float]:
        return getattr(self.model, "default_sampling_temperature", None)

    @property
    def max_positions(self) -> Optional[int]:
        cfg = getattr(self.model, "config", None)
        mpe = getattr(cfg, "max_position_embeddings", None) if cfg else None
        return int(mpe) if mpe else None

    @contextlib.contextmanager
    def eval_mode(self):
        # Generation must not run through torch.compile: the decode loop
        # changes python-level guard inputs every token (cache pos, live
        # segment length, current_depth), so compiled frames blow the
        # dynamo recompile limit; eager decode is cheap with the KV cache.
        training = self.model.training
        self.model.eval()
        stance = contextlib.ExitStack()
        try:
            try:
                stance.enter_context(torch.compiler.set_stance("force_eager"))
            except Exception:
                pass  # older torch: stance API unavailable, run as-is
            yield
        finally:
            stance.close()
            self.model.train(training)

    def generate_until_halt(
        self, tokens: torch.Tensor, step_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        outputs = self.model.generate(
            tokens,
            generation_config=GenerationConfig(**step_kwargs),
            tokenizer=self.tokenizer,
            return_dict_in_generate=True,
        )
        return outputs.sequences


class MonoForwardBackend(DecodeBackend):
    """Routes decoding through ``MonoForwardTrainer.generate`` (Ray actor
    chain). The trainer yields one token at a time; we accumulate until a
    halt token to honour the shared halt-and-resume tool loop."""

    def __init__(self, trainer, tokenizer, default_temperature: float = 0.5) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.device = "cpu"  # actors run CPU-only; prompts must live on CPU
        self._default_temperature = default_temperature

    @property
    def default_sampling_temperature(self) -> Optional[float]:
        return self._default_temperature

    @staticmethod
    def _stop_ids(step_kwargs: Dict[str, Any]) -> Set[int]:
        eos = step_kwargs.get("eos_token_id")
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(x) for x in eos}
        return {int(eos)}

    def generate_until_halt(
        self, tokens: torch.Tensor, step_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        stop_ids = self._stop_ids(step_kwargs)
        max_new_tokens = int(step_kwargs.get("max_new_tokens", 100))
        top_k = step_kwargs.get("top_k")
        produced = []
        for tok in self.trainer.generate(
            tokens.cpu(),
            max_new_tokens=max_new_tokens,
            eos_token_id=None,  # we own halting against the full stop set
            do_sample=bool(step_kwargs.get("do_sample", True)),
            temperature=float(step_kwargs.get("temperature", 1.0)),
            top_k=int(top_k) if top_k else None,
        ):
            produced.append(tok)
            if int(tok.view(-1)[0].item()) in stop_ids:
                break
        if not produced:
            return tokens
        new_ids = torch.stack(produced, dim=-1).to(tokens.device)
        return torch.cat([tokens, new_ids], dim=-1)
