"""Decode backends: the one thing that genuinely differs between
inference paths, factored out so a single :class:`Generator` owns the
request queue, tool-call loop, prompt handling, and sampling defaults.

A backend answers "extend this token sequence until the next halt token"
plus a little metadata (device, positional capacity, eval-mode context,
preferred sampling temperature). Everything else is shared.

- :class:`ModelBackend` wraps ``model.generate`` (halt-and-resume native:
  the boundary tokens sit in ``eos_token_id``, and text boundaries in
  ``stop_strings``, which transformers honours because the tokenizer is
  passed through).
- :class:`MonoForwardBackend` drives ``MonoForwardTrainer.generate``,
  whose streaming token iterator hops activations through Ray actors. It
  implements the same halt-and-resume contract by stopping its yield loop
  when a produced token lands in the stop set, or when the decoded tail
  completes a stop string.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

import torch
from transformers import GenerationConfig

from praxis.environments import EnvironmentFeatures

_log = logging.getLogger("praxis.generation")


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

    def warmup(self) -> None:
        """Pay any one-time decode setup now, before a caller is waiting.

        Default is a no-op. A backend that compiles on first use overrides it,
        because that cost otherwise lands inside somebody's request: the web
        chat gives up after 60s (``generate_from_messages``) and the deadline
        cannot interrupt a single forward, so a first request arriving during
        compilation comes back empty.
        """

    @abstractmethod
    def generate_until_halt(
        self,
        tokens: torch.Tensor,
        step_kwargs: Dict[str, Any],
        deadline: Optional[float] = None,
    ) -> torch.Tensor:
        """Extend ``tokens`` until a halt token (any id in
        ``step_kwargs['eos_token_id']``), a completed stop string (any of
        ``step_kwargs['stop_strings']``), ``max_new_tokens``, or ``deadline``.
        Return the full ``[1, L]`` sequence, including the halt token or
        boundary. Returns the input unchanged when nothing was produced.

        Only tokens produced by THIS call can halt it: the caller resumes from
        a sequence that already ends in the boundary it last halted on, so a
        backend that re-tested the whole sequence would return zero new tokens
        forever.

        ``deadline`` is wall-clock (``time.time()`` scale) and has to be honored
        PER STEP, not just on entry. The queued path decodes inside the training
        loop, and a plain turn is one call to this method, so a check that only
        ran between calls would never fire."""


class ModelBackend(DecodeBackend):
    """Standard in-process ``model.generate`` backend."""

    def __init__(self, model, tokenizer) -> None:
        # Decode on the UNCOMPILED module, deliberately. Whole-model compile is
        # ruinous here (see eval_mode), and today the Generator is handed
        # `bundle.model` while the LightningModule keeps the `try_compile`
        # wrapper, so this is already what happens - unwrapping just stops that
        # from being an accident of wiring that a future caller could undo.
        self.model = getattr(model, "_orig_mod", model)
        self.tokenizer = tokenizer
        # OFF BY DEFAULT, and the reason is a measured regression rather than
        # caution. Compiling the decode-time memory bodies is worth 1.43x on a
        # turn, but with static shapes it recompiles as the rolling context
        # GROWS: the terminal generates every `infer_every` seconds from a
        # buffer that gains bytes each time, and every crossing of a patch
        # boundary (patch_size 8) is a new trunk length, a new graph, and
        # another wake-up for Inductor's 8-worker compile pool. Measured across
        # two runs of the same model, child-process memory:
        #
        #     abstractinator-t (off)  mean  599MB,  4% of samples over 1GB
        #     abstractinator-u (on)   mean 2839MB, 62% of samples over 1GB
        #
        # -u died at its first validation with swap exhausted, ~2h in, where -t
        # had run 14h on the same host. The benchmark that justified this ran a
        # FIXED prompt length, which is exactly the case that never recompiles,
        # so it measured wall clock and missed the cost entirely.
        #
        # Re-enable per environment once the shape set is bounded (symbolic
        # shapes, or bucketing the decode length), and measure host RSS and
        # child memory over a GROWING context, not a fixed one.
        cfg = getattr(self.model, "config", None)
        self._compile_memory = EnvironmentFeatures.is_enabled(
            "compile_decode_memory"
        ) and not bool(getattr(cfg, "no_compile", False))

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
        """Scope a generation: eval mode, no whole-model compile, compiled
        NeuralMemory.

        WHOLE-MODEL COMPILE STAYS OFF, and the original reason for that holds
        up: the recurrent loop passes ``current_depth`` as a python int and KL
        halting varies the loop count per input, so Dynamo re-traces on nearly
        every call. Measured on abstractinator-t, compiling the decoder made a
        136-byte forward 21288 ms against 202 ms eager - a hundred times slower,
        with 143 compiled frames and no sign of settling.

        THE ``force_eager`` STANCE IS GONE, and it has to be: the stance is
        global, so it forces ANY compiled callable entered inside the window
        back to eager - including the one this method now installs on purpose.
        With it in place the compiled memory measured 1.01x, i.e. exactly
        nothing. What keeps the trunk eager instead is the constructor
        unwrapping ``_orig_mod``, which is narrow enough to name the thing it
        is preventing.

        A second, independent blocker is worth recording so nobody re-attempts
        it blind: flex attention's Triton template needs power-of-two block
        shapes, and ``head_size: 37`` is not one, so on abstractinator-t the
        decoder fails to compile at ANY sequence length with `Shape element 2
        must be a power of 2`. Training never trips it because the packer
        supplies ``block_ids``, which routes to the materialized
        ``_local_attention_blocked`` path instead of flex.

        What DOES pay is compiling the one module that is ~59% of the forward's
        dispatch count and has stable shapes - see ``decode_compiled``.
        """
        from praxis.memory.neural_memory import decode_compiled

        training = self.model.training
        self.model.eval()
        try:
            with decode_compiled(self.model, enabled=self._compile_memory):
                yield
        finally:
            self.model.train(training)

    # Probe lengths for warmup, spread over the range a rolling context and a
    # chat turn actually occupy. Cheap to add to (each is one short forward);
    # the cost that matters is Inductor's, and that is per distinct shape.
    WARMUP_LENGTHS = (8, 64, 128, 256, 512, 1024)

    def warmup(self) -> None:
        """Compile the decode-time memory bodies on throwaway forwards.

        Cold, this is minutes of Inductor; Torch's on-disk graph cache makes
        every later run of the same config far cheaper.

        A LADDER of lengths, because the bodies compile with static shapes and
        a turn walks a range of them (see ``_DECODE_COMPILE_KWARGS`` for why
        symbolic shapes lost). The ladder is what makes this worth doing: a
        single-length probe left the first real turn at 55s against 44s eager,
        still tracing its way up, while the ladder lands it at 30s - steady
        state from the very first request. Anything past ``max_positions`` is
        skipped rather than clamped, since a probe the model cannot represent
        is not a warm graph.
        """
        if not self._compile_memory:
            return
        cap = self.max_positions
        try:
            with self.eval_mode(), torch.no_grad():
                for length in self.WARMUP_LENGTHS:
                    if cap is not None and length > cap:
                        break
                    probe = torch.zeros((1, length), dtype=torch.long, device=self.device)
                    self.model(input_ids=probe)
        except Exception:
            # A warmup is an optimization. Never let it end a run.
            _log.debug("Decode warmup failed; first request pays instead", exc_info=True)

    def generate_until_halt(
        self,
        tokens: torch.Tensor,
        step_kwargs: Dict[str, Any],
        deadline: Optional[float] = None,
    ) -> torch.Tensor:
        from praxis.generation.stopping import deadline_criteria

        extra = {}
        criteria = deadline_criteria(deadline)
        if criteria is not None:
            # Honored by the transformers loop natively, and by our own
            # speculative loop explicitly (see _speculative_generate).
            extra["stopping_criteria"] = criteria
        outputs = self.model.generate(
            tokens,
            generation_config=GenerationConfig(**step_kwargs),
            tokenizer=self.tokenizer,
            return_dict_in_generate=True,
            **extra,
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
        self,
        tokens: torch.Tensor,
        step_kwargs: Dict[str, Any],
        deadline: Optional[float] = None,
    ) -> torch.Tensor:
        import time

        from praxis.generation.stopping import find_stop_cut, normalize_stop_strings

        stop_ids = self._stop_ids(step_kwargs)
        stop_strings = normalize_stop_strings(step_kwargs.get("stop_strings"))
        max_new_tokens = int(step_kwargs.get("max_new_tokens", 100))
        top_k = step_kwargs.get("top_k")
        prefix = tokens[0].tolist()
        produced = []
        for tok in self.trainer.generate(
            tokens.cpu(),
            max_new_tokens=max_new_tokens,
            eos_token_id=None,  # we own halting against the full stop set
            do_sample=bool(step_kwargs.get("do_sample", True)),
            temperature=float(step_kwargs.get("temperature", 1.0)),
            top_k=int(top_k) if top_k else None,
            # This loop owns its sampling, so transformers never builds a
            # processor list for it - the format's unproducible control ids
            # have to be masked explicitly, exactly as the speculative path
            # does (praxis/modeling.py::_speculative_generate).
            suppress_tokens=step_kwargs.get("suppress_tokens"),
        ):
            produced.append(tok)
            if deadline is not None and time.time() >= deadline:
                break
            if int(tok.view(-1)[0].item()) in stop_ids:
                break
            # Text-boundary halt. The scan starts at the prompt's length so the
            # boundary we resumed from cannot re-halt this step; only one new
            # token arrives per iteration, so the earliest completion is here.
            if stop_strings:
                ids = prefix + [int(t.view(-1)[0].item()) for t in produced]
                keep = find_stop_cut(self.tokenizer, ids, len(ids) - 1, stop_strings)
                if keep is not None:
                    break
        if not produced:
            return tokens
        new_ids = torch.stack(produced, dim=-1).to(tokens.device)
        return torch.cat([tokens, new_ids], dim=-1)
