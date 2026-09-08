"""Decode backends: the one thing that genuinely differs between
inference paths, factored out so a single :class:`Generator` owns the
request queue, tool-call loop, prompt handling, and sampling defaults.

A backend answers "extend this token sequence until the next halt token"
plus a little metadata (device, positional capacity, eval-mode context,
preferred sampling temperature). Everything else is shared.

There is one backend, :class:`ModelBackend`, and that is the point: it wraps
``model.generate``, so every halt in the contract below is transformers' own -
``eos_token_id`` becomes an ``EosTokenCriteria``, ``stop_strings`` a
``StopStringCriteria`` (the tokenizer is passed through for it), the request
deadline a ``MaxTimeCriteria``, and the positional cap a ``MaxLengthCriteria``.
Nothing here re-implements any of them.

Mono-Forward, whose weights live on Ray actors rather than in one module, used
to be a second backend carrying its own copies of all of that. It is now a
``PreTrainedModel`` face over the actor chain
(:class:`praxis.trainers.mono_forward.hf_model.MonoForwardLM`) handed to this
same backend, so the difference is confined to the forward.
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

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
        streamer: Optional[Any] = None,
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
        ran between calls would never fire.

        ``streamer`` is a ``transformers.generation.BaseStreamer``. It sees the
        prompt of every step plus each token as it is produced, so a consumer
        that wants only the model's own new text skips what it is handed before
        the first token of each step - see
        :class:`praxis.generation.streamers.ReplyStreamer`, which the
        ``Generator`` drives across the whole halt-and-resume turn."""


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
        """Where prompts have to live.

        ``PreTrainedModel`` answers this itself, and asking it rather than
        walking ``parameters()`` is what lets a model with no parameters of its
        own work - the Mono-Forward face holds none, because its weights are on
        Ray actors. The walk stays as the fallback for a bare module.
        """
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
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
                    probe = torch.zeros(
                        (1, length), dtype=torch.long, device=self.device
                    )
                    self.model(input_ids=probe)
        except Exception:
            # A warmup is an optimization. Never let it end a run.
            _log.debug(
                "Decode warmup failed; first request pays instead", exc_info=True
            )

    def generate_until_halt(
        self,
        tokens: torch.Tensor,
        step_kwargs: Dict[str, Any],
        deadline: Optional[float] = None,
        streamer: Optional[Any] = None,
    ) -> torch.Tensor:
        step_kwargs = dict(step_kwargs)
        if deadline is not None:
            # The deadline is ABSOLUTE wall-clock; `max_time` is a budget
            # measured from the moment `generate` builds its criteria. A turn
            # that halts and resumes calls this several times, so the budget is
            # recomputed per step rather than set once - otherwise each step
            # would restart the clock and get the full timeout again.
            #
            # Clamped at 0 rather than skipped when already expired: a
            # MaxTimeCriteria with a zero budget halts before the first token,
            # which is what an expired request should cost.
            step_kwargs["max_time"] = max(0.0, deadline - time.time())
        # On the config, not as a sibling kwarg: transformers deprecates mixing
        # a GenerationConfig with generation parameters passed alongside it.
        step_kwargs["return_dict_in_generate"] = True
        outputs = self.model.generate(
            tokens,
            generation_config=GenerationConfig(**step_kwargs),
            tokenizer=self.tokenizer,
            streamer=streamer,
        )
        return outputs.sequences
