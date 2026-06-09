"""Streaming-generation context with degeneracy detection.

Shared utility for trainers that emit incremental decoded text during
training. Both the Lightning ``TerminalInterface`` callback (backprop
path) and the Ray Mono-Forward ``_maybe_run_inference_hook`` want the
same loop:

    1. Start from a seed (``initial_text``, typically the bos token).
    2. Every inference opportunity, generate a few new tokens,
       decode, and *append* to a running text buffer rather than
       starting fresh - so the user sees a continuously growing
       passage evolving with the model.
    3. Detect when the context goes degenerate (stuck output,
       n-gram repetition, sequential repetition, bracket-pipe
       pattern, all-whitespace) and reset to the seed, rather than
       letting the buffer lock into a pathological loop.

Both trainers now use this shared helper: the Lightning
``TerminalInterface`` callback (backprop path) and the Ray
Mono-Forward ``_maybe_run_inference_hook`` delegate stuck-output
tracking and degeneracy detection to :class:`StreamingContext`.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Callable, List, Optional, Sequence, Union

# Predictable, always-decodable seed pool: letters, digits, punctuation.
# Sampling real characters (rather than random vocab tokens) avoids the
# replacement-character (U+FFFD) seeds that byte-latent and similar
# tokenizers emit for unmapped IDs.
SEED_CHARS = string.ascii_letters + string.digits + string.punctuation


def random_char_seed() -> str:
    """Return a single random printable character to seed a buffer.

    The model must attend to *something* at position 0; a fixed seed
    imprints that choice on every sample. Re-rolling a real character
    spreads the starting condition without ever producing an invalid
    (un-decodable) seed. Usable directly as ``initial_text`` so each
    ``reset()`` draws a fresh character. Shared by every streaming path
    (Lightning backprop callback and Ray mono-forward hook).
    """
    return random.choice(SEED_CHARS)


class StreamingContext:
    """Ongoing text buffer with reset-on-degeneracy semantics.

    Caller pattern:

        ctx = StreamingContext()  # seeds with a random printable char
        # ... in inference hook:
        new_text = decode(encode(ctx.text) + generated_tokens)
        ctx.update(new_text)  # returns True if a reset fired
        print(ctx.text)

    The buffer is character-length capped at ``max_length`` - once it
    exceeds that, the oldest characters are dropped. This keeps the
    prompt-encode cost bounded even if the model produces a clean
    passage for hours.

    ``initial_text`` may be either a fixed string or a zero-arg callable
    that returns one. Callable seeds are invoked on every ``reset()``,
    which lets callers re-roll a fresh first character each time the
    buffer goes degenerate without re-instantiating the context. It
    defaults to :func:`random_char_seed`.
    """

    def __init__(
        self,
        initial_text: Union[str, Callable[[], str]] = random_char_seed,
        max_length: int = 512,
        unchanged_threshold: int = 30,
        ignored_n_grams: Optional[Sequence[str]] = None,
        repetition_n_gram_size: int = 7,
        repetition_frequency: int = 20,
    ) -> None:
        if callable(initial_text):
            self._seed_factory: Callable[[], str] = initial_text
        else:
            _fixed = initial_text
            self._seed_factory = lambda: _fixed
        self.max_length = max_length
        self.unchanged_threshold = unchanged_threshold
        self.ignored_n_grams: List[str] = list(ignored_n_grams or [])
        self.repetition_n_gram_size = repetition_n_gram_size
        self.repetition_frequency = repetition_frequency

        # Snapshot of the most recent seed; refreshed by reset().
        self._initial_text: str = self._seed_factory()
        self.text: str = self._initial_text
        self._previous_texts: List[str] = []
        self._unchanged_count: int = 0

    @property
    def initial_text(self) -> str:
        """Most recent seed (the value reset() snaps back to)."""
        return self._initial_text

    def set_seed_source(self, factory: Callable[[], str], reseed: bool = True) -> None:
        """Point this context at a (possibly shared) seed source.

        Lets an owner like :class:`ContextStreams` hand several contexts the
        same anchor factory so they all reset to one common seed. With
        ``reseed=True`` the buffer immediately snaps to a fresh seed from it.
        """
        self._seed_factory = factory
        if reseed:
            self.reset()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update(self, new_text: str) -> bool:
        """Absorb one decoded generation result into the buffer.

        Returns ``True`` if the update triggered a reset back to the
        seed text (stuck output or detected degeneracy), ``False``
        otherwise. Callers typically only need the return value for
        logging ("[INFO] context reset - stuck").
        """
        # Track whether the model produced anything new.
        if self._previous_texts and new_text == self._previous_texts[-1]:
            self._unchanged_count += 1
        else:
            self._unchanged_count = 0

        self._previous_texts.append(new_text)
        if len(self._previous_texts) > 5:
            self._previous_texts.pop(0)

        # Reset path 1: stuck for too many fires in a row.
        if self._unchanged_count >= self.unchanged_threshold:
            self.reset()
            return True

        self.text = new_text
        # Character-based left-truncate. Generous default (512 chars)
        # is cheap to re-encode but small enough to keep prefill fast
        # under the Ray MF prefill-every-step model.
        if len(self.text) > self.max_length:
            self.text = self.text[-self.max_length :]

        # Reset path 2: the buffer itself is degenerate.
        if self._is_degenerate(self.text):
            self.reset()
            return True

        return False

    def reset(self) -> None:
        """Snap the buffer back to a fresh seed, clearing all history.

        With a callable seed factory this re-rolls a new starting text.
        """
        self._initial_text = self._seed_factory()
        self.text = self._initial_text
        self._previous_texts = []
        self._unchanged_count = 0

    @property
    def unchanged_count(self) -> int:
        return self._unchanged_count

    # ------------------------------------------------------------------
    # degeneracy heuristics (ported from TerminalInterface)
    # ------------------------------------------------------------------

    def _is_degenerate(self, text: str) -> bool:
        if not text:
            return False
        if text.isspace():
            return True
        if self._detect_ngram_repetition(
            text,
            n=self.repetition_n_gram_size,
            threshold=self.repetition_frequency,
        ):
            return True
        if self._detect_sequential_repetition(text, threshold=5, min_segment_length=8):
            return True
        if self._is_bracket_pipe_pattern(text):
            return True
        return False

    def _detect_ngram_repetition(self, text: str, n: int, threshold: int) -> bool:
        """Any character n-gram appears more than ``threshold`` times."""
        if len(text) < n:
            return False
        excluded = set(self.ignored_n_grams)
        n_grams = [text[i : i + n] for i in range(len(text) - n + 1)]
        filtered = [ng for ng in n_grams if ng not in excluded]
        counts = Counter(filtered)
        for count in counts.values():
            if count > threshold:
                return True
        return False

    @staticmethod
    def _detect_sequential_repetition(
        text: str, threshold: int, min_segment_length: int = 3
    ) -> bool:
        """Any unbroken ``pattern * k`` sequence with k >= ``threshold``."""
        if len(text) < min_segment_length:
            return False
        max_pattern_length = len(text) // 2
        for pattern_length in range(1, max_pattern_length + 1):
            if pattern_length * threshold < min_segment_length:
                continue
            for start in range(len(text) - pattern_length * threshold + 1):
                pattern = text[start : start + pattern_length]
                repeat_count = 1
                current_pos = start + pattern_length
                while (
                    current_pos + pattern_length <= len(text)
                    and text[current_pos : current_pos + pattern_length] == pattern
                ):
                    repeat_count += 1
                    current_pos += pattern_length
                    if (
                        repeat_count >= threshold
                        and pattern_length * repeat_count >= min_segment_length
                    ):
                        return True
        return False

    @staticmethod
    def _is_bracket_pipe_pattern(text: str) -> bool:
        """Detect the ``[tag]|[tag]|[tag]`` bracket-pipe degeneration."""
        if not text or len(text.strip()) == 0:
            return False
        lines = text.strip().split("\n")
        if len(lines) <= 1:
            return False
        pattern_lines = 0
        bracket_pipe_pattern = r"\[.+?\](\||\s*$)"
        for line in lines:
            if re.search(bracket_pipe_pattern, line):
                brackets = line.count("[") + line.count("]")
                pipes = line.count("|")
                if brackets >= 4 and pipes >= 1:
                    pattern_lines += 1
        return (pattern_lines / len(lines)) >= 0.5
