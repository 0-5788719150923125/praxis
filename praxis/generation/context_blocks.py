"""Declarative schema for the Terminal's rolling generation contexts.

The Terminal shows one or more "rolling contexts": passages the model grows
incrementally during training (see :class:`StreamingContext`). Each context is a
:class:`ContextBlock` - a name, a description, a sampling ``temperature``, and a
per-step ``chance`` of running inference. The chance throttles cost: the default
0.5/0.7/1.0 blocks fire at 1.0/0.1/0.01, so the two hotter contexts tick over
rarely and add little to the training forward pass.

Producers (the Lightning ``TerminalInterface`` callback and the mono-forward
hook) share one :class:`ContextStreams` manager and supply a
``generate_fn(prompt, temperature) -> str`` so the temperature-experiment schema
stays decoupled from how any given trainer actually runs generation. Override the
default blocks per experiment by passing your own list to ``ContextStreams``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from .streaming import StreamingContext, random_char_seed


@dataclass(frozen=True)
class ContextBlock:
    """One rolling context: a temperature experiment with a call probability."""

    name: str
    description: str
    temperature: float
    chance: float = 1.0  # per-step probability this block runs inference (1.0 = always)
    context_scale: float = (
        1.0  # multiplies the producer's base max_length for this block
    )


# Default Terminal contexts. Temperature rises (more exploratory) as the call
# chance falls, so the hotter contexts sample rarely and cost little. All three
# share one context window (context_scale 1.0) so the only thing that differs
# between them is temperature.
#
# Temperatures are reciprocals of integers (1/3, 1/2, 1) so the three views
# stay distinct under CALM's count-based patch-vote sampling, which quantizes
# T to round(1/T): the old 0.7 collapsed onto T=1 there, making Balanced and
# Wild the same experiment. For token models these are ordinary softmax temps
# with the same focused < balanced < wild spread.
DEFAULT_CONTEXT_BLOCKS: List[ContextBlock] = [
    ContextBlock(
        "Focused", "Low temperature - the most likely continuation.", 1.0 / 3.0, 1.0
    ),
    ContextBlock("Balanced", "Mid temperature - samples about 1 step in 10.", 0.5, 0.1),
    ContextBlock("Wild", "High temperature - samples about 1 step in 100.", 1.0, 0.01),
]


class ContextStreams:
    """An *anchored cohort* of rolling contexts: N temperature paths that all
    diverge from one shared seed (the **anchor**) and reset back to it.

    The point is a controlled experiment - every path starts from the identical
    initial condition, so what differs between them is purely temperature, not the
    seed. Each path that goes degenerate snaps back to the *same* anchor rather
    than re-rolling its own, so the comparison holds across resets. The anchor is
    only re-rolled once a **quorum** of paths (a majority) has each gone degenerate
    at least ``reseed_threshold`` times - i.e. the cohort has voted that this seed
    is exhausted.

    ``context_factory`` builds a fresh StreamingContext with the producer's own
    tuning (max length, repetition thresholds); it is called once per block. Its
    seed source is then overridden with the shared anchor.

    ``seed_factory`` mints a new anchor (default: a random printable char).
    ``token_counter`` (optional) sizes a block's text into prompt tokens.
    """

    def __init__(
        self,
        context_factory: Callable[[ContextBlock], StreamingContext],
        blocks: Optional[List[ContextBlock]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        seed_factory: Callable[[], str] = random_char_seed,
        reseed_threshold: int = 3,
    ) -> None:
        self.blocks: List[ContextBlock] = list(blocks or DEFAULT_CONTEXT_BLOCKS)
        self.contexts: List[StreamingContext] = [
            context_factory(b) for b in self.blocks
        ]
        self._token_counter = token_counter
        self._mint = seed_factory
        self.reseed_threshold = reseed_threshold

        # The shared anchor. Every context reads it through one closure, so a
        # context's own reset() snaps it back to the cohort's current seed.
        self._anchor: str = self._mint()
        self._degeneracy_counts: List[int] = [0] * len(self.contexts)
        for ctx in self.contexts:
            ctx.set_seed_source(lambda: self._anchor)  # also reseeds to the anchor

    @property
    def primary(self) -> StreamingContext:
        """The always-on (chance 1.0) context; drives the CLI + back-compat status_text."""
        return self.contexts[0]

    @property
    def anchor(self) -> str:
        """The shared seed every path currently resets to."""
        return self._anchor

    @property
    def quorum(self) -> int:
        """How many paths must each hit ``reseed_threshold`` to re-anchor (majority)."""
        return len(self.contexts) // 2 + 1

    def step(self, generate_fn: Callable[[str, float], Optional[str]]) -> List[dict]:
        """One inference opportunity. For each block, with probability ``chance``,
        generate at the block's temperature from its running text and fold the
        result back in. A path that degenerates is tallied; once a quorum of paths
        is exhausted the shared anchor is re-rolled. Returns the LiveMetrics payload."""
        for i, (block, ctx) in enumerate(zip(self.blocks, self.contexts)):
            if block.chance >= 1.0 or random.random() < block.chance:
                result = generate_fn(ctx.text, block.temperature)
                if result is not None and ctx.update(result):
                    # update() returns True when it reset back to the anchor.
                    self._degeneracy_counts[i] += 1
        self._maybe_reanchor()
        return self.payload()

    def _maybe_reanchor(self) -> None:
        """Re-roll the shared anchor once a quorum of paths has each gone
        degenerate at least ``reseed_threshold`` times - the seed is spent."""
        exhausted = sum(c >= self.reseed_threshold for c in self._degeneracy_counts)
        if exhausted >= self.quorum:
            self._reanchor()

    def _reanchor(self) -> None:
        """Mint a new shared anchor, clear the tally, snap every path to it."""
        self._anchor = self._mint()
        self._degeneracy_counts = [0] * len(self.contexts)
        for ctx in self.contexts:
            ctx.reset()  # reads the new anchor through its shared seed source

    def payload(self) -> List[dict]:
        """Current state of every block, for the live snapshot / web stream."""
        return [
            {
                "name": b.name,
                "description": b.description,
                "temperature": b.temperature,
                "chance": b.chance,
                "text": c.text,
                "tokens": self._token_counter(c.text) if self._token_counter else None,
            }
            for b, c in zip(self.blocks, self.contexts)
        ]

    def reset(self) -> None:
        """Full cohort reset: a new shared anchor and a cleared tally."""
        self._reanchor()
