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
# chance falls, so the hotter contexts sample rarely and cost little. The two
# rarely-fired blocks grow a double-length buffer since their cost is amortized
# across far fewer inference steps.
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
    ContextBlock(
        "Balanced", "Mid temperature - samples about 1 step in 10.", 0.5, 0.1, 2.0
    ),
    ContextBlock(
        "Wild", "High temperature - samples about 1 step in 100.", 1.0, 0.01, 2.0
    ),
]


class ContextStreams:
    """N independent rolling contexts, one :class:`StreamingContext` per block.

    ``context_factory`` builds a fresh StreamingContext with the producer's own
    tuning (seed, max length, repetition thresholds); it is called once per block
    and receives that block so it can honor per-block knobs like ``context_scale``.

    ``token_counter`` (optional) encodes a block's text into prompt tokens for the
    per-block counter chip; without it, blocks report no token count.
    """

    def __init__(
        self,
        context_factory: Callable[[ContextBlock], StreamingContext],
        blocks: Optional[List[ContextBlock]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.blocks: List[ContextBlock] = list(blocks or DEFAULT_CONTEXT_BLOCKS)
        self.contexts: List[StreamingContext] = [
            context_factory(b) for b in self.blocks
        ]
        self._token_counter = token_counter

    @property
    def primary(self) -> StreamingContext:
        """The always-on (chance 1.0) context; drives the CLI + back-compat status_text."""
        return self.contexts[0]

    def step(self, generate_fn: Callable[[str, float], Optional[str]]) -> List[dict]:
        """One inference opportunity. For each block, with probability ``chance``,
        generate at the block's temperature from its running text and fold the
        result back in. Returns the JSON-serializable payload for LiveMetrics."""
        for block, ctx in zip(self.blocks, self.contexts):
            if block.chance >= 1.0 or random.random() < block.chance:
                result = generate_fn(ctx.text, block.temperature)
                if result is not None:
                    ctx.update(result)
        return self.payload()

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
        for c in self.contexts:
            c.reset()
