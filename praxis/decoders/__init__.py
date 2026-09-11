from functools import partial

from praxis import registry
from praxis.decoders.parallel import ParallelDecoder
from praxis.decoders.sequential import SequentialDecoder
from praxis.registry import Entry

registry.declare(
    "decoders",
    title="Block-stacking decoders",
    doc=(
        ("How the stack of blocks is composed (sequential, parallel, weighted, ...).")
    ),
    entries={
        "sequential": SequentialDecoder,
        "parallel_mean": Entry(
            partial(ParallelDecoder, mode="mean"),
            (
                "Run every expert in parallel on the same input and average their "
                "outputs. Parallel decoders have no early exit."
            ),
        ),
        "parallel_variance": Entry(
            partial(ParallelDecoder, mode="variance"),
            (
                "Run every expert in parallel on the same input and sum their outputs, "
                "each feature weighted by a sigmoid of that expert's log-variance "
                "around the expert mean, so outputs that depart from the consensus "
                "count for more."
            ),
        ),
        "parallel_weighted": Entry(
            partial(ParallelDecoder, mode="weighted"),
            (
                "Run every expert in parallel on the same input and sum their outputs "
                "with learned per-expert, per-feature weights, softmaxed across "
                "experts."
            ),
        ),
    },
)
