from functools import partial

from praxis import registry
from praxis.compression.base import NoCompression
from praxis.compression.sequence_interpolation import SequenceInterpolation
from praxis.registry import Entry

registry.declare(
    "compression",
    title="Sequence compression",
    doc=(("Strategies for reducing sequence length between layers.")),
    entries={
        "none": NoCompression,
        "linear": Entry(
            partial(SequenceInterpolation, method="linear", factor=0.9),
            (
                "Sequence interpolation with linear resampling, shrinking the sequence "
                "to 90% of its length and expanding it back."
            ),
        ),
        "nearest": Entry(
            partial(SequenceInterpolation, method="nearest", factor=0.9),
            (
                "Sequence interpolation with nearest-neighbour resampling, shrinking "
                "the sequence to 90% of its length and expanding it back."
            ),
        ),
    },
)
