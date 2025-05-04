from functools import partial

from praxis.compression.base import NoCompression
from praxis.compression.sequence_interpolation import SequenceInterpolation

COMPRESSION_REGISTRY = dict(
    none=NoCompression,
    linear=partial(SequenceInterpolation, method="linear", factor=0.9),
    nearest=partial(SequenceInterpolation, method="nearest", factor=0.9),
)
