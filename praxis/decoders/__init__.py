from functools import partial

from praxis.decoders.mono_forward import MonoForwardDecoder
from praxis.decoders.parallel import ParallelDecoder
from praxis.decoders.sequential import SequentialDecoder

DECODER_REGISTRY = dict(
    sequential=SequentialDecoder,
    mono_forward=MonoForwardDecoder,
    parallel_mean=partial(ParallelDecoder, mode="mean"),
    parallel_variance=partial(ParallelDecoder, mode="variance"),
    parallel_weighted=partial(ParallelDecoder, mode="weighted"),
)
