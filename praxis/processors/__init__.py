from functools import partial

from praxis.processors.parallel import ParallelProcessor
from praxis.processors.sequential import SequentialProcessor

PROCESSOR_REGISTRY = dict(
    sequential=SequentialProcessor,
    parallel_mean=partial(ParallelProcessor, mode="mean"),
    parallel_weighted=partial(ParallelProcessor, mode="weighted"),
)
