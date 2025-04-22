from praxis.processors.parallel import parallel_processor
from praxis.processors.sequential import sequential_processor

PROCESSOR_REGISTRY = dict(sequential=sequential_processor, parallel=parallel_processor)
