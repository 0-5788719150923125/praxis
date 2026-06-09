"""Praxis data loading and processing utilities."""

# Sampler (weighting mode) registry. Each mode is a string consumed by
# InterleaveDataManager (see praxis/data/datasets/manager.py), which holds
# the actual logic. The registry exists so the CLI can validate --sampler.
SAMPLER_REGISTRY = {
    "novelty": "novelty",
    "dynamic": "dynamic",
    "static": "static",
    "loss": "loss",
    "tasker": "tasker",
    "uniform": "uniform",
}

# Surfaced in docs/data.md. Keep these tight - the implementation in
# manager.py is the source of truth for behavior.
SAMPLER_DESCRIPTIONS = {
    "static": (
        "Use the per-dataset weights from ``DATASET_COLLECTIONS`` as configured "
        "and never adapt them. Predictable, no feedback loop. Pick this when "
        "you want full control of the data mix."
    ),
    "dynamic": (
        "Balance datasets by document length, tracked via EMA "
        "(``ema_alpha=0.3``). A dataset that produces longer documents gets "
        "sampled less often so the per-token mix matches the configured "
        "weights. Useful when sources have very different document sizes."
    ),
    "novelty": (
        "Bias sampling toward datasets that are still producing novel content. "
        "A Count-Min Sketch tracks bigram frequencies per dataset (see "
        "``NoveltyTracker`` in ``praxis/data/datasets/novelty.py``); datasets "
        "whose bigrams are mostly already-seen get down-weighted. Good for "
        "noisy or repetitive corpora. The default."
    ),
    "loss": (
        "Bias sampling toward datasets the model is currently doing worst on. "
        "The trainer reports per-dataset losses via "
        "``InterleaveDataManager.update_losses``; weights are computed by "
        "temperature-softmax (``T=8``) over EMA-smoothed losses, mixed with a "
        "uniform floor (``alpha=0.2``) so no single high-loss source captures "
        "everything, then length-normalized so a dataset that produces N "
        "sequences per document is fetched ~1/N as often."
    ),
    "tasker": (
        "Close the loop between sampling and the model's own per-task loss "
        "weighter (``--task-weights``). The trainer pushes the tasker's "
        "``effective_weights`` via ``InterleaveDataManager.update_task_weights``; "
        "each dataset's sampling weight becomes its configured weight times its "
        "task's learned weight, mixed with a uniform floor and normalized. With "
        "``difficulty`` task weights a hard task is both upweighted (loss) and "
        "upsampled (data) - the model spends more steps where it is worst. Pair "
        "with a ``difficulty`` weighter; the ``learnable`` variant downweights "
        "hard tasks and would invert the intent."
    ),
    "uniform": (
        "Ignore the per-dataset weights from ``DATASET_COLLECTIONS`` - every "
        "sampler is weighted 1.0 (then normalized) and never adapts. If you "
        "want source biasing without changing the data mix, pair this with "
        "``--task-weights`` to bias at the loss level instead."
    ),
}

# Core data structures and configuration
from praxis.data.config import (
    DATASET_COLLECTIONS,
    DATASETS,
    DEFAULT_WEIGHT,
    DEVELOPER_PROMPTS,
    DIR_WEIGHT,
    SYSTEM_PROMPT,
    TOOLS_WEIGHT,
    DataFormat,
    sample_developer_prompt,
)

# DataModule
from praxis.data.datamodule import PraxisDataModule

# Datasets
from praxis.data.datasets import (
    FORMAT_HANDLERS,
    HuggingfaceDataset,
    InterleaveDataManager,
    MultiDirectoryDataset,
    PraxisSampler,
    SyntheticToolCallingDataset,
    WeightedIterableDataset,
    load_dataset_smart,
)

# Data formats
from praxis.data.formats import DataFormat as DataFormatEnum
from praxis.data.formats import detect_format

# Formatters
from praxis.data.formatters import (  # Base utilities; Format functions; Utilities; RL logging; CoT tags
    COT_TAGS,
    RLLogger,
    _rl_logger,
    add_newline_before_lists,
    create_person_mapping,
    format_conversation,
    format_cot,
    format_instruction,
    format_messages,
    format_personachat,
    format_rl,
    format_simple,
    format_soda,
    format_tool_calling,
    format_wiki,
    repair_broken_emoticons,
    repair_text_punctuation,
    replace_person_references,
    simple_truecase,
    text_formatter,
)

# Utility functions
from praxis.data.utils import (
    add_collection,
    get_datamodules,
    get_dataset,
    get_dataset_configs,
)

__all__ = [
    # Registries
    "SAMPLER_REGISTRY",
    "SAMPLER_DESCRIPTIONS",
    # Configuration
    "DataFormat",
    "DataFormatEnum",
    "DATASET_COLLECTIONS",
    "DATASETS",
    "SYSTEM_PROMPT",
    "DEVELOPER_PROMPTS",  # Now returns lists of keywords
    "sample_developer_prompt",  # Use this to sample from keyword lists
    "DEFAULT_WEIGHT",
    "DIR_WEIGHT",
    "TOOLS_WEIGHT",
    # Format detection
    "detect_format",
    # Formatters
    "text_formatter",
    "add_newline_before_lists",
    "repair_text_punctuation",
    "repair_broken_emoticons",
    "simple_truecase",
    "format_simple",
    "format_instruction",
    "format_conversation",
    "format_messages",
    "format_soda",
    "format_personachat",
    "format_wiki",
    "format_rl",
    "format_cot",
    "format_tool_calling",
    "create_person_mapping",
    "replace_person_references",
    "RLLogger",
    "_rl_logger",
    "COT_TAGS",
    # Datasets
    "PraxisSampler",
    "load_dataset_smart",
    "HuggingfaceDataset",
    "SyntheticToolCallingDataset",
    "MultiDirectoryDataset",
    "WeightedIterableDataset",
    "InterleaveDataManager",
    "FORMAT_HANDLERS",
    # DataModule
    "PraxisDataModule",
    # Utilities
    "get_datamodules",
    "get_dataset",
    "add_collection",
    "get_dataset_configs",
]
