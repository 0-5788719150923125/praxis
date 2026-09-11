"""Praxis data loading and processing utilities."""

from praxis import registry
from praxis.registry import Entry

registry.declare(
    "samplers",
    title="Data sampler strategies",
    doc=(
        (
            "How datasets are interleaved during training. Praxis trains on multiple "
            "datasets at once: at every step the trainer picks a dataset, draws a "
            "document, and tokenizes it (see ``InterleaveDataManager`` in "
            "``praxis/data/datasets/manager.py``). The sampler chosen here decides *how* "
            "that pick is biased - either statically from configured weights, or "
            "adaptively based on document length, novelty, or per-dataset loss. Each entry "
            "is a mode string that ``InterleaveDataManager`` interprets."
        )
    ),
    entries={
        "novelty": Entry(
            "novelty",
            (
                "Bias sampling toward datasets that are still producing novel content. "
                "A Count-Min Sketch tracks bigram frequencies per dataset (see "
                "``NoveltyTracker`` in ``praxis/data/datasets/novelty.py``); datasets "
                "whose bigrams are mostly already-seen get down-weighted. Good for "
                "noisy or repetitive corpora."
            ),
        ),
        "dynamic": Entry(
            "dynamic",
            (
                "Balance datasets by document length, tracked via EMA "
                "(``ema_alpha=0.3``). A dataset that produces longer documents gets "
                "sampled less often so the per-token mix matches the configured "
                "weights. Useful when sources have very different document sizes."
            ),
        ),
        "static": Entry(
            "static",
            (
                "Use the per-dataset weights from ``DATASET_COLLECTIONS`` as "
                "configured and never adapt them. Predictable, no feedback loop. Pick "
                "this when you want full control of the data mix."
            ),
        ),
        "loss": Entry(
            "loss",
            (
                "Bias sampling toward datasets the model is currently doing worst on. "
                "The trainer reports per-dataset losses via "
                "``InterleaveDataManager.update_losses``; weights are computed by "
                "temperature-softmax (``T=8``) over EMA-smoothed losses, mixed with a "
                "uniform floor (``alpha=0.2``) so no single high-loss source captures "
                "everything, then length-normalized so a dataset that produces N "
                "sequences per document is fetched ~1/N as often."
            ),
        ),
        "tasker": Entry(
            "tasker",
            (
                "Close the loop between sampling and the model's own per-task loss "
                "weighter (``--task-weights``). The trainer pushes the tasker's "
                "``effective_weights`` via "
                "``InterleaveDataManager.update_task_weights``; each dataset's "
                "sampling weight becomes its configured weight times its task's "
                "learned weight, mixed with a uniform floor and normalized. With "
                "``difficulty`` task weights a hard task is both upweighted (loss) and "
                "upsampled (data) - the model spends more steps where it is worst. "
                "Pair with a ``difficulty`` weighter; the ``learnable`` variant "
                "downweights hard tasks and would invert the intent."
            ),
        ),
        "uniform": Entry(
            "uniform",
            (
                "Ignore the per-dataset weights from ``DATASET_COLLECTIONS`` - every "
                "sampler is weighted 1.0 (then normalized) and never adapts. If you "
                "want source biasing without changing the data mix, pair this with "
                "``--task-weights`` to bias at the loss level instead."
            ),
        ),
    },
)

registry.declare(
    "seq_curriculum",
    title="Sequence-length curriculum",
    doc=(
        (
            "How the per-batch sequence-length multiplier is chosen. Every batch trades "
            "batch size for sequence length at constant attention cost; this picks the "
            "mix."
        )
    ),
    entries={
        "fixed": Entry(
            "fixed",
            (
                "Roll the static per-tier chances in ``SEQUENCE_MULTIPLIER_TIERS`` "
                "(the default). Each batch independently trades batch size for "
                "sequence length at constant attention cost, with fixed probabilities."
            ),
        ),
        "probe": Entry(
            "probe",
            (
                "Fit the sequence-length mix by attributing a held-out probe's "
                "improvement to the arm mixture that produced it: every window of "
                "optimizer steps, re-score a fixed probe and regress its loss decrease "
                "onto the window's per-arm visit counts (recursive least squares with "
                "forgetting). The coefficients are each arm's measured value in "
                "held-out loss, scored as t-statistics so an absence of signal renders "
                "as a uniform mix rather than a confident one. Costs one probe forward "
                "per arm per window. See ``praxis/data/seq_probe.py``."
            ),
        ),
    },
)

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
    add_datasets,
    get_datamodules,
    get_dataset,
    get_dataset_configs,
)

__all__ = [
    # Registries
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
    "add_datasets",
    "get_dataset_configs",
]
