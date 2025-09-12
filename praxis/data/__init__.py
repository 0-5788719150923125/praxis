"""Praxis data loading and processing utilities."""

# Core data structures and configuration
from praxis.data.config import (
    DataFormat,
    DATASET_COLLECTIONS,
    HUGGINGFACE_DATASETS,
    SYSTEM_PROMPT,
    DEVELOPER_PROMPTS,
    DEFAULT_WEIGHT,
    SRC_WEIGHT,
    DIR_WEIGHT,
    TOOLS_WEIGHT,
)

# Data formats
from praxis.data.formats import DataFormat as DataFormatEnum, detect_format

# Formatters
from praxis.data.formatters import (
    # Base utilities
    text_formatter,
    add_newline_before_lists,
    repair_text_punctuation,
    repair_broken_emoticons,
    simple_truecase,
    # Format functions
    format_simple,
    format_instruction,
    format_conversation,
    format_messages,
    format_soda,
    format_personachat,
    format_wiki,
    format_rl,
    format_cot,
    format_tool_calling,
    # Utilities
    create_person_mapping,
    replace_person_references,
    # RL logging
    RLLogger,
    _rl_logger,
    # CoT tags
    COT_TAGS,
)

# Datasets
from praxis.data.datasets import (
    PraxisSampler,
    load_dataset_smart,
    HuggingfaceDataset,
    SyntheticToolCallingDataset,
    MultiDirectoryDataset,
    WeightedIterableDataset,
    InterleaveDataManager,
    FORMAT_HANDLERS,
)

# DataModule
from praxis.data.datamodule import PraxisDataModule

# Utility functions
from praxis.data.utils import (
    get_datamodules,
    get_dataset,
    add_collection,
    get_dataset_configs,
)

__all__ = [
    # Configuration
    "DataFormat",
    "DataFormatEnum",
    "DATASET_COLLECTIONS",
    "HUGGINGFACE_DATASETS",
    "SYSTEM_PROMPT",
    "DEVELOPER_PROMPTS",
    "DEFAULT_WEIGHT",
    "SRC_WEIGHT",
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
