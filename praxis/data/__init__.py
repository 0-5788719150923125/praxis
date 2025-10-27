"""Praxis data loading and processing utilities."""

# Core data structures and configuration
from praxis.data.config import (
    DATASET_COLLECTIONS,
    DEFAULT_WEIGHT,
    DEVELOPER_PROMPTS,
    DIR_WEIGHT,
    HUGGINGFACE_DATASETS,
    SRC_WEIGHT,
    SYSTEM_PROMPT,
    TOOLS_WEIGHT,
    DataFormat,
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
