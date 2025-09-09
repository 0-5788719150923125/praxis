"""Data formatters for various dataset types."""

from praxis.data.formatters.base import (
    text_formatter,
    add_newline_before_lists,
    repair_text_punctuation,
    repair_broken_emoticons,
    simple_truecase,
)
from praxis.data.formatters.simple import format_simple
from praxis.data.formatters.instruction import format_instruction
from praxis.data.formatters.conversation import (
    format_conversation,
    format_messages,
    format_soda,
    create_person_mapping,
    replace_person_references,
)
from praxis.data.formatters.persona import format_personachat
from praxis.data.formatters.wiki import format_wiki
from praxis.data.formatters.rl import format_rl, RLLogger, _rl_logger
from praxis.data.formatters.cot import format_cot, COT_TAGS
from praxis.data.formatters.tools import format_tool_calling
from praxis.data.formatters.files import format_file_as_messages

__all__ = [
    # Base utilities
    "text_formatter",
    "add_newline_before_lists",
    "repair_text_punctuation",
    "repair_broken_emoticons",
    "simple_truecase",
    # Format functions
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
    "format_file_as_messages",
    # Utilities
    "create_person_mapping",
    "replace_person_references",
    # RL logging
    "RLLogger",
    "_rl_logger",
    # CoT tags
    "COT_TAGS",
]