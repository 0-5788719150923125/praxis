"""Data formatters for various dataset types."""

from praxis.data.formatters.base import (
    add_newline_before_lists,
    repair_broken_emoticons,
    repair_text_punctuation,
    simple_truecase,
    text_formatter,
)
from praxis.data.formatters.conversation import (
    create_person_mapping,
    format_conversation,
    format_messages,
    format_soda,
    replace_person_references,
)
from praxis.data.formatters.cot import COT_TAGS, format_cot
from praxis.data.formatters.files import format_file_as_messages
from praxis.data.formatters.instruction import format_instruction
from praxis.data.formatters.persona import format_personachat
from praxis.data.formatters.rl import RLLogger, _rl_logger, format_rl
from praxis.data.formatters.simple import format_simple
from praxis.data.formatters.tools import format_tool_calling
from praxis.data.formatters.wiki import format_wiki

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
