"""Instruction-following format."""

from typing import Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt
from praxis.data.formatters.base import text_formatter


def format_instruction(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Format as instruction/output pairs with unified system/developer prompts.

    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document (must be exactly 2)
        tokenizer: Tokenizer with chat template support

    Returns:
        Dictionary with messages and metadata
    """
    assert len(keys) == 2, "Instruction format requires exactly 2 keys"
    instruction = text_formatter(document.get(keys[0], ""))
    output = text_formatter(document.get(keys[1], ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": sample_developer_prompt("follow_instruction")},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    return {
        "messages": messages,
        "metadata": {"format": "instruction", "source_keys": keys},
    }
