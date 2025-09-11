"""Simple text formatting."""

from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS
from praxis.data.formatters.base import text_formatter


def format_simple(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Convert raw text to unified format with system/developer prompts.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dictionary with messages and metadata
    """
    text = document.get(keys[0], "")
    if not text:
        return {"messages": [], "metadata": {}}

    # For simple/raw text, we treat it as a direct completion task
    # The model should learn to continue/complete texts naturally
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["continue_text"]},
        {"role": "assistant", "content": text_formatter(text)},
    ]

    return {
        "messages": messages,
        "metadata": {
            "format": "simple",
            "source_keys": keys
        }
    }