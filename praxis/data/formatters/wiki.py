"""Wikipedia article formatting."""

from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS


def format_wiki(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Format wiki text with unified system/developer prompts.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document (must be exactly 2 - title and body)
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dictionary with messages and metadata
    """
    assert len(keys) == 2, "Wiki format requires exactly 2 keys"
    title = document.get(keys[0], "")
    body = document.get(keys[1], "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["write_article"]},
        {"role": "user", "content": f"Write an article about: {title}"},
        {"role": "assistant", "content": body},
    ]

    return {
        "messages": messages,
        "metadata": {
            "format": "wiki",
            "source_keys": keys,
            "title": title
        }
    }