"""Conversation and dialogue formatting."""

import json
import random
from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS
from praxis.data.formatters.base import text_formatter, repair_text_punctuation, repair_broken_emoticons, simple_truecase


def format_conversation(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Format as a conversation with unified system/developer prompts.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document (must be exactly 3)
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dictionary with messages and metadata
    """
    assert len(keys) == 3, "Conversation format requires exactly 3 keys"

    # Original system message becomes developer message
    original_system = text_formatter(document.get(keys[0], ""))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": original_system or DEVELOPER_PROMPTS["engage_conversation"],
        },
        {"role": "user", "content": text_formatter(document.get(keys[1], ""))},
        {"role": "assistant", "content": text_formatter(document.get(keys[2], ""))},
    ]

    return {
        "messages": messages,
        "metadata": {
            "format": "conversation",
            "source_keys": keys
        }
    }


def format_messages(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Convert already formatted messages with unified system/developer prompts.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dictionary with messages and metadata
    """
    messages_key = keys[0]
    messages = document.get(messages_key, [])

    if not messages:
        return {"messages": [], "metadata": {}}

    # Preprocess messages to add unified system/developer prompts
    processed_messages = []

    # Add our unified system prompt
    processed_messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Determine developer prompt based on content
    if any(msg.get("role") == "user" for msg in messages):
        developer_prompt = DEVELOPER_PROMPTS["engage_conversation"]
    else:
        developer_prompt = DEVELOPER_PROMPTS["continue_text"]

    processed_messages.append({"role": "developer", "content": developer_prompt})

    # Process original messages, filtering out original system messages
    for msg in messages:
        role = msg.get("role", "")

        # Skip original system messages - we have our own
        if role == "system":
            continue

        # Convert 'human' to 'user' for consistency
        if role == "human":
            role = "user"

        content = text_formatter(msg.get("content", ""))
        if content:
            processed_messages.append({"role": role, "content": content})

    return {
        "messages": processed_messages,
        "metadata": {
            "format": "messages",
            "source_keys": keys
        }
    }


def format_soda(document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer) -> str:
    """Format SODA dataset entries as conversations.
    
    The SODA dataset contains dialogues with context and speaker information.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Formatted text with chat template applied
    """
    speakers = document.get(keys[0], [])  # speakers
    narrative = document.get(keys[1], "")  # narrative
    literal = document.get(keys[2], "")  # literal
    dialogue = document.get(keys[3], [])  # dialogue
    head = document.get(keys[4], "")  # head
    relation = document.get(keys[5], "")  # relation
    tail = document.get(keys[6], "")  # tail

    if not dialogue or len(dialogue) < 2:
        return ""

    # Create context from narrative and relation
    context_parts = []
    if narrative:
        context_parts.append(f"Context: {narrative}")
    if head and relation and tail:
        context_parts.append(f"Situation: {head} {relation} {tail}")

    # Build conversation with unified system/developer prompts
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": DEVELOPER_PROMPTS["soda_dialogue"]},
    ]

    # Add context as a developer message if available
    if context_parts:
        messages.append(
            {"role": "developer", "content": "\n".join(context_parts)}
        )

    # Create person mapping if we have speakers
    person_mapping = create_person_mapping({"speakers": speakers}) if speakers else {}

    # Add dialogue turns
    for i, turn in enumerate(dialogue):
        # Clean and repair the text
        cleaned_turn = repair_broken_emoticons(repair_text_punctuation(turn.strip()))
        # Apply truecasing
        cleaned_turn = simple_truecase(cleaned_turn)
        # Replace person references
        if person_mapping:
            cleaned_turn = replace_person_references(cleaned_turn, person_mapping)

        # Alternate between user and assistant
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": cleaned_turn})

    return {
        "messages": messages,
        "metadata": {
            "format": "soda",
            "source_keys": keys,
            "dialogue_turns": len(dialogue)
        }
    }


def create_person_mapping(example: Dict) -> Dict[str, str]:
    """Create a mapping from PersonX/Y/Z to random names."""
    names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery"]
    random.shuffle(names)

    mapping = {}
    person_count = 0

    # Check all fields for PersonX/Y/Z references
    for field in example.values():
        if isinstance(field, str):
            text = field
        elif isinstance(field, list):
            text = " ".join(str(item) for item in field)
        else:
            continue

        for person in ["PersonX", "PersonY", "PersonZ"]:
            if person in text and person not in mapping:
                if person_count < len(names):
                    mapping[person] = names[person_count]
                    person_count += 1

    return mapping


def replace_person_references(text: str, mapping: Dict[str, str]) -> str:
    """Replace PersonX/Y/Z with actual names."""
    for person, name in mapping.items():
        # Replace the person reference
        text = text.replace(person, name)
        # Also handle possessive forms
        text = text.replace(f"{person}'s", f"{name}'s")

    return text