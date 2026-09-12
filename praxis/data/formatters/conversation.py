"""Conversation and dialogue formatting."""

import json
import random
import re
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt
from praxis.data.formatters.base import (
    repair_broken_emoticons,
    repair_text_punctuation,
    simple_truecase,
    text_formatter,
)


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
            "content": original_system
            or sample_developer_prompt("engage_conversation"),
        },
        {"role": "user", "content": text_formatter(document.get(keys[1], ""))},
        {"role": "assistant", "content": text_formatter(document.get(keys[2], ""))},
    ]

    return {
        "messages": messages,
        "metadata": {"format": "conversation", "source_keys": keys},
    }


def format_messages(
    document: Dict,
    keys: List[str],
    tokenizer: PreTrainedTokenizer,
    developer_prompt: Optional[str] = None,
) -> Dict:
    """Convert already formatted messages with unified system/developer prompts.

    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document
        tokenizer: Tokenizer with chat template support
        developer_prompt: Use this developer prompt instead of sampling one.
            The two sides of a preference pair have to carry an IDENTICAL
            prompt, and sampling is random.

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
    if developer_prompt is None:
        if any(msg.get("role") == "user" for msg in messages):
            developer_prompt = sample_developer_prompt("engage_conversation")
        else:
            developer_prompt = sample_developer_prompt("continue_text")

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
        "metadata": {"format": "messages", "source_keys": keys},
    }


# A "\n\nHuman: ... \n\nAssistant: ..." transcript turn. The leading separator
# a caller prepends guarantees the first marker matches even after strip().
_TRANSCRIPT_TURN = re.compile(
    r"\n\n(Human|Assistant): (.*?)(?=\n\n(?:Human|Assistant): |\Z)", flags=re.S
)


def parse_transcript(transcript: str) -> List[Dict]:
    """Raw "Human:/Assistant:" transcript -> chat messages, empty turns dropped."""
    turns = _TRANSCRIPT_TURN.findall("\n\n" + (transcript or "").strip())
    return [
        {"role": "user" if speaker == "Human" else "assistant", "content": text}
        for speaker, text in turns
        if text.strip()
    ]


def format_human_assistant(
    document: Dict,
    keys: List[str],
    tokenizer: PreTrainedTokenizer,
    developer_prompt: Optional[str] = None,
) -> Dict:
    """Parse a raw "\\n\\nHuman: ... \\n\\nAssistant: ..." transcript (e.g. the
    Anthropic/hh-rlhf `chosen` column) into chat messages, then reuse the
    messages pipeline so it gets the unified system/developer prompts.
    """
    messages = parse_transcript(document.get(keys[0], "") or "")
    if not messages:
        return {"messages": [], "metadata": {}}
    return format_messages(
        {"messages": messages},
        ["messages"],
        tokenizer,
        developer_prompt=developer_prompt,
    )


def format_preference_pair(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Preference pair (e.g. Anthropic/hh-rlhf ``chosen``/``rejected``): emit
    BOTH sides of one pair, truncated at their first divergent turn and sharing
    a character-identical prompt, so the margin compares two answers to the
    same question.

    The rejected side rides along under ``pair_with``; the manager enqueues the
    two adjacently under a shared ``pair_id`` and the packer carries that id on
    the divergent response only (see ``MessageQueueManager.get_batch``). Both
    halves are needed: without them co-resident the margin contrasts one
    conversation against an unrelated one, and without the truncation it
    contrasts a transcript against its own shared prefix - measured at 42% of
    the assistant tokens on a rejected draw, text character-identical to the
    chosen side that the margin was pushing DOWN.

    Task tags stay per document, which is what keeps the card's contract
    simple: chosen text is ``PREF_CHOSEN`` (trains as conversation data and
    anchors the margin), rejected text is ``PREF_REJECTED`` (contrast-only -
    the main CE excludes ALL of it, including the duplicated prompt, so the
    shared context is never trained twice).
    """
    from praxis.tasks import TaskType

    chosen_key = keys[0] if keys else "chosen"
    rejected_key = keys[1] if len(keys) > 1 else "rejected"

    chosen = parse_transcript(document.get(chosen_key, ""))
    rejected = parse_transcript(document.get(rejected_key, ""))

    # Find the shared prompt. Measured over 4k hh-rlhf pairs: 100% share a turn
    # prefix and diverge on an assistant turn, and 99.7% diverge at the final
    # turn, so truncating at the divergence costs almost nothing. A pair that
    # diverges on a user turn is asking two different questions and has no
    # shared context to score against.
    split = 0
    while split < min(len(chosen), len(rejected)) and chosen[split] == rejected[split]:
        split += 1
    if split >= len(chosen) or split >= len(rejected):
        return {"messages": [], "metadata": {}}
    if chosen[split]["role"] != "assistant" or rejected[split]["role"] != "assistant":
        return {"messages": [], "metadata": {}}

    # Sampled ONCE. Formatting the sides independently draws a different
    # developer prompt for each (sample_developer_prompt is random), which is a
    # difference between the two that has nothing to do with the preference.
    developer_prompt = sample_developer_prompt("engage_conversation")

    def build_side(messages: List[Dict], task: "TaskType") -> Optional[Dict]:
        side = format_messages(
            {"messages": messages[: split + 1]},
            ["messages"],
            tokenizer,
            developer_prompt=developer_prompt,
        )
        if not side.get("messages"):
            return None
        # The packer reads the pair span off the document's LAST assistant run,
        # which is the divergent turn only because of the truncation above.
        # text_formatter can empty a turn, and format_messages drops it, which
        # would leave a SHARED turn last - so check rather than assume.
        if side["messages"][-1].get("role") != "assistant":
            return None
        side["metadata"]["task_type"] = int(task)
        return side

    chosen_side = build_side(chosen, TaskType.PREF_CHOSEN)
    rejected_side = build_side(rejected, TaskType.PREF_REJECTED)
    if chosen_side is None or rejected_side is None:
        return {"messages": [], "metadata": {}}

    chosen_side["pair_with"] = rejected_side
    return chosen_side


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
        {"role": "developer", "content": sample_developer_prompt("soda_dialogue")},
    ]

    # Add context as a developer message if available
    if context_parts:
        messages.append({"role": "developer", "content": "\n".join(context_parts)})

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
            "dialogue_turns": len(dialogue),
        },
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
