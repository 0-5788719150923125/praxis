"""Persona-based chat formatting."""

import random
from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS
from praxis.data.formatters.base import text_formatter, simple_truecase


def format_personachat(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Format AlekseyKorshuk persona-chat dataset with JSON structure for better randomization.
    
    Args:
        document: Dictionary containing the document data
        keys: List of keys to extract from document
        tokenizer: Tokenizer with chat template support
        
    Returns:
        Dictionary with messages and metadata
    """
    # Extract personality traits and utterances
    personality = document.get("personality", [])
    utterances = document.get("utterances", [])

    if not utterances:
        return {"messages": [], "metadata": {}}

    # Randomly select how much of the conversation to include (1 to all utterances)
    num_utterances = random.randint(1, len(utterances))
    selected_utterance = utterances[num_utterances - 1]  # Get the selected utterance

    # Decide whether to apply truecasing (50% chance, applies to entire conversation)
    apply_truecase = random.random() < 0.5

    # Build personality description for developer message
    # Always include personality when available (it's the assistant's personality)
    developer_message = ""
    if personality:
        developer_message = "You have the following personality traits:\n"
        # Randomly select subset of personality traits (or all)
        if random.random() < 0.5:
            # 50% chance to use all traits
            selected_traits = personality
        else:
            # 50% chance to use a random subset
            num_traits = random.randint(1, len(personality))
            selected_traits = random.sample(personality, num_traits)

        for trait in selected_traits:
            # Always truecase personality traits (they're for the assistant)
            trait_text = simple_truecase(trait) if trait else trait
            developer_message += f"- {trait_text}\n"
        developer_message = developer_message.strip()
    else:
        # Use default persona chat prompt if no personality provided
        developer_message = DEVELOPER_PROMPTS["persona_chat"]

    # Build messages list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "developer", "content": developer_message},
    ]

    # Add the conversation history from the selected utterance
    history = selected_utterance.get("history", [])
    candidates = selected_utterance.get("candidates", [])

    # Add history messages with proper role alternation (user starts)
    for i, hist_msg in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        # Apply truecasing based on role and setting
        if role == "assistant":
            # Assistant always uses proper casing
            content = simple_truecase(hist_msg) if hist_msg else hist_msg
        else:
            # User uses truecasing only if apply_truecase is True
            content = (
                simple_truecase(hist_msg) if (apply_truecase and hist_msg) else hist_msg
            )
        messages.append({"role": role, "content": text_formatter(content)})

    # Pick one candidate as the final assistant response
    if candidates:
        # Filter out candidates that look like conversation starters
        # (these sometimes appear incorrectly in the dataset)
        filtered_candidates = []
        for cand in candidates:
            cand_lower = cand.lower().strip()
            # Skip greetings that don't make sense in context
            if len(history) > 2:  # If we're deep in conversation
                if any(
                    phrase in cand_lower
                    for phrase in [
                        "hi there",
                        "hello",
                        "how are you today",
                        "hey there",
                        "hi, how are you",
                        "hello, how are you",
                    ]
                ):
                    continue  # Skip this candidate
            filtered_candidates.append(cand)

        # Use filtered candidates if any remain, otherwise fall back to all
        candidates_to_use = filtered_candidates if filtered_candidates else candidates
        response = random.choice(candidates_to_use)

        # Assistant always uses proper casing
        response = simple_truecase(response) if response else response
        messages.append({"role": "assistant", "content": text_formatter(response)})

    return {
        "messages": messages,
        "metadata": {
            "format": "personachat",
            "source_keys": keys,
            "has_personality": bool(personality),
            "num_utterances": num_utterances
        }
    }