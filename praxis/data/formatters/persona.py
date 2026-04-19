"""Persona-based chat formatting."""

import random
from typing import Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt
from praxis.data.formatters.base import (
    repair_broken_emoticons,
    repair_text_punctuation,
    simple_truecase,
    text_formatter,
)


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

    # When personality traits are available, use them (shuffled into a
    # paragraph) as the developer message directly. Lists biased the model
    # toward list-shaped outputs, and a sampled keyword prefix added noise.
    developer_message = ""
    if personality:
        if random.random() < 0.5:
            selected_traits = list(personality)
        else:
            num_traits = random.randint(1, len(personality))
            selected_traits = random.sample(personality, num_traits)

        random.shuffle(selected_traits)

        sentences = []
        for trait in selected_traits:
            if not trait:
                continue
            cleaned = repair_broken_emoticons(
                repair_text_punctuation(simple_truecase(trait))
            )
            if not cleaned:
                continue
            if not cleaned.rstrip().endswith((".", "!", "?")):
                cleaned = cleaned.rstrip() + "."
            sentences.append(cleaned)

        if sentences:
            developer_message = " ".join(sentences)

    if not developer_message:
        # Fallback when persona is missing or all traits were empty.
        developer_message = sample_developer_prompt("persona_chat")

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
        content = (
            repair_broken_emoticons(repair_text_punctuation(simple_truecase(hist_msg)))
            if hist_msg
            else hist_msg
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

        response = (
            repair_broken_emoticons(repair_text_punctuation(simple_truecase(response)))
            if response
            else response
        )
        messages.append({"role": "assistant", "content": text_formatter(response)})

    return {
        "messages": messages,
        "metadata": {
            "format": "personachat",
            "source_keys": keys,
            "has_personality": bool(personality),
            "num_utterances": num_utterances,
        },
    }
