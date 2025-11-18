"""Message formatting utilities."""

from typing import Any, Dict, List


def format_messages_to_chatml(messages: List[Dict[str, str]], tokenizer: Any) -> str:
    """Format a list of message objects using the tokenizer's chat template.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tokenizer: Tokenizer with chat template support

    Returns:
        Formatted string using the chat template

    Raises:
        ValueError: If an invalid role is provided
    """
    # Validate message roles
    for message in messages:
        role = message.get("role", "").strip()
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Invalid role: {role}")

    # Apply the chat template and add assistant generation prompt
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_assistant_reply(generated_text: str, tokenizer: Any) -> str:
    """Extract the assistant's reply from the generated text.

    Args:
        generated_text: Full generated text including tokens
        tokenizer: Tokenizer with special token definitions

    Returns:
        Extracted assistant reply text
    """
    # Find the pattern that marks the start of the assistant's response
    assistant_start = f"{tokenizer.bos_token}assistant"

    # Find the last occurrence of the assistant's start token
    start_index = generated_text.rfind(assistant_start)
    if start_index == -1:
        # If the start token is not found, return the whole text
        return generated_text.strip()

    # Skip past the start token AND the "assistant" role identifier
    start_index += len(assistant_start)

    # Find the end token after the start_index - check for both EOS and SEP tokens
    eos_index = generated_text.find(tokenizer.eos_token, start_index)
    sep_index = generated_text.find(tokenizer.sep_token, start_index)

    # Use whichever comes first (and exists)
    end_index = -1
    if eos_index != -1 and sep_index != -1:
        end_index = min(eos_index, sep_index)
    elif eos_index != -1:
        end_index = eos_index
    elif sep_index != -1:
        end_index = sep_index

    if end_index == -1:
        # If no end token is found, return everything after the start token
        assistant_reply = generated_text[start_index:].strip()
    else:
        assistant_reply = generated_text[start_index:end_index].strip()

    # Remove any remaining BOS token that might appear at the beginning
    if assistant_reply.startswith(tokenizer.bos_token):
        assistant_reply = assistant_reply[len(tokenizer.bos_token) :].strip()

    # Strip '#RESPONSE' prefix from training data if present
    if assistant_reply.startswith("#RESPONSE"):
        # Remove '#RESPONSE' and any following whitespace/newlines
        assistant_reply = assistant_reply[len("#RESPONSE") :].lstrip()

    return assistant_reply
