"""Message formatting utilities."""

import logging
import time
from typing import Any, Dict, List, Optional

api_logger = logging.getLogger("praxis.api")


def generate_from_messages(
    messages: List[Dict[str, str]],
    generator: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    repetition_penalty: float = 1.15,
    do_sample: bool = True,
    truncate_to: Optional[int] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Generate a response from a list of messages.

    This is the unified generation function used by both the API routes
    and integrations like Discord.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        generator: Generator instance for inference
        tokenizer: Tokenizer with chat template support
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeated tokens
        do_sample: Whether to use sampling
        truncate_to: Maximum prompt length (truncates from beginning if exceeded)
        timeout: Maximum time to wait for generation (seconds)

    Returns:
        Generated assistant reply, or None on failure
    """
    if not messages:
        return None

    # Format messages using chat template
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        api_logger.error(f"Error formatting messages: {e}")
        formatted_prompt = "\n".join(
            [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages]
        )

    # Generation parameters
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "use_cache": False,
        "skip_special_tokens": False,
    }

    if truncate_to is not None:
        kwargs["truncate_to"] = truncate_to

    # Queue the generation request
    request_id = generator.request_generation(formatted_prompt, kwargs)

    # Wait for result with timeout
    start_time = time.time()
    while True:
        result = generator.get_result(request_id)
        if result is not None:
            break
        if time.time() - start_time > timeout:
            api_logger.error(f"Generation timed out after {timeout}s")
            return None
        time.sleep(0.1)

    if not result:
        return None

    # Extract assistant's reply
    return extract_assistant_reply(result, tokenizer)


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
