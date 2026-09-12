"""Message formatting utilities.

Reply extraction moved to :mod:`praxis.inference.reply` - it is a property of
the chat format and the generation result, not of HTTP, and the streamer needs
it too. Re-exported here so the existing call sites keep working.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from praxis.inference.prompts import apply_standing_prompts
from praxis.inference.reply import (  # noqa: F401  (re-exported)
    EMPTY_REPLY_PLACEHOLDER,
    extract_assistant_reply,
)
from praxis.tokenizers.chat_templates import chat_format_of

api_logger = logging.getLogger("praxis.web")

# Extra seconds to wait for a turn the deadline CUT SHORT. The deadline bounds
# the decode; this bounds how long we wait to collect what it produced, which is
# only as long as the training loop needs to store the result it already has.
PARTIAL_TURN_GRACE = 5.0


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
    on_text: Optional[Callable[[str], None]] = None,
    on_reset: Optional[Callable[[], None]] = None,
    on_tool: Optional[Callable[[str], None]] = None,
    system_prompt: Optional[str] = None,
    developer_prompt: Optional[str] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
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
        on_text: Optional callback receiving the reply as text deltas while it
            is decoded. Purely additive - the return value below is unchanged
            and stays authoritative, so a caller that passes nothing behaves
            exactly as before.
        on_reset: Optional callback meaning "drop what was published so far".
        on_tool: Optional callback receiving the name of each tool the runtime
            runs. NOT recoverable from the return value below - the reply
            extractor strips the tool exchange - so a caller that wants to
            report tool use has to pass this.
        system_prompt: Run-level ``system`` message, prepended unless
            ``messages`` already carries one.
        developer_prompt: Run-level ``developer`` message, same rule. Folded
            into the system message under a format with no ``developer`` role.
        generation_kwargs: Run-level decoding parameters, merged OVER the
            named arguments above so a run can raise ``top_p`` or drop
            ``do_sample`` without every call site growing a parameter. A
            caller that wants to win passes the value here itself.

    Returns:
        Generated assistant reply, or None on failure
    """
    if not messages:
        return None

    # The run's standing instructions, if it has any. A caller's own message of
    # that role wins, which is what makes the web app's editable developer
    # prompt an override rather than a second copy.
    messages = apply_standing_prompts(
        messages,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        chat_format=tokenizer,
    )

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
        # KV caching for incremental decode. The model gates this itself:
        # encoder stacks (CALM) and cache-less attentions fall back to full
        # recompute, so this is safe to leave on.
        "use_cache": True,
        "skip_special_tokens": False,
    }

    if truncate_to is not None:
        kwargs["truncate_to"] = truncate_to

    # The run's decode knobs, last so they win over the defaults above. The
    # route has already folded any per-request override into this dict, so
    # there is exactly one precedence rule and it lives at the call site.
    kwargs.update(generation_kwargs or {})
    # Not a generate() parameter - the deadline below is what bounds the decode.
    timeout = float(kwargs.pop("timeout", timeout))

    # Queue the generation request, with the deadline attached rather than kept
    # here. The wait below is client-side only: the queued path is served inside
    # the training loop (GenerationQueueCallback), so giving up here stops us
    # listening but does NOT stop the run from decoding the whole turn for
    # nobody. The deadline is what actually bounds that.
    deadline = time.time() + timeout
    request_id = generator.request_generation(
        formatted_prompt,
        kwargs,
        deadline=deadline,
        on_text=on_text,
        on_reset=on_reset,
        on_tool=on_tool,
    )

    # Keep listening a little PAST the deadline. The deadline stops the decode
    # (MaxTimeCriteria, per step), and what it stops is a turn the model was
    # part-way through - `_process_single_request` returns that partial turn
    # rather than discarding it. Giving up at exactly the deadline meant nobody
    # ever collected it: the route answered with "" and the client, which had
    # been watching the reply stream in the whole time, replaced it with an
    # error. The grace is short because the result is already decoded by then;
    # all that remains is the training loop storing it.
    while True:
        result = generator.get_result(request_id)
        if result is not None:
            break
        if time.time() > deadline + PARTIAL_TURN_GRACE:
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
    # Validate message roles against the active format's vocabulary, minus the
    # tool-flow roles: those carry runtime-injected content, so accepting them
    # from an API caller would let a client fabricate a tool result.
    fmt = chat_format_of(tokenizer)
    allowed = set(fmt.roles) - {fmt.call_role, fmt.result_role}
    for message in messages:
        role = message.get("role", "").strip()
        if role not in allowed:
            raise ValueError(f"Invalid role: {role}")

    # Apply the chat template and add assistant generation prompt
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
