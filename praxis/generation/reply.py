"""Turning a finished generation into the model's reply.

The runtime writes more than the model does - the prompt, the turn boundaries,
and any tool result it spliced - so "what did the model say" is a real question
with a real answer, and this is where it is answered. One implementation, used
by every consumer: the web routes, the Discord integration, and
:class:`praxis.generation.streamers.ReplyStreamer`, which runs it over the
tokens seen so far to publish a reply while it is still being written.

That last caller is why this is not in the web layer any more. A streaming
reply and a waited-for reply must agree exactly, which only holds while there
is a single extractor for both.
"""

import re
from typing import Any, Optional

from praxis.tokenizers.chat_templates import chat_format_of
from praxis.tools import get_tool_input_pattern, get_tool_output_pattern

__all__ = ["extract_assistant_reply", "EMPTY_REPLY_PLACEHOLDER"]


# Combined regex that matches either a [TOOL_CALL]...[/TOOL_CALL] or a
# [TOOL_RESULT]...[/TOOL_RESULT] block, including any trailing newline so
# stripping doesn't leave a blank line behind.
_TOOL_BLOCK_RE = re.compile(
    rf"(?:{get_tool_input_pattern()}|{get_tool_output_pattern()})\n?",
    re.DOTALL,
)

# Shown when the assistant turn carried no text of its own - the model went
# straight to the next boundary, ended the document, or spoke only in tool
# plumbing. Surfaces the empty case so it doesn't look like a silent success.
# Deliberately does not claim tools ran: under text boundaries the commonest
# way to land here is an empty turn with no tool involved at all.
EMPTY_REPLY_PLACEHOLDER = "(model produced an empty turn)"


def _reply_slice(generated_text: str, fmt, reply_start: Optional[int]) -> Optional[str]:
    """The text of the model's turn, or None when it cannot be located.

    ``reply_start`` is the runtime's own record of where the turn begins (see
    ``GenerationResult``) and is always right when present. The scan below is
    the fallback for callers holding a bare string: it takes the LAST boundary
    opening a reply turn, which is correct for the tool-splice case it was
    written for but wrong whenever the model wrote that boundary itself.
    """
    if reply_start is not None:
        return generated_text[reply_start:]

    start_marker = fmt.boundary(fmt.reply_role)
    start_index = generated_text.rfind(start_marker)
    if start_index == -1:
        # The prompt's own generation cue has no leading blank line when it
        # opens the document; fall back to that shape before giving up.
        bare = start_marker.lstrip("\n")
        start_index = generated_text.rfind(bare)
        if start_index == -1:
            return None
        start_index += len(bare)
    else:
        start_index += len(start_marker)
    return generated_text[start_index:]


def _extract_reply_text_boundaries(
    generated_text: str, fmt, tokenizer: Any, reply_start: Optional[int] = None
) -> str:
    """Reply extraction for text-boundary formats.

    The reply runs from the start of the model's turn to the next boundary of
    any kind, and the tool exchange in between is plumbing the client should
    not see. Everything is ordinary text here, so this is string slicing rather
    than token bookkeeping.

    Two things end the turn, and both have to be cut. A ROLE boundary is the
    normal case, but the generation prompt already ends with the reply role's
    own blank line, so a model that writes nothing and goes straight to naming
    the next speaker emits only ``user\\n\\n`` - the boundary's leading
    ``\\n\\n`` came from the prompt and is not part of the slice. Generation
    halts correctly there (the full sequence does end in the stop string); it
    is only the cut that misses, and without it the bare role word IS what the
    client gets back as the model's answer. A STOP TOKEN is the other case:
    prose keeps the document separator, so a turn can end on ``[EOS]``, and the
    token-boundary branch below cuts at those strings while this one did not.
    """
    reply = _reply_slice(generated_text, fmt, reply_start)
    if reply is None:
        return generated_text.strip()

    # The model restating the boundary the runtime just wrote is noise, not a
    # new turn - the prompt already ends at that boundary, so an immediate
    # repeat adds nothing. Skipping it matters because the cut below treats a
    # reply-role boundary at offset 0 as "this turn is over", which would throw
    # away the reply that follows the repetition.
    opener = fmt.boundary(fmt.reply_role)
    for form in (opener, opener.lstrip("\n")):
        while form and reply.startswith(form):
            reply = reply[len(form) :]

    # Cut at whatever boundary ends the turn. Every role is a candidate,
    # ``assistant`` included: the model can halt on a stop boundary, drift into
    # one we don't halt on, or open a fresh turn of its own - which ends this
    # one just as surely, and used to swallow it instead.
    cut = len(reply)
    for role in fmt.roles:
        boundary = fmt.boundary(role)
        idx = reply.find(boundary)
        if idx != -1:
            cut = min(cut, idx)
        # The seam case: the prompt supplied the boundary's leading blank line,
        # so the model's whole contribution is the bare role line. An empty
        # turn, not a reply of "user". Only when it IS the whole contribution:
        # generation halts at these boundaries, so text after one means the
        # slice is ordinary prose that merely opens with a role word.
        bare = boundary.lstrip("\n")
        if reply.startswith(bare) and not reply[len(bare) :].strip():
            cut = 0
    # Halting on a stop token leaves its text in the slice.
    for token_id in fmt.stop_token_ids(tokenizer):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        idx = reply.find(text) if text else -1
        if idx != -1:
            cut = min(cut, idx)
    return reply[:cut].strip()


def extract_assistant_reply(generated_text: str, tokenizer: Any) -> str:
    """Extract the assistant's reply from the generated text.

    Args:
        generated_text: Full generated text including tokens
        tokenizer: Tokenizer with special token definitions

    Returns:
        Extracted assistant reply text
    """
    fmt = chat_format_of(tokenizer)
    # The runtime's own record of where the model's turn starts, when the text
    # came from the Generator. Absent for a bare string, which falls back to
    # scanning. See GenerationResult.
    reply_start = getattr(generated_text, "reply_start", None)

    if fmt.text_boundaries:
        reply = _extract_reply_text_boundaries(
            generated_text, fmt, tokenizer, reply_start
        )
        return reply or EMPTY_REPLY_PLACEHOLDER

    if reply_start is not None:
        start_index = reply_start
    else:
        # Find the pattern that marks the start of the assistant's response
        assistant_start = f"{tokenizer.bos_token}assistant"

        # Find the last occurrence of the assistant's start token
        start_index = generated_text.rfind(assistant_start)
        if start_index == -1:
            # If the start token is not found, return the whole text
            return generated_text.strip()

        # Skip past the start token AND the "assistant" role identifier
        start_index += len(assistant_start)

    # Same repetition the text-boundary branch skips: the prompt already ends
    # at the turn opener, so the model emitting it again is noise. Without this
    # the BOS cut below lands on offset 0 and reports an empty turn. Only the
    # REPLY role's opener is skipped - `[BOS]user` really does end this turn.
    turn_opener = (
        f"{tokenizer.bos_token}{fmt.reply_role}" if tokenizer.bos_token else ""
    )
    while turn_opener and generated_text.startswith(turn_opener, start_index):
        start_index += len(turn_opener)
        if generated_text[start_index : start_index + 1] == "\n":
            start_index += 1

    # Find the end token after the start_index. EOS and SEP end the turn; a BOS
    # is the model opening the NEXT one, which ends this one too - anchored at
    # the runtime's offset that boundary now falls inside the slice instead of
    # behind it, and cutting there beats returning the plumbing verbatim.
    candidates = [
        generated_text.find(token, start_index)
        for token in (tokenizer.eos_token, tokenizer.sep_token, tokenizer.bos_token)
        if token
    ]
    found = [i for i in candidates if i != -1]
    end_index = min(found) if found else -1

    if end_index == -1:
        # If no end token is found, return everything after the start token
        assistant_reply = generated_text[start_index:].strip()
    else:
        assistant_reply = generated_text[start_index:end_index].strip()

    # Strip '#RESPONSE' prefix from training data if present
    if assistant_reply.startswith("#RESPONSE"):
        # Remove '#RESPONSE' and any following whitespace/newlines
        assistant_reply = assistant_reply[len("#RESPONSE") :].lstrip()

    # Drop inline tool plumbing - chat clients should only see the
    # model's natural-language reply, not the runtime call/result
    # exchange. If that leaves nothing, surface the empty case rather
    # than silently returning ''.
    assistant_reply = _TOOL_BLOCK_RE.sub("", assistant_reply).strip()
    if not assistant_reply:
        assistant_reply = EMPTY_REPLY_PLACEHOLDER

    return assistant_reply
