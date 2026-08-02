"""Chat formats for Praxis tokenizers.

A chat format is more than a Jinja string. It decides:

- how a turn boundary is written (control tokens vs plain text),
- which roles the model is supervised to produce (the ``{% generation %}``
  blocks that drive ``assistant_mask``),
- how generation HALTS, and
- how a tool call and its result are laid out.

Those four things have to move together. A template whose boundaries the
generator cannot detect produces runaway generation instead of an error, so
the halting contract lives in the format record rather than in the generator.

Two profiles ship:

``default``
    ChatML-with-a-developer-role, unchanged. ``[BOS]role\\ncontent\\n[SEP]\\n``
    per turn; halting on the ``[EOS]``/``[SEP]`` ids; tool calls wrapped in
    the atomic ``[TOOL_CALL]``/``[TOOL_RESULT]`` control tokens.

``prose``
    No control tokens anywhere. A turn boundary is the role name alone on its
    own line, blank-line separated (``\\n\\nuser\\n\\n``), and halting is by
    stop STRING. Two things motivate it, both measured on the byte-latent
    stack (see ``experiments/abstractinator-g.yml``):

    1. Under ``default``, ``[BOS]`` occupies ~3% of input positions and 0% of
       gradient targets - it sits outside every ``{% generation %}`` block, so
       the model conditions on a symbol it is structurally forbidden from
       producing. ``prose`` puts the boundary that ENDS a turn inside that
       turn's generation block, so the model is trained to terminate itself.
    2. The byte-latent space patcher cuts unconditionally on ids below
       ``OFFSET`` (``find_space_patch_start_ids``), so every control token buys
       its own patch. A ``[SEP]``/``[BOS]`` pair spends two patch codes on
       markers carrying no content and splits the role word into a third
       patch; ``\\n\\nuser\\n\\n`` is one patch that carries the boundary AND
       the role together.

Turn layout in ``prose`` is "each turn owns the boundary that names the next
speaker". Rendering three messages gives::

    system\\n\\n<system text>\\n\\nuser\\n\\n<user text>\\n\\nassistant\\n\\n<reply>\\n\\n

so the assistant's ``{% generation %}`` block covers its reply *plus* the
``\\n\\nuser\\n\\n`` that ends it. That trailing boundary is the halt signal,
and it is a trained target - which is the whole point.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Standard ChatML template with extra developer role.
# https://huggingface.co/docs/transformers/en/conversations
# https://huggingface.co/docs/transformers/en/chat_extras
# https://cookbook.openai.com/articles/openai-harmony
#
# All roles use identical formatting: BOS + role + content + SEP.
#
# When `omit_leading_bos` is passed and truthy, the first message's leading
# BOS is skipped. The packer uses this to drop the redundant BOS at doc-to-doc
# boundaries that fall mid-sequence, so every sequence's position 0 is a real
# BOS + role transition (matching inference) without discarding tokens.
#
# Assistant content + SEP is wrapped in {% generation %} markers so the
# tokenizer's `return_assistant_tokens_mask=True` returns a per-token mask
# marking which positions belong to the assistant's turn. The training
# loss always uses this mask to skip prompt / template / role-marker
# tokens. The markers emit no text -- output is byte-identical to a
# template without them.
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{% if not (loop.first and (omit_leading_bos is defined) and omit_leading_bos) %}{{ bos_token }}{% endif %}{{ message['role'] }}
{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}
{{ sep_token }}
{% endgeneration %}{% else %}{{ message['content'] }}
{{ sep_token }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}"""


# Plain-text boundaries. `omit_leading_bos` is deliberately ignored: in this
# format the boundary IS the separator, so dropping it at a doc-to-doc
# boundary would run the next document's role name into the previous
# document's last word.
#
# The tail of each turn names the next speaker, and for a generated role that
# tail lives INSIDE the {% generation %} block. The last message's tail is the
# bare blank line: whatever document the packer appends next opens with its own
# role name, so the packed stream still reads `\n\n<role>\n\n` at the seam.
PROSE_CHAT_TEMPLATE = (
    "{% if messages %}{{ messages[0]['role'] }}{{ '\\n\\n' }}{% endif %}"
    "{% for message in messages %}"
    "{% set tail %}"
    "{% if loop.last %}{{ '\\n\\n' }}"
    "{% else %}{{ '\\n\\n' }}{{ messages[loop.index]['role'] }}{{ '\\n\\n' }}{% endif %}"
    "{% endset %}"
    "{% if message['role'] in ['assistant', 'call'] %}"
    "{% generation %}{{ message['content'] }}{{ tail }}{% endgeneration %}"
    "{% else %}{{ message['content'] }}{{ tail }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant{{ '\\n\\n' }}{% endif %}"
)


@dataclass(frozen=True)
class ChatFormat:
    """A chat format and everything that has to agree with it.

    Attributes:
        name: Registry key.
        template: Jinja chat template.
        roles: Every role the format knows. Used to validate rendered data
            and to bound the set of text boundaries we scan for.
        generated_roles: Roles wrapped in ``{% generation %}`` - the positions
            the training loss keeps (see ``assistant_mask``).
        boundary_style: ``"tokens"`` (control-token turn markers) or
            ``"text"`` (role name on its own line).
        stop_token_names: Tokenizer attributes (``"eos_token_id"`` style)
            whose ids halt generation. Empty for text boundaries.
        stop_roles: Roles whose boundary ends the model's turn. Text style
            only; these become stop strings.
        tool_style: ``"tokens"`` (atomic ``[TOOL_CALL]`` markers inside an
            assistant turn) or ``"roles"`` (the call and its result are turns
            in their own right).
        call_role: Role carrying the tool-call body. ``tool_style="roles"``
            only; the boundary opening it switches decoding to greedy.
        result_role: Role carrying the tool result. Its boundary is where the
            runtime intercepts, executes, and splices the real result.
        reply_role: Role the runtime hands control back to after a tool result.
        document_separator: Tokenizer attribute naming the id the packer writes
            at the end of each document. ``None`` writes nothing. See
            ``MessageQueueManager._tokenize_doc`` for why this is not optional
            in practice.
    """

    name: str
    template: str
    roles: Tuple[str, ...]
    generated_roles: Tuple[str, ...]
    boundary_style: str
    stop_token_names: Tuple[str, ...] = ()
    stop_roles: Tuple[str, ...] = ()
    tool_style: str = "tokens"
    call_role: str = "call"
    result_role: str = "tool"
    reply_role: str = "assistant"
    document_separator: Optional[str] = "eos_token_id"

    @property
    def text_boundaries(self) -> bool:
        """True when turns are delimited by text rather than control tokens."""
        return self.boundary_style == "text"

    @property
    def uses_tool_tokens(self) -> bool:
        """Whether rendered text ever contains the atomic tool-control tokens.

        Tokenizers consult this before REGISTERING those tokens, because an id
        the data never contains is still an id the output head can sample. The
        byte-level head is exactly ``byte_alphabet_size`` wide (264 with the
        tool tokens), so under ``prose`` four of those logits were reachable by
        sampling and unreachable by training - which is why generations came
        back with ``[TOOL_CALL]`` in them.
        """
        return self.tool_style == "tokens"

    def render_segments(
        self,
        messages: List[Dict[str, str]],
        tokenizer,
        add_generation_prompt: bool = False,
        omit_leading_bos: bool = False,
    ) -> List[Tuple[str, bool]]:
        """The document as ordered ``(text, is_generated)`` segments.

        This exists because ``return_assistant_tokens_mask`` cannot be trusted
        here. HuggingFace implements it by recording CHARACTER offsets of the
        ``{% generation %}`` spans and mapping them through the tokenizer's
        offset table - which assumes a character maps to a bounded, tracked span
        of tokens. Under a byte-level tokenizer one non-ASCII character is
        several tokens, and the mapping slips: measured on the byte tokenizer,
        every multi-byte character before a span shifted the ``prose`` mask two
        tokens (cumulatively), and a multi-byte character at the start of a span
        shifted the ``default`` mask by its byte length. The result was a
        silently misaligned prompt-loss mask on any text containing a curly
        quote, an accent, an em dash or an emoji - it trained on some prompt
        tokens and skipped some assistant ones.

        Tokenizing segment by segment and concatenating sidesteps offsets
        entirely: each segment's own token count IS its span. The join must be
        byte-identical to what the Jinja template renders, which
        ``tests/test_chat_formats.py`` asserts for both formats.
        """
        segments: List[Tuple[str, bool]] = []
        if not messages:
            return segments

        if self.text_boundaries:
            # Boundary opens the document, then each turn owns the tail that
            # names the next speaker (the tail is INSIDE a generated turn's
            # span, which is what makes the halt signal a trained target).
            segments.append((f"{messages[0].get('role', '')}\n\n", False))
            last = len(messages) - 1
            for i, message in enumerate(messages):
                role = message.get("role", "")
                tail = (
                    "\n\n"
                    if i == last
                    else f"\n\n{messages[i + 1].get('role', '')}\n\n"
                )
                text = f"{message.get('content', '')}{tail}"
                segments.append((text, role in self.generated_roles))
            if add_generation_prompt:
                segments.append((f"{self.reply_role}\n\n", False))
            return segments

        bos = getattr(tokenizer, "bos_token", "") or ""
        sep = getattr(tokenizer, "sep_token", "") or ""
        for i, message in enumerate(messages):
            role = message.get("role", "")
            opener = "" if (i == 0 and omit_leading_bos) else bos
            segments.append((f"{opener}{role}\n", False))
            body = f"{message.get('content', '')}\n{sep}\n"
            segments.append((body, role in self.generated_roles))
        if add_generation_prompt:
            segments.append((f"{bos}{self.reply_role}\n", False))
        return segments

    def boundary(self, role: str) -> str:
        """The text that opens ``role``'s turn.

        Empty for token-boundary formats, whose openers need tokenizer-specific
        ids and are built by the tool/splice helpers instead.
        """
        if not self.text_boundaries:
            return ""
        return f"\n\n{role}\n\n"

    def stop_strings(self) -> Tuple[str, ...]:
        """Strings whose appearance ends a generation step.

        ``reply_role`` is deliberately excluded even though a spontaneous
        transition to it would also end a turn: the generation prompt and the
        post-tool splice both END with that boundary, so treating it as a stop
        string would halt every resumed step before it produced a token.
        """
        if not self.text_boundaries:
            return ()
        return tuple(self.boundary(role) for role in self.stop_roles)

    def stop_token_ids(self, tokenizer) -> List[int]:
        """Halt ids resolved against ``tokenizer``, in declaration order."""
        ids: List[int] = []
        for attr in self.stop_token_names:
            tid = getattr(tokenizer, attr, None)
            if tid is not None and int(tid) not in ids:
                ids.append(int(tid))
        return ids

    def document_separator_id(self, tokenizer) -> Optional[int]:
        """Id the packer appends after each document, or ``None``."""
        if not self.document_separator:
            return None
        tid = getattr(tokenizer, self.document_separator, None)
        return None if tid is None else int(tid)

    # Named control ids that could conceivably reach the output head. Bytes
    # start at OFFSET=4 in the byte-latent layout, so all four exist whether or
    # not a format uses them - the ids below OFFSET are structural (the space
    # patcher force-cuts on ``tokens < OFFSET``) and renumbering them would
    # move every byte.
    _NAMED_CONTROL_IDS = (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "sep_token_id",
    )

    def produced_token_names(self) -> Tuple[str, ...]:
        """Control tokens the training data actually makes a target."""
        produced = {self.document_separator} if self.document_separator else set()
        if not self.text_boundaries:
            produced |= {"bos_token_id", "sep_token_id"}
        return tuple(n for n in self._NAMED_CONTROL_IDS if n in produced)

    def suppressed_token_ids(self, tokenizer) -> List[int]:
        """Control ids to keep out of samples (``generate(suppress_tokens=)``).

        Conditioning on a symbol the loss never makes a target is one thing;
        SAMPLING it is another. Under ``prose`` the turn boundaries are plain
        text, so ``[BOS]`` and ``[SEP]`` are never targets, yet they still hold
        ids 1 and 3 - and two untrained logits out of 260 are enough for a
        generation to come back with a bracketed token in it. ``[PAD]`` is in
        the same position under both formats: the packer zero-fills the tail of
        an under-filled sequence and the mask zeroes it out.

        The document separator is deliberately NOT suppressed: the packer
        writes it and the model is trained to produce it, so emitting it is the
        model correctly ending a document.
        """
        produced = set(self.produced_token_names())
        ids: List[int] = []
        for name in self._NAMED_CONTROL_IDS:
            if name in produced:
                continue
            tid = getattr(tokenizer, name, None)
            if tid is not None and int(tid) not in ids:
                ids.append(int(tid))
        return ids

    def tool_call_messages(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        reply: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """The message sequence for one call/result[/reply] exchange.

        The two styles differ in message STRUCTURE, not just rendering, so
        training-data builders have to ask the format rather than hard-code a
        layout. ``tokens`` keeps the call inline in an assistant turn wrapped
        in control tokens; ``roles`` promotes the call and the result to turns
        of their own, which is what makes their boundaries trainable.

        ``reply`` is optional: some samples exercise a call purely to show the
        result (the ``get_tools`` probe) and never speak afterwards.

        Under ``roles`` the exchange opens with an EMPTY ``reply_role`` turn,
        which is what makes the call reachable at all. Two things force it:

        1. Every boundary is the tail of the turn BEFORE it, and a tail is
           supervised only when that turn is in ``generated_roles``. Running
           ``user`` straight into ``call`` puts ``\\n\\ncall\\n\\n`` in the user
           turn's tail, so the call's own opening boundary is never a training
           target - the model can continue a call it was handed but can never
           decide to open one. That is exactly the defect this format exists to
           remove, relocated from ``[BOS]`` to the call boundary.
        2. Inference starts from ``add_generation_prompt=True``, so the prompt
           ends at ``reply_role``'s boundary. A layout with no ``assistant``
           turn before the call never shows the model that position, so even a
           supervised boundary would be one it had to reach from a context it
           had never seen.

        An empty turn fixes both: the tail lands inside a generated span, and
        the rendered ``...assistant\\n\\n\\n\\ncall\\n\\n`` is precisely the
        continuation inference asks for. The content is empty deliberately - a
        fixed lead-in phrase ("Let me calculate that") would be the single
        most-repeated string in the tool corpus and the easiest thing in it to
        memorize, which is the same reason the ``get_tools`` probe is rare.
        """
        from praxis.tools.tags import (
            format_call_body,
            format_tool_input,
            format_tool_output,
        )

        if self.tool_style == "roles":
            messages = [
                {"role": self.reply_role, "content": ""},
                {
                    "role": self.call_role,
                    "content": format_call_body(tool_name, arguments),
                },
                {"role": self.result_role, "content": str(result)},
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": format_tool_input(tool_name, arguments),
                },
                {"role": self.result_role, "content": format_tool_output(result)},
            ]
        if reply is not None:
            messages.append({"role": self.reply_role, "content": reply})
        return messages


DEFAULT_FORMAT = ChatFormat(
    name="default",
    template=DEFAULT_CHAT_TEMPLATE,
    roles=("system", "developer", "user", "assistant", "tool"),
    generated_roles=("assistant",),
    boundary_style="tokens",
    stop_token_names=("eos_token_id", "sep_token_id"),
    tool_style="tokens",
)

PROSE_FORMAT = ChatFormat(
    name="prose",
    template=PROSE_CHAT_TEMPLATE,
    roles=("system", "developer", "user", "assistant", "call", "tool"),
    # `call` is model-produced, so it must be supervised - otherwise the model
    # inherits exactly the defect this format exists to remove.
    generated_roles=("assistant", "call"),
    boundary_style="text",
    # The TEMPLATE emits no control token, but the PACKER terminates every
    # document with `document_separator` - so [EOS] does occur in the data, as
    # the one control token this format keeps, and halting on it is honest
    # rather than dead weight. The stop STRINGS below remain the primary halt:
    # a turn normally ends by naming the next speaker, and only the last turn
    # of a document reaches the separator.
    stop_token_names=("eos_token_id",),
    stop_roles=("user", "system", "developer", "call", "tool"),
    tool_style="roles",
)

CHAT_FORMAT_REGISTRY: Dict[str, ChatFormat] = {
    "default": DEFAULT_FORMAT,
    "prose": PROSE_FORMAT,
}


def resolve_chat_format(name: Optional[str]) -> ChatFormat:
    """Look up a format by registry key. Unknown keys are a hard error.

    Use this on the config path, where a typo should fail the run rather than
    silently train on a different format than the one requested.
    """
    if name is None:
        return DEFAULT_FORMAT
    try:
        return CHAT_FORMAT_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown chat_format={name!r}. "
            f"Valid choices: {sorted(CHAT_FORMAT_REGISTRY)}"
        ) from None


def get_chat_format(tokenizer_or_name: Any = None) -> ChatFormat:
    """Best-effort format resolution from a tokenizer, a name, or nothing.

    Lenient by design: every call site that predates the registry passes a
    tokenizer TYPE (``"byte_level"``, ``"bpe"``), which is not a format name.
    Those keep getting the default.
    """
    if tokenizer_or_name is None:
        return DEFAULT_FORMAT
    if isinstance(tokenizer_or_name, ChatFormat):
        return tokenizer_or_name
    if isinstance(tokenizer_or_name, str):
        return CHAT_FORMAT_REGISTRY.get(tokenizer_or_name, DEFAULT_FORMAT)
    fmt = getattr(tokenizer_or_name, "chat_format", None)
    if isinstance(fmt, ChatFormat):
        return fmt
    if isinstance(fmt, str):
        return CHAT_FORMAT_REGISTRY.get(fmt, DEFAULT_FORMAT)
    # `chat_format` is a plain attribute, so it does NOT survive
    # save_pretrained/from_pretrained - but `chat_template` does. Recover the
    # format from the template rather than silently pairing a prose template
    # with the default halting contract, which would never terminate.
    template = getattr(tokenizer_or_name, "chat_template", None)
    if isinstance(template, str):
        for candidate in CHAT_FORMAT_REGISTRY.values():
            if candidate.template == template:
                return candidate
    return DEFAULT_FORMAT


def tokenize_with_mask(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    omit_leading_bos: bool = False,
) -> Optional[Tuple[List[int], List[int]]]:
    """``(ids, assistant_mask)`` built segment-wise, or None if unsupported.

    Returns None unless the tokenizer declares ``context_free_tokenization`` -
    that piece-wise encoding concatenates to the same ids as encoding the whole
    string. Byte- and char-level tokenizers satisfy it (no merges, so a token
    never spans a segment boundary); BPE and unigram do NOT, because a merge can
    straddle the boundary and piece-wise encoding would silently change the
    tokenization. Those keep HuggingFace's offset-based mask, which is correct
    for them precisely because their characters map to tokens cleanly.

    So this is not a general replacement - it is the path for the tokenizers
    whose character-to-token map is not 1:1, which is exactly where the offset
    mask breaks. See ``ChatFormat.render_segments``.
    """
    if not getattr(tokenizer, "context_free_tokenization", False):
        return None
    fmt = get_chat_format(tokenizer)
    segments = fmt.render_segments(
        messages,
        tokenizer,
        add_generation_prompt=add_generation_prompt,
        omit_leading_bos=omit_leading_bos,
    )
    ids: List[int] = []
    mask: List[int] = []
    for text, generated in segments:
        piece = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(piece)
        mask.extend([1 if generated else 0] * len(piece))
    return ids, mask


def apply_chat_format(tokenizer, name_or_format: Any = None) -> ChatFormat:
    """Bind a format to ``tokenizer``, setting both halves at once.

    The template and the format record have to agree, so this is the only
    supported way to change a tokenizer's chat format - assigning
    ``chat_template`` alone leaves the halting contract pointing at the old
    boundaries.
    """
    fmt = (
        name_or_format
        if isinstance(name_or_format, ChatFormat)
        else resolve_chat_format(name_or_format)
    )
    tokenizer.chat_format = fmt
    tokenizer.chat_template = fmt.template
    return fmt


# Alias kept for readability at call sites that pass a tokenizer.
chat_format_of = get_chat_format


def get_chat_template(tokenizer_type: str = "default") -> str:
    """Get the chat template string for a tokenizer type or format name.

    Kept for the pre-registry call sites, which pass a tokenizer type rather
    than a format name; those resolve to the default template as before.
    """
    return get_chat_format(tokenizer_type).template
