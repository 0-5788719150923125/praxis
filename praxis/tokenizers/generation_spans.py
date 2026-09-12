"""Teaching a foreign chat template to report its assistant spans.

HuggingFace derives ``assistant_mask`` from ``{% generation %}`` blocks in the
chat template, and almost no published template has them - the feature came
after most templates were written. Without the mask the prompt-loss mask is all
zeros, ``_build_loss_weights`` multiplies the weights by zero, and
``weighted_reduce`` returns ``0.0`` *with the autograd graph intact*: a run that
trains on nothing and never raises.

So the template is rewritten rather than replaced. The transform wraps a chat
loop's body in a role test::

    {% for message in messages %}BODY{% endfor %}

    {% for message in messages %}
      {% if message['role'] == 'assistant' %}{% generation %}BODY{% endgeneration %}
      {% else %}BODY{% endif %}
    {% endfor %}

which emits exactly the same text - the markers render nothing - so the model
still sees the format it was trained on, byte for byte. That equality is
CHECKED, not assumed (:func:`add_generation_spans` re-renders a probe
conversation and compares), because a template that silently drifted would
train the model on a format it has never seen.
"""

import re
from typing import Iterable, List, Optional, Tuple

# `{% for <var> in <expr> %}` / `{% endfor %}`, with Jinja's whitespace-control
# dashes optional on either side.
_FOR = re.compile(r"\{%-?\s*for\s+(\w+)\s+in\s+([^%]+?)\s*-?%\}")
_ENDFOR = re.compile(r"\{%-?\s*endfor\s*-?%\}")

PROBE_MESSAGES = [
    {"role": "user", "content": "PRAXIS_PROBE_USER"},
    {"role": "assistant", "content": "PRAXIS_PROBE_ASSISTANT"},
]


class GenerationSpanError(RuntimeError):
    """The template could not be taught to report assistant spans."""


def has_generation_spans(template: Optional[str]) -> bool:
    return bool(template) and "{% generation %}" in template.replace("{%-", "{%")


def _outermost_message_loop(template: str) -> Optional[Tuple[int, int, int, int, str]]:
    """``(for_start, body_start, body_end, end, loop_var)`` for the outermost
    ``for`` loop that iterates messages, or None.

    "Iterates messages" is judged by the body mentioning the loop variable's
    ``role``, which is what makes it a chat loop rather than, say, a loop over
    tool definitions.
    """
    for match in _FOR.finditer(template):
        loop_var = match.group(1)
        depth = 1
        cursor = match.end()
        while depth:
            nxt_for = _FOR.search(template, cursor)
            nxt_end = _ENDFOR.search(template, cursor)
            if nxt_end is None:
                return None
            if nxt_for is not None and nxt_for.start() < nxt_end.start():
                depth += 1
                cursor = nxt_for.end()
                continue
            depth -= 1
            cursor = nxt_end.end()
            if depth == 0:
                body = template[match.end() : nxt_end.start()]
                if re.search(rf"{loop_var}(\[['\"]role['\"]\]|\.role)", body):
                    return (
                        match.start(),
                        match.end(),
                        nxt_end.start(),
                        nxt_end.end(),
                        loop_var,
                    )
    return None


def rewrite_template(template: str, generated_roles: Iterable[str]) -> str:
    """The template with its chat loop's body wrapped in generation spans."""
    found = _outermost_message_loop(template)
    if found is None:
        raise GenerationSpanError(
            "no message loop found: the template has no `{% for ... %}` whose "
            "body reads a message role, so there is nothing to wrap"
        )
    _, body_start, body_end, _, loop_var = found
    body = template[body_start:body_end]
    roles = list(generated_roles) or ["assistant"]
    test = " or ".join(f"{loop_var}['role'] == '{role}'" for role in roles)
    wrapped = (
        "{% if " + test + " %}"
        "{% generation %}" + body + "{% endgeneration %}"
        "{% else %}" + body + "{% endif %}"
    )
    return template[:body_start] + wrapped + template[body_end:]


def _render(tokenizer, template: str, messages) -> str:
    return tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, add_generation_prompt=False
    )


def add_generation_spans(
    tokenizer,
    template: Optional[str] = None,
    generated_roles: Iterable[str] = ("assistant",),
    probe: Optional[List[dict]] = None,
) -> str:
    """``template`` with assistant spans marked, verified against the original.

    Returns the template unchanged when it already has them. Raises
    :class:`GenerationSpanError` when the rewrite cannot be made or does not
    reproduce the original output exactly - never a template that silently
    renders something else.
    """
    template = template or getattr(tokenizer, "chat_template", None)
    if not template:
        raise GenerationSpanError("tokenizer declares no chat template")
    if has_generation_spans(template):
        return template

    rewritten = rewrite_template(template, generated_roles)
    messages = probe or PROBE_MESSAGES

    try:
        before = _render(tokenizer, template, messages)
        after = _render(tokenizer, rewritten, messages)
    except Exception as e:  # a template that cannot render is not our bug to fix
        raise GenerationSpanError(f"template failed to render: {e}") from e

    if before != after:
        raise GenerationSpanError(
            "the rewrite changed the rendered text, which would train the model "
            "on a format it has never seen"
        )

    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=rewritten,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    mask = encoded.get("assistant_masks")
    if isinstance(mask, list) and mask and isinstance(mask[0], list):
        mask = mask[0]
    if not mask or not any(mask):
        raise GenerationSpanError(
            "the rewritten template still reports no assistant tokens"
        )
    if all(mask):
        raise GenerationSpanError(
            "the rewritten template marks every token as assistant, so the "
            "prompt would be trained on as well"
        )
    return rewritten
