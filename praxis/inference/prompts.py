"""The run's standing instructions, and the knobs it decodes with.

Two things a run can fix once and have every inference path honor:

``system_prompt`` / ``developer_prompt``
    Messages prepended to a conversation before it reaches the model. The rule
    is one line: **a caller's own message of that role wins**, so the web chat
    can hand back an edited developer prompt and get exactly that, while
    Discord and the raw API get the run's.

``generation_kwargs``
    Decoding parameters merged under whatever the caller sends, so a run can
    say "this model needs a shorter leash" once instead of at every call site.

Where a prompt LANDS depends on the active chat format, which is why this is
not a two-line prepend. ``hf_native`` - a published checkpoint's own template -
declares no ``developer`` role, and emitting one would render a role name the
model has never seen. There the developer prompt folds into the system message
instead, which is the closest thing the format actually has.

The Terminal tab's rolling contexts do NOT come through here. They decode from
their own static arguments against their own temperatures
(``callbacks/lightning/terminal.py``), and that is deliberate: they are a
fixed probe of the model's unguided behaviour, not a chat.
"""

from typing import Any, Dict, List, Optional

from praxis.utils import coerce_to_mapping

# Praxis-side decode options that are not ``GenerationConfig`` fields but are
# understood by the generator and its backend.
PRAXIS_GENERATION_KEYS = frozenset(
    {
        "skip_special_tokens",
        "truncate_to",
        "timeout",
        "use_cache",
    }
)


def parse_generation_kwargs(value: Any) -> Dict[str, Any]:
    """Normalize and validate a ``generation_kwargs`` value.

    Accepts a YAML mapping or a list of ``key=value`` strings (see
    :func:`praxis.utils.coerce_to_mapping`). An unknown key is a hard error:
    ``transformers`` accepts arbitrary kwargs on ``generate`` and quietly
    ignores the ones it does not know, so a typo'd ``temperture`` would
    otherwise decode at the default forever with nothing to show for it.
    """
    kwargs = coerce_to_mapping(value)
    if not kwargs:
        return {}

    from transformers import GenerationConfig

    known = set(GenerationConfig().to_dict()) | set(PRAXIS_GENERATION_KEYS)
    # Valid, and inert unless beams are on. transformers logs its own notice
    # ("generation flags are not valid and may be ignored"), which is easy to
    # lose in a dashboard run - and the setting looks exactly like the fix for
    # runaway replies, so it gets reached for first. Not an error: it is real
    # under beam search.
    if kwargs.get("length_penalty") is not None and not kwargs.get("num_beams"):
        print(
            "[GENERATION] length_penalty only ranks finished beams and does "
            "nothing under sampling. For shorter replies without a hard cap "
            "use exponential_decay_length_penalty=[start, factor]."
        )

    unknown = sorted(key for key in kwargs if key not in known)
    if unknown:
        raise ValueError(
            f"unknown generation kwarg(s): {', '.join(unknown)}. Valid keys are "
            "any transformers GenerationConfig field (max_new_tokens, "
            "temperature, top_p, top_k, min_p, repetition_penalty, "
            "no_repeat_ngram_size, do_sample, ...) plus "
            f"{', '.join(sorted(PRAXIS_GENERATION_KEYS))}."
        )
    return kwargs


def _has_role(messages: List[Dict[str, str]], role: str) -> bool:
    return any(m.get("role") == role for m in messages)


def apply_standing_prompts(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    developer_prompt: Optional[str] = None,
    chat_format: Any = None,
) -> List[Dict[str, str]]:
    """``messages`` with the run's standing instructions prepended.

    Returns a new list; the caller's own ``system``/``developer`` messages are
    left untouched and suppress the matching injection, which is what makes the
    web app's editable developer prompt an override rather than a duplicate.

    ``chat_format`` is the active :class:`~praxis.tokenizers.chat_templates.ChatFormat`
    (or a tokenizer to read it from). A format with no ``developer`` role gets
    the developer text folded into the system message instead of a role its
    template cannot render.
    """
    from praxis.tokenizers.chat_templates import get_chat_format

    if not messages:
        return list(messages)

    fmt = get_chat_format(chat_format)
    system = (system_prompt or "").strip()
    developer = (developer_prompt or "").strip()

    # A format that cannot say "developer" says it as system instead. Joined
    # rather than replaced: both are standing instructions, and dropping one
    # silently is the failure this whole module exists to avoid.
    if developer and "developer" not in fmt.roles:
        system = f"{system}\n\n{developer}".strip() if system else developer
        developer = ""

    preamble: List[Dict[str, str]] = []
    if system and not _has_role(messages, "system"):
        preamble.append({"role": "system", "content": system})
    if developer and not _has_role(messages, "developer"):
        preamble.append({"role": "developer", "content": developer})

    return preamble + list(messages)
