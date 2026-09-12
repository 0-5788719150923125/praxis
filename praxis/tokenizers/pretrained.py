"""A published tokenizer, loaded as-is, with the two things Praxis needs added.

Nothing here rewrites how the model tokenizes. It resolves two gaps that stop a
published tokenizer from driving a Praxis training run:

1. **A pad token.** Base checkpoints often ship without one (SmolLM2-135M does),
   and the packer needs an id to pad rows with. EOS is the standard stand-in and
   costs nothing, because padded positions are excluded by ``ignore_index``
   rather than by their id.

2. **An assistant mask.** Which needs either ``{% generation %}`` spans in the
   model's own chat template (added by
   :mod:`praxis.tokenizers.generation_spans`, verified byte-identical), or - for
   a base model that declares no template at all - a Praxis format that uses no
   control tokens. ``prose`` is that format: plain-text turn boundaries, halting
   on a trained stop string, and nothing assumed about the vocabulary.
"""

from typing import Any, Optional

from transformers import AutoTokenizer

from praxis.tokenizers.chat_templates import (
    apply_chat_format,
    get_chat_format,
    resolve_chat_format,
)
from praxis.tokenizers.generation_spans import (
    GenerationSpanError,
    add_generation_spans,
)


def _ensure_pad_token(tokenizer) -> None:
    """Give the tokenizer a pad id if it has none."""
    if tokenizer.pad_token_id is not None:
        return
    for candidate in ("eos_token", "unk_token", "bos_token"):
        token = getattr(tokenizer, candidate, None)
        if token:
            tokenizer.pad_token = token
            print(
                f"[TOKENIZER] {candidate} used as the pad token; the checkpoint "
                "declares none."
            )
            return
    raise ValueError(
        "the tokenizer declares no pad, eos, unk or bos token, so there is "
        "nothing to pad a packed batch with"
    )


def load_pretrained_tokenizer(
    model_name: str,
    *,
    chat_format: Any = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load ``model_name``'s tokenizer and bind a usable chat format to it.

    ``chat_format`` is honored when given. Left unset, the choice follows the
    checkpoint: its own template when it has one (``hf_native``), and ``prose``
    when it does not - a base model has no conversational contract of its own,
    and inventing one that needs control tokens its vocabulary lacks would be
    worse than using plain text.
    """
    if revision:
        kwargs.setdefault("revision", revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
    _ensure_pad_token(tokenizer)

    own_template = getattr(tokenizer, "chat_template", None)

    if chat_format is not None:
        fmt = (
            resolve_chat_format(chat_format)
            if isinstance(chat_format, str)
            else chat_format
        )
        if fmt.name == "hf_native" and not own_template:
            raise ValueError(
                f"--chat-format hf_native needs the checkpoint's own chat "
                f"template, and {model_name} declares none. Use 'prose' (plain "
                "text, no control tokens) or leave --chat-format unset."
            )
        apply_chat_format(tokenizer, fmt)
        if fmt.name == "hf_native":
            _install_generation_spans(tokenizer, fmt, model_name)
        return tokenizer

    if own_template:
        fmt = apply_chat_format(tokenizer, "hf_native")
        _install_generation_spans(tokenizer, fmt, model_name)
    else:
        print(
            f"[TOKENIZER] {model_name} declares no chat template; using the "
            "'prose' format (plain-text turn boundaries, no control tokens)."
        )
        apply_chat_format(tokenizer, "prose")
    return tokenizer


def _install_generation_spans(tokenizer, fmt, model_name: str) -> None:
    """Mark the assistant spans in the tokenizer's own template, or fail loud.

    Falling back to a Praxis template here would be the wrong repair: it renders
    text the checkpoint was never trained on. The caller's options are a
    different ``--chat-format`` or a ``model_adapters`` entry, and both are
    decisions, not defaults.
    """
    try:
        tokenizer.chat_template = add_generation_spans(
            tokenizer,
            template=getattr(tokenizer, "chat_template", None),
            generated_roles=fmt.generated_roles,
        )
    except GenerationSpanError as e:
        raise ValueError(
            f"cannot locate the assistant turns in {model_name}'s chat "
            f"template ({e}), so every token would be masked out of the loss "
            "and the run would train on nothing. Pass --chat-format prose to "
            "train on plain-text boundaries instead, or --no-mask-prompts to "
            "train on the whole transcript."
        ) from e
