"""Chat templates for Praxis tokenizers."""

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


def get_chat_template(tokenizer_type: str = "default") -> str:
    """
    Get the chat template for a tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer (currently all types use the same template)

    Returns:
        Chat template string
    """
    # All tokenizer types use the same template
    return DEFAULT_CHAT_TEMPLATE
