"""Chat templates for Praxis tokenizers."""

# Standard ChatML template with extra developer role
# https://huggingface.co/docs/transformers/en/conversations
# https://huggingface.co/docs/transformers/en/chat_extras
# https://cookbook.openai.com/articles/openai-harmony
#
# All roles use identical formatting: BOS + role + content + SEP
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{{ bos_token }}{{ message['role'] }}
{{ message['content'] }}
{{ sep_token }}
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
