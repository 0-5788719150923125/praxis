"""Chat templates for Praxis tokenizers."""

# Default ChatML template with tool support and developer role
# https://huggingface.co/docs/transformers/en/conversations
# https://huggingface.co/docs/transformers/en/chat_extras
# https://cookbook.openai.com/articles/openai-harmony
DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ bos_token }}system
{{ message['content'] }}
{{ sep_token }}
{% elif message['role'] == 'developer' %}
{{ bos_token }}developer
{{ message['content'] }}
{{ sep_token }}
{% elif message['role'] == 'user' %}
{{ bos_token }}user
{{ message['content'] }}
{{ sep_token }}
{% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{{ message['content'] }}
{% if message.tool_calls is defined %}
{% for tool_call in message.tool_calls %}
<tool_call>
{"name": "{{ tool_call.function.name }}", "arguments": {{ tool_call.function.arguments | tojson }}}
</tool_call>
{% endfor %}
{% endif %}
{{ sep_token }}
{% elif message['role'] == 'tool' %}
{{ bos_token }}tool
{{ message['content'] }}
{{ sep_token }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}"""

# ByteLevel tokenizer template (same as default for now)
BYTE_LEVEL_CHAT_TEMPLATE = DEFAULT_CHAT_TEMPLATE

# Standard tokenizer template (same as default)
STANDARD_CHAT_TEMPLATE = DEFAULT_CHAT_TEMPLATE

# Export the default template
CHAT_TEMPLATE = DEFAULT_CHAT_TEMPLATE


def get_chat_template(tokenizer_type: str = "default") -> str:
    """
    Get the appropriate chat template for a tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer ("byte_level", "standard", or "default")

    Returns:
        Chat template string
    """
    templates = {
        "default": DEFAULT_CHAT_TEMPLATE,
        "byte_level": BYTE_LEVEL_CHAT_TEMPLATE,
        "byte": BYTE_LEVEL_CHAT_TEMPLATE,
        "standard": STANDARD_CHAT_TEMPLATE,
        "bpe": STANDARD_CHAT_TEMPLATE,
        "unigram": STANDARD_CHAT_TEMPLATE,
    }

    return templates.get(tokenizer_type, DEFAULT_CHAT_TEMPLATE)
