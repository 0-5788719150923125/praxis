"""Conversations the chat-template tests render, mask and sweep."""

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]

# One character is several byte tokens here, which is where an offset-mapped
# assistant mask slips.
MULTIBYTE = [
    {"role": "system", "content": "\u201cquoted\u201d"},
    {"role": "user", "content": "Calculate √1156 \U0001f600"},
    {"role": "assistant", "content": "\u2014 the answer is 34."},
    {"role": "user", "content": "and été?"},
    {"role": "assistant", "content": "Summer."},
]
