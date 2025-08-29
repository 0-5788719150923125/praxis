import argparse
import os
from itertools import islice

from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    choices=["bpe", "unigram"],
    default="unigram",
    help="The type of tokenizer to train",
)
parser.add_argument(
    "--num-examples",
    type=int,
    default=5_000_000,
    help="The number of examples to train",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    default=16384,
    help="The absolute vocab size to use",
)

args = parser.parse_args()

tokenizer_type = args.type
num_examples = args.num_examples
vocab_size = args.vocab_size

dropout = 0.1

save_path = "data/praxis"
archive_path = f"{save_path}-{vocab_size}-{tokenizer_type}"

special_tokens = {
    "pad_token": "[PAD]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "sep_token": "[SEP]",
    # "mask_token": "[MASK]",
}
additional_special_tokens = []

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-350BT",
    split="train",
    streaming=True,
    cache_dir="data/datasets",
).shuffle(
    seed=42,
    buffer_size=10_000,
)

key = "text"
iterator = islice((item[key] for item in dataset), num_examples)

all_special_tokens = list(
    set(list(special_tokens.values()) + additional_special_tokens)
)

trainer_args = dict(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
    special_tokens=all_special_tokens,
)

if tokenizer_type == "bpe":
    tokenizer = Tokenizer(models.BPE(dropout=dropout, byte_fallback=True))
    trainer = trainers.BpeTrainer(**trainer_args)
elif tokenizer_type == "unigram":
    tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
    trainer = trainers.UnigramTrainer(
        shrinking_factor=0.75, max_piece_length=16, n_sub_iterations=8, **trainer_args
    )

tokenizer.add_special_tokens(all_special_tokens)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()

tokenizer.train_from_iterator(iterator=iterator, trainer=trainer, length=num_examples)

trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
trained_tokenizer.add_special_tokens(
    {**special_tokens, "additional_special_tokens": additional_special_tokens}
)

# Define a ChatML template with tool support and developer role
# https://huggingface.co/docs/transformers/en/conversations
# https://huggingface.co/docs/transformers/en/chat_extras
# Tools are now rendered in system/developer blocks for persistence (like Harmony)
chat_template = """{% if tools is defined and tools %}
{% set tools_json = tools | tojson %}
{% endif %}
{% for message in messages %}
{% if message['role'] == 'system' %}
{{ bos_token }}system
{{ message['content'] }}
{% if tools_json is defined and loop.first %}
Available tools:
{{ tools_json }}
{% endif %}
{% elif message['role'] == 'developer' %}
{{ bos_token }}developer
{{ message['content'] }}
{% if tools_json is defined and loop.first and not (messages[0]['role'] == 'system') %}
Available tools:
{{ tools_json }}
{% endif %}
{% elif message['role'] == 'user' %}
{{ bos_token }}user
{{ message['content'] }}
{% elif message['role'] == 'tool' %}
{{ bos_token }}tool
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{% if message.tool_calls is defined %}
{% for tool_call in message.tool_calls %}
<tool_call>
{"name": "{{ tool_call.function.name }}", "arguments": {{ tool_call.function.arguments | tojson }}}
</tool_call>
{% endfor %}
{% endif %}
{{ message['content'] }}
{% endif %}
{{ sep_token }}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}"""


trained_tokenizer.chat_template = chat_template

os.makedirs(save_path, exist_ok=True)
os.makedirs(archive_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
trained_tokenizer.save_pretrained(archive_path)

print(f"\nSample ChatML-formatted messages demonstrating the unified format:")

# Test both regular conversation and tool usage
from praxis.tools import call_tool, get_tools_json_schema

tools = get_tools_json_schema()
result = call_tool("calc", {"values": [25, 17], "op": "add"})

# Coherent multi-turn conversation demonstrating all components
print(
    trained_tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation.",
            },
            {
                "role": "developer",
                "content": "Engage in a scientific discussion, using tools when needed for calculations.",
            },
            {
                "role": "user",
                "content": "I'm studying quantum mechanics. Can you explain superposition?",
            },
            {
                "role": "assistant",
                "content": "Superposition is a fundamental principle where a quantum system exists in multiple states simultaneously until measured. Think of Schrödinger's cat - before observation, it's both alive and dead. When we measure the system, it 'collapses' into one definite state.",
            },
            {
                "role": "user",
                "content": "Interesting! If I have a quantum computer with 25 qubits, and I add 17 more, how many total qubits would I have?",
            },
            {
                "role": "assistant",
                "content": "Let me calculate that for you.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": {"values": [25, 17], "op": "add"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": str(result)},
            {
                "role": "assistant",
                "content": f"You would have {result} qubits total. With 42 qubits, your quantum computer could theoretically represent 2^42 (about 4.4 trillion) different states simultaneously - that's the power of quantum superposition at scale!",
            },
            {
                "role": "user",
                "content": "That's incredible computational power!",
            },
            {
                "role": "assistant",
                "content": "Indeed! To put it in perspective, while a classical 42-bit computer can only be in one of those 4.4 trillion states at any given time, a 42-qubit quantum computer can explore all of them simultaneously through superposition. This is why quantum computers excel at certain problems like cryptography and optimization.",
            },
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
)
