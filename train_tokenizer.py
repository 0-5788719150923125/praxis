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
    trust_remote_code=True,
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

# Define a ChatML template
# https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chat-markup-language
chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ bos_token }}system
{{ message['content'] }}
{% elif message['role'] == 'user' %}
{{ bos_token }}user
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
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

print(f"A sample ChatML-formatted message:")
print(
    trained_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "The assistant is an AI."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ],
        tokenize=False,
    )
)
