import os
from itertools import islice
from typing import List, Union

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

seed = 59
save_path = "data/praxis"

vocab_size = 8192
dropout = 0.1
min_frequency = 3
max_token_length = 7

pad_token = "[PAD]"
bos_token = "[BOS]"
eos_token = "[EOS]"
unk_token = "[UNK]"

num_examples = 10_000_000

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
    cache_dir="./tmp/pile",
    trust_remote_code=True,
).shuffle(
    seed=seed,
    buffer_size=100_000,
)

column = "text"
iterator = islice((item[key] for item in dataset), num_examples)

tokenizer = Tokenizer(
    models.BPE(
        dropout=dropout,
        unk_token=unk_token,
        byte_fallback=True,
        fuse_unk=True,
        cache_capacity=65536,
    )
)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    max_token_length=max_token_length,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=[
        unk_token,
        pad_token,
        bos_token,
        eos_token,
    ],
)

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Punctuation(behavior="isolated"),
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False, trim_offsets=True, use_regex=True
        ),
    ]
)

tokenizer.normalizer = normalizers.NFD()

tokenizer.decoder = decoders.ByteLevel(
    add_prefix_space=True, trim_offsets=True, use_regex=True
)

tokenizer.post_processor = processors.ByteLevel(
    add_prefix_space=True, trim_offsets=True, use_regex=True
)

tokenizer.train_from_iterator(iterator=iterator, trainer=trainer, length=num_examples)

trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

trained_tokenizer.add_special_tokens(
    {
        "unk_token": unk_token,
        "pad_token": pad_token,
        "bos_token": bos_token,
        "eos_token": eos_token,
    }
)

os.makedirs(save_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
