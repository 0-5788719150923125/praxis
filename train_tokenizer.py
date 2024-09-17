import os
from itertools import islice
from typing import List, Union

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

save_path = "data/praxis"

vocab_size = 8192
dropout = 0.1
min_frequency = 2
max_token_length = 5

pad_token = "[PAD]"
bos_token = "[BOS]"
eos_token = "[EOS]"
unk_token = "[UNK]"

train_steps = 10_000_000

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    split="train",
    streaming=True,
    cache_dir="./tmp/pile",
    trust_remote_code=True,
).shuffle(
    seed=42,
    buffer_size=100_000,
)

key = "text"

data = islice((item[key] for item in dataset), train_steps)

tokenizer = Tokenizer(
    models.BPE(
        dropout=dropout,
        byte_fallback=True,
        unk_token=unk_token,
        fuse_unk=True,
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
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False, trim_offsets=True, use_regex=True
        ),
    ]
)

tokenizer.decoder = decoders.ByteLevel(
    add_prefix_space=True, trim_offsets=True, use_regex=True
)

tokenizer.post_processor = processors.ByteLevel(
    add_prefix_space=True, trim_offsets=True, use_regex=True
)

tokenizer.train_from_iterator(iterator=data, trainer=trainer, length=train_steps)

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
