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

num_examples = 1_000_000

save_path = f"data/praxis"


vocab_size = 8192
max_token_length = 3
dropout = 0.5


pad_token = "[PAD]"
bos_token = "[BOS]"
eos_token = "[EOS]"
unk_token = "[UNK]"

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
    trust_remote_code=True,
    cache_dir="tmp",
).shuffle(
    seed=44,
    buffer_size=100_000,
)

column = "text"
iterator = islice((item[column] for item in dataset), num_examples)

tokenizer = Tokenizer(
    models.BPE(
        dropout=dropout,
        unk_token=unk_token,
        byte_fallback=True,
        fuse_unk=True,
        cache_capacity=4096,
    )
)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    max_token_length=max_token_length,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
    special_tokens=[
        unk_token,
        pad_token,
        bos_token,
        eos_token,
    ],
)

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        pre_tokenizers.Punctuation(behavior="isolated"),
        pre_tokenizers.Digits(individual_digits=True),
    ]
)

tokenizer.normalizer = normalizers.NFC()
tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, use_regex=True)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

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

archive_path = save_path + f"-{vocab_size}"
os.makedirs(archive_path, exist_ok=True)
trained_tokenizer.save_pretrained(archive_path)
