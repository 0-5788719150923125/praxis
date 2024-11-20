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

num_examples = 10_000_000

vocab_size = 8192
dropout = 0.1

save_path = "data/praxis"
archive_path = save_path + f"-{vocab_size}"

pad_token = "<|pad|>"
bos_token = "<|bos|>"
eos_token = "<|eos|>"
unk_token = "<|unk|>"
start_token = "<|im_start|>"
end_token = "<|im_end|>"


dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    streaming=True,
    trust_remote_code=True,
    cache_dir="data/datasets",
).shuffle(
    seed=59,
    buffer_size=10_000,
)

column = "text"
iterator = islice((item[column] for item in dataset), num_examples)

tokenizer = Tokenizer(
    models.BPE(
        dropout=dropout,
        unk_token=unk_token,
        cache_capacity=4096,
    )
)

tokenizer.add_special_tokens(
    [
        unk_token,
        pad_token,
        bos_token,
        eos_token,
        start_token,
        end_token,
    ]
)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
    special_tokens=[
        unk_token,
        pad_token,
        bos_token,
        eos_token,
        start_token,
        end_token,
    ],
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True, use_regex=True
)

tokenizer.normalizer = normalizers.NFC()

tokenizer.decoder = decoders.ByteLevel()
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
custom_special_tokens = {"additional_special_tokens": [start_token, end_token]}
trained_tokenizer.add_special_tokens(custom_special_tokens)

os.makedirs(save_path, exist_ok=True)
os.makedirs(archive_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
trained_tokenizer.save_pretrained(archive_path)
