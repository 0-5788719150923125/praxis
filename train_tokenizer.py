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

num_examples = 5_000_000

vocab_size = 1024 * 8
dropout = 0.1

save_path = "data/praxis"
archive_path = f"{save_path}-{vocab_size}"

pad_token = "<|endoftext|>"
bos_token = "<|im_start|>"
eos_token = "<|im_end|>"

dataset = load_dataset(
    "tiiuae/falcon-refinedweb",
    name="default",
    split="train",
    streaming=True,
    trust_remote_code=True,
    cache_dir="data/datasets",
).shuffle(
    seed=42,
    buffer_size=10_000,
)

column = "content"
iterator = islice((item[column] for item in dataset), num_examples)

tokenizer = Tokenizer(
    models.BPE(dropout=dropout, cache_capacity=4096 * 8, byte_fallback=True)
)

tokenizer.add_special_tokens(
    [
        pad_token,
        bos_token,
        eos_token,
    ]
)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
    special_tokens=[
        pad_token,
        bos_token,
        eos_token,
    ],
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True, use_regex=True
)

tokenizer.normalizer = normalizers.NFKC()

tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
# bos = "<|begin_of_text|>"
# eos = "<|end_of_text|>"
# tokenizer._tokenizer.post_processor = processors.Sequence(
#     [
#         processors.ByteLevel(trim_offsets=False),
#         processors.TemplateProcessing(
#             single=f"{bos}:0 $A:0 {eos}:0",
#             pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
#             special_tokens=[
#                 (bos, tokenizer.bos_token_id),
#                 (eos, tokenizer.eos_token_id),
#             ],
#         ),
#     ]
# )

tokenizer.train_from_iterator(iterator=iterator, trainer=trainer, length=num_examples)

trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
trained_tokenizer.add_special_tokens(
    {
        "pad_token": pad_token,
        "bos_token": bos_token,
        "eos_token": eos_token,
    }
)

os.makedirs(save_path, exist_ok=True)
os.makedirs(archive_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
trained_tokenizer.save_pretrained(archive_path)
