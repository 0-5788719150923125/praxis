import argparse
import os
from itertools import islice
from typing import List, Union

from datasets import load_dataset
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

os.environ["RUST_BACKTRACE"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    choices=["bpe", "unigram"],
    default=8192,
    help="The type of tokenizer to train",
)
parser.add_argument(
    "--num_examples",
    type=int,
    default=1_000_000,
    help="The number of examples to train",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
    default=8192,
    help="The absolute vocab size to use",
)

args = parser.parse_args()

tokenizer_type = args.type
num_examples = args.num_examples
vocab_size = args.vocab_size

dropout = 0.1

save_path = "data/praxis"
archive_path = f"{save_path}-{vocab_size}"

pad_token = "<|endoftext|>"
bos_token = "<|im_start|>"
eos_token = "<|im_end|>"

key = "text"
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

if tokenizer_type == "bpe":
    tokenizer = Tokenizer(
        models.BPE(dropout=dropout, cache_capacity=4096 * 8, byte_fallback=True)
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
elif tokenizer_type == "unigram":
    tokenizer = Tokenizer(models.Unigram(byte_fallback=True))
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        shrinking_factor=0.75,
        max_piece_length=16,
        n_sub_iterations=2,
        special_tokens=[
            pad_token,
            bos_token,
            eos_token,
        ],
    )

tokenizer.add_special_tokens([pad_token, bos_token, eos_token])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True, use_regex=True
)
tokenizer.normalizer = normalizers.NFKC()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

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
