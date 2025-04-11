import argparse
import os
from itertools import islice
from typing import List, Union

from datasets import load_dataset
from more_itertools import chunked
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

parser = argparse.ArgumentParser(
    description="User-supplied arguments to this script.",
)
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
chunk_size = 256

dropout = 0.1

save_path = "data/praxis"
archive_path = f"{save_path}-{vocab_size}"

pad_token = "<|endoftext|>"
bos_token = "<|im_start|>"
eos_token = "<|im_end|>"
unk_token = "<|unk|>"

key = "text"
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-350BT",
    split="train",
    streaming=True,
    trust_remote_code=True,
    cache_dir="data/datasets",
).shuffle(
    seed=42,
    buffer_size=10_000,
)


def chunks_from_iterator(
    dataset, key="text", chunk_size=1024, num_chunks=1_000_000, min_length=32
):
    chunks_yielded = 0
    for item in dataset:
        text = item[key]

        # Skip if text is None or empty
        if not text:
            continue

        # Process this example into chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]

            # Skip very short chunks
            if len(chunk) < min_length:
                continue

            # # Skip chunks with unusual character patterns that might cause problems
            # if len(chunk) > 0 and (
            #     chunk.count("\n") > len(chunk) * 0.5  # Too many newlines
            #     or len(set(chunk)) < 3  # Too few unique characters
            # ):
            #     continue

            yield chunk
            chunks_yielded += 1
            if chunks_yielded >= num_chunks:
                return


iterator = chunks_from_iterator(
    dataset, key=key, chunk_size=chunk_size, num_chunks=num_examples
)

tokenizer = Tokenizer(models.Unigram())

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

if tokenizer_type == "bpe":
    tokenizer.add_special_tokens([pad_token, bos_token, eos_token])
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
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
elif tokenizer_type == "unigram":
    tokenizer.add_special_tokens(
        [
            pad_token,
            bos_token,
            eos_token,
            # unk_token
        ]
    )
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        shrinking_factor=0.75,
        unk_token=unk_token,
        max_piece_length=16,
        n_sub_iterations=2,
        special_tokens=[
            pad_token,
            bos_token,
            eos_token,
            # unk_token,
        ],
    )
    replacement = "▁"
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=replacement, prepend_scheme="always"
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement=replacement, prepend_scheme="always"
    )

tokenizer.train_from_iterator(iterator=iterator, trainer=trainer, length=num_examples)

trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

if tokenizer_type == "bpe":
    trained_tokenizer.add_special_tokens(
        {
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }
    )
elif tokenizer_type == "unigram":
    trained_tokenizer.add_special_tokens(
        {
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "unk_token": unk_token,
        }
    )

os.makedirs(save_path, exist_ok=True)
os.makedirs(archive_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
trained_tokenizer.save_pretrained(archive_path)
