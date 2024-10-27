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

save_path = "data/praxis"


vocab_size = 8192
dropout = 0.1


pad_token = "[PAD]"
bos_token = "[BOS]"
eos_token = "[EOS]"
unk_token = "[UNK]"
start_ctx_token = "[CTX]"
end_ctx_token = "[XTC]"
start_cat_token = "[CAT]"
end_cat_token = "[TAC]"


dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-100BT",
    split="train",
    streaming=True,
    trust_remote_code=True,
    cache_dir="tmp",
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
        start_ctx_token,
        end_ctx_token,
        start_cat_token,
        end_cat_token,
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
        start_ctx_token,
        end_ctx_token,
        start_cat_token,
        end_cat_token,
    ],
)

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Punctuation(behavior="isolated"),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ]
)

tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])

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
custom_special_tokens = {
    "additional_special_tokens": ["[CTX]", "[XTC]", "[CAT]", "[TAC]"]
}
trained_tokenizer.add_special_tokens(custom_special_tokens)
archive_path = save_path + f"-{vocab_size}"

os.makedirs(save_path, exist_ok=True)
os.makedirs(archive_path, exist_ok=True)

trained_tokenizer.save_pretrained(save_path)
trained_tokenizer.save_pretrained(archive_path)
