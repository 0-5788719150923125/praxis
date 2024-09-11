import os
from typing import List, Optional, Union

import numpy as np
import tokenmonster
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer


class TokenMonsterConfig(PretrainedConfig):
    model_type = "tokenmonster"

    def __init__(
        self,
        vocab_file="englishcode-32000-consistent-v1",
        add_bos_token=True,
        add_eos_token=False,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        **kwargs
    ):
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        super().__init__(**kwargs)


class TokenMonsterTokenizer(PreTrainedTokenizer):
    config_class = TokenMonsterConfig

    def __init__(self, config, **kwargs):
        self.vocab_file = config.vocab_file
        self.add_bos_token = config.add_bos_token
        self.add_eos_token = config.add_eos_token

        tokenmonster.set_local_directory("./data/tokenmonster")
        self.tokenizer = self.load_vocab(self.vocab_file)

        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.tokenizer.enable_unk_token()

        original_size = self.tokenizer.vocab_size

        self.tokenizer.modify(self.pad_token, resize=original_size + 1)
        self.tokenizer.modify(self.bos_token, resize=original_size + 2)
        self.tokenizer.modify(self.eos_token, resize=original_size + 3)
        self.tokenizer.modify(self.unk_token, resize=original_size + 4, change_unk=True)

        # Verify the new vocabulary size
        assert (
            self.tokenizer.vocab_size == original_size + 4
        ), "Vocabulary size mismatch"

        super().__init__(
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    # def get_vocab(self):
    #     vocab = {}
    #     for i in range(self.vocab_size):
    #         token = self.convert_ids_to_tokens(i)
    #         if token not in vocab:
    #             vocab[token] = set()
    #         vocab[token].add(i)

    #     # Convert sets to sorted lists for consistency
    #     return {token: sorted(list(ids)) for token, ids in vocab.items()}

    def get_vocab(self):
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: Union[str, int, np.integer]) -> int:
        if isinstance(token, (int, np.integer)):
            return int(token)
        id = self.tokenizer.token_to_id(token)
        return id if id is not None else self.tokenizer.token_to_id(self.unk_token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(self.convert_tokens_to_ids(tokens))

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> List[str]:
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.vocab",
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            self.tokenizer.save(out_vocab_file)
        return [out_vocab_file]

    # def build_inputs_with_special_tokens(
    #     self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # ) -> List[int]:
    #     if token_ids_1 is None:
    #         return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    #     return (
    #         [self.bos_token_id]
    #         + token_ids_0
    #         + [self.eos_token_id]
    #         + token_ids_1
    #         + [self.eos_token_id]
    #     )

    def load_vocab(self, vocab_file):
        return tokenmonster.load(vocab_file)


if __name__ == "__main__":
    # Register the tokenizer
    AutoTokenizer.register(TokenMonsterConfig, TokenMonsterTokenizer)

    # Usage example
    config = TokenMonsterConfig(vocab_file="englishcode-32000-consistent-nocapcode-v1")
    tokenizer = TokenMonsterTokenizer(config)

    # Test the tokenizer
    text = "Hello, how are you?"
    print("\nEncoding text:", text)
    encoded = tokenizer.encode(text, return_tensors="pt")
    print("Encoded:", encoded)
    print("Input IDs shape:", encoded.shape)

    print("\nDecoding (skip_special_tokens=True):")
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print("Decoded:", decoded)

    print("\nDecoding (skip_special_tokens=False):")
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=False)
    print("Decoded (w/ special tokens):", decoded)

    # Test get_vocab
    print("\nVocabulary size:", tokenizer.vocab_size)

    vocab = tokenizer.get_vocab()
    print("First 10 items:", dict(list(vocab.items())[:10]))
    print("get_vocab() length:", len(vocab))

    # Test tokenizer with parameters similar to production use case
    long_text = (
        "This is a longer piece of text. We will use this to test the tokenizer.\n\nBut should we we, or shouldn't we?\nDoes it even matter? "
        * 10
    )
    print("\nTokenizing long text:")
    tokens = tokenizer(
        text=long_text,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    print("\nTokens:", tokens)
    print("Input IDs shape:", tokens["input_ids"].shape)
    print("First sequence of tokens:", tokens["input_ids"][0])
    print("Decoded first sequence:", tokenizer.decode(tokens["input_ids"][0]))
