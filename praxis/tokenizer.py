import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tokenmonster
import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TensorType,
    TruncationStrategy,
)


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
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        super().__init__(**kwargs)


class TokenMonster(PreTrainedTokenizer):
    config_class = TokenMonsterConfig

    def __init__(
        self,
        config,
        **kwargs,
    ):

        # print(config)
        # config = TokenMonsterConfig(
        #     vocab_file=vocab_file,
        #     add_bos_token=add_bos_token,
        #     add_eos_token=add_eos_token,
        #     **kwargs,
        # )
        # print(kwargs)
        self.vocab_file = config.vocab_file
        self.add_bos_token = config.add_bos_token
        self.add_eos_token = config.add_eos_token

        tokenmonster.set_local_directory("./data/tokenmonster")
        self.tokenizer = self.load_vocab(self.vocab_file)

        # Add special tokens directly to the TokenMonster vocabulary
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        return cls(
            vocab_file=config.vocab_file,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab_dict = {
            item["token_decoded"]: item["id"]
            for item in self.tokenizer.get_dictionary().values()
        }
        return vocab_dict

    def load_vocab(self, vocab_file):
        return tokenmonster.load(vocab_file)

    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = (
                token_ids.astype(np.uint16)
                if isinstance(token_ids, np.ndarray)
                else token_ids.to(torch.uint16)
            )
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list):
            token_ids = [np.uint16(id) for id in token_ids]
        else:
            token_ids = [np.uint16(token_ids)]

        # Decode using TokenMonster's built-in method
        decoded = self.tokenizer.decode(token_ids)

        if skip_special_tokens:
            # Remove special tokens after decoding
            special_tokens = [
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ]
            for token in special_tokens:
                decoded = decoded.replace(token, "")
            # Remove any extra spaces that might have been left
            decoded = " ".join(decoded.split())

        return decoded

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[Dict[str, Union[List[int], np.ndarray, torch.Tensor]], BatchEncoding]:
        if isinstance(text, str):
            text = [text]

        encoded = []
        for t in text:
            if t:
                ids = np.array(self.tokenizer.tokenize(t), dtype=np.uint16)
            else:
                ids = np.array([], dtype=np.uint16)
            encoded.append(ids)

        if add_special_tokens:
            if self.add_bos_token:
                bos_token_id = np.array(
                    [self.tokenizer.token_to_id(self.bos_token)], dtype=np.uint16
                )
                encoded = [np.concatenate((bos_token_id, ids)) for ids in encoded]

            if self.add_eos_token:
                eos_token_id = np.array(
                    [self.tokenizer.token_to_id(self.eos_token)], dtype=np.uint16
                )
                encoded = [np.concatenate((ids, eos_token_id)) for ids in encoded]

        # Handle truncation and padding here if needed
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if padding:
            pad_token_id = self.tokenizer.token_to_id(self.pad_token)
            if max_length is not None:
                encoded = [
                    np.pad(
                        ids,
                        (0, max_length - len(ids)),
                        "constant",
                        constant_values=pad_token_id,
                    )
                    for ids in encoded
                ]
            else:
                max_len = max(len(ids) for ids in encoded)
                encoded = [
                    np.pad(
                        ids,
                        (0, max_len - len(ids)),
                        "constant",
                        constant_values=pad_token_id,
                    )
                    for ids in encoded
                ]

        # Prepare the output dictionary
        output = {"input_ids": encoded}

        if return_tensors == "np":
            output = {k: np.array(v) for k, v in output.items()}
        elif return_tensors == "pt":
            output = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        elif return_tensors == "tf":
            import tensorflow as tf

            output = {k: tf.constant(v, dtype=tf.int32) for k, v in output.items()}

        return BatchEncoding(output)

    # def _tokenize(self, text: str, **kwargs) -> List[str]:
    #     return self.encode(text, kwargs)

    # def _convert_token_to_id(self, token: str) -> int:
    #     return self.tokenizer.token_to_id(token)

    # def _convert_id_to_token(self, index: int) -> str:
    #     return self.tokenizer.id_to_token(index)

    def encode_plus(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]

        encoded = [list(map(int, self.tokenizer.tokenize(t))) for t in text]

        # Handle striding and overflowing tokens
        all_input_ids = []
        for ids in encoded:
            if return_overflowing_tokens and max_length and stride > 0:
                stride_encoded = []
                for i in range(0, len(ids), stride):
                    chunk = ids[i : i + max_length]
                    if len(chunk) < max_length:
                        if padding:
                            chunk = chunk + [self.pad_token_id] * (
                                max_length - len(chunk)
                            )
                        elif truncation:
                            continue
                    stride_encoded.append(chunk)
                all_input_ids.extend(stride_encoded)
            else:
                all_input_ids.append(ids)

        # Add special tokens only once, after handling striding and overflowing
        if add_special_tokens:
            all_input_ids = [self._add_special_tokens(ids) for ids in all_input_ids]

        # Pad or truncate if needed
        if padding or truncation:
            all_input_ids = [
                self._pad_and_truncate(
                    ids, max_length, padding, truncation, pad_to_multiple_of
                )
                for ids in all_input_ids
            ]

        # Prepare the outputs
        outputs = {"input_ids": all_input_ids}

        if return_attention_mask:
            outputs["attention_mask"] = [[1] * len(ids) for ids in all_input_ids]

        if return_token_type_ids:
            outputs["token_type_ids"] = [[0] * len(ids) for ids in all_input_ids]

        if return_special_tokens_mask:
            outputs["special_tokens_mask"] = [
                [1 if token in self._special_token_ids.values() else 0 for token in ids]
                for ids in all_input_ids
            ]

        if return_length:
            outputs["length"] = [len(ids) for ids in all_input_ids]

        # Convert to tensors if needed
        if return_tensors == "pt":
            outputs = {k: torch.tensor(v) for k, v in outputs.items()}
        elif return_tensors == "tf":
            import tensorflow as tf

            outputs = {k: tf.constant(v) for k, v in outputs.items()}
        elif return_tensors == "np":
            outputs = {k: np.array(v) for k, v in outputs.items()}

        return BatchEncoding(outputs)

    def _add_special_tokens(self, ids: List[int]) -> List[int]:
        if self.add_bos_token:
            bos_token_id = self.tokenizer.token_to_id(self.bos_token)
            if ids[0] != bos_token_id:
                ids = [bos_token_id] + ids
        if self.add_eos_token:
            eos_token_id = self.tokenizer.token_to_id(self.eos_token)
            if ids[-1] != eos_token_id:
                ids = ids + [eos_token_id]
        return ids

    def _pad_and_truncate(
        self,
        ids: List[int],
        max_length: Optional[int],
        padding: Union[bool, str, PaddingStrategy],
        truncation: Union[bool, str, TruncationStrategy],
        pad_to_multiple_of: Optional[int],
    ) -> List[int]:
        if max_length is not None:
            if truncation:
                ids = ids[:max_length]
            if padding:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if pad_to_multiple_of:
            pad_len = (
                pad_to_multiple_of - len(ids) % pad_to_multiple_of
            ) % pad_to_multiple_of
            ids = ids + [self.pad_token_id] * pad_len
        return ids


if __name__ == "__main__":
    # Register the tokenizer
    AutoTokenizer.register(TokenMonsterConfig, TokenMonster)

    # Usage example
    config = TokenMonsterConfig(vocab_file="englishcode-32000-consistent-v1")
    tokenizer = TokenMonster(vocab_file=config.vocab_file)

    # Test the tokenizer
    text = "Hello, how are you?"
    encoded = tokenizer.encode(text, return_tensors="pt")
    print("Encoded:", encoded)
    print("Input IDs shape:", encoded["input_ids"].shape)

    decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    print("Decoded:", decoded)
    decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    print("Decoded (w/ special tokens):", decoded)

    # Test get_vocab
    print("Vocabulary size:", tokenizer.vocab_size)

    vocab = tokenizer.get_vocab()
    print("First 10 items:", dict(list(vocab.items())[:10]))

    # Test tokenizer with parameters similar to production use case
    long_text = (
        "This is a longer piece of text that we will use to test the tokenizer with striding and overflow tokens. "
        * 10
    )
    tokens = tokenizer(
        text=long_text,
        max_length=128,
        stride=16,
        padding=True,
        truncation=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    print("\nTokens:", tokens)
    print("Input IDs shape:", tokens["input_ids"].shape)
    print("First sequence of tokens:", tokens["input_ids"][0])
    print("Number of sequences:", len(tokens["input_ids"]))
