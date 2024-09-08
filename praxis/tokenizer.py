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
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        super().__init__(**kwargs)


class TokenMonster(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.tm"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file="englishcode-32000-consistent-v1",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.tokenizer = self.load_vocab(vocab_file)

        # Set up special tokens
        self.pad_token = kwargs.pop("pad_token", "<pad>")
        self.bos_token = kwargs.pop("bos_token", "<s>")
        self.eos_token = kwargs.pop("eos_token", "</s>")
        self.unk_token = kwargs.pop("unk_token", "<unk>")

        # Initialize special tokens list
        self._all_special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        # Extend vocabulary with special tokens
        self._base_vocab_size = self.tokenizer.vocab_size
        self._special_token_ids = {}
        for i, token in enumerate(self._all_special_tokens):
            self._special_token_ids[token] = self._base_vocab_size + i

        # Now call the parent constructor
        super().__init__(
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            **kwargs,
        )

        # Initialize other TokenMonster-specific attributes
        self._vocab_size = self._base_vocab_size + len(self._all_special_tokens)
        self._all_special_ids = list(self._special_token_ids.values())

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab_dict = {
            item["token_decoded"]: item["id"]
            for item in self.tokenizer.get_dictionary().values()
        }
        vocab_dict.update(self._special_token_ids)
        return vocab_dict

    def get_decoder(self):
        return self.tokenizer.decoder()

    def _tokenize(self, text: str) -> List[str]:
        return [self.tokenizer.id_to_token(id) for id in self.tokenizer.tokenize(text)]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_token_ids:
            return self._special_token_ids[token]
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        if index >= self._base_vocab_size:
            return list(self._special_token_ids.keys())[
                list(self._special_token_ids.values()).index(index)
            ]
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        non_special_tokens = [
            token for token in tokens if token not in self._special_token_ids
        ]
        return self.tokenizer.decode(
            [self.tokenizer.token_to_id(token) for token in non_special_tokens]
        )

    def load_vocab(self, vocab_file):
        return tokenmonster.load(vocab_file)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        eos = [self.eos_token_id]
        bos = [self.bos_token_id]
        if token_ids_1 is None:
            return [0] * len(bos + token_ids_0 + eos)
        return [0] * len(bos + token_ids_0 + eos) + [1] * len(token_ids_1 + eos)

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
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
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        # Handle special tokens
        if add_special_tokens:
            if pair_ids is not None:
                ids, pair_ids = self.build_inputs_with_special_tokens(ids, pair_ids)
            else:
                ids = self.build_inputs_with_special_tokens(ids)

        # Truncation and padding
        padding_strategy = PaddingStrategy(padding)
        truncation_strategy = (
            TruncationStrategy(truncation)
            if truncation is not None
            else TruncationStrategy.DO_NOT_TRUNCATE
        )

        # Truncate and/or pad
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            or padding_strategy != PaddingStrategy.DO_NOT_PAD
        ):
            ids = self._pad_and_truncate(
                ids,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
            )
            if pair_ids is not None:
                pair_ids = self._pad_and_truncate(
                    pair_ids,
                    max_length=max_length,
                    stride=stride,
                    pad_to_multiple_of=pad_to_multiple_of,
                    padding_strategy=padding_strategy,
                    truncation_strategy=truncation_strategy,
                )

        # Prepare outputs
        encoded_inputs = {}
        encoded_inputs["input_ids"] = ids
        if pair_ids is not None:
            encoded_inputs["input_ids"] = [ids, pair_ids]

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = (
                self.create_token_type_ids_from_sequences(ids, pair_ids)
            )

        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(ids)
            if pair_ids is not None:
                encoded_inputs["attention_mask"] = [
                    encoded_inputs["attention_mask"],
                    [1] * len(pair_ids),
                ]

        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(
                ids, pair_ids
            )

        # Convert to tensors if needed
        if return_tensors is not None:
            encoded_inputs = self.convert_to_tensors(encoded_inputs, return_tensors)

        if return_length:
            encoded_inputs["length"] = len(ids)

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    def convert_to_tensors(
        self, encoded_inputs: Dict[str, Any], return_tensors: str
    ) -> Dict[str, Any]:
        if return_tensors == "np":
            return {k: np.array(v) for k, v in encoded_inputs.items()}
        elif return_tensors == "pt":
            import torch

            return {k: torch.tensor(v) for k, v in encoded_inputs.items()}
        elif return_tensors == "tf":
            import tensorflow as tf

            return {k: tf.constant(v) for k, v in encoded_inputs.items()}
        else:
            return encoded_inputs

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        return self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

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
            if ids[0] != self._special_token_ids[self.bos_token]:
                ids = [self._special_token_ids[self.bos_token]] + ids
        if self.add_eos_token:
            if ids[-1] != self._special_token_ids[self.eos_token]:
                ids = ids + [self._special_token_ids[self.eos_token]]
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

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], List[List[int]], np.ndarray, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        encoded = [self.tokenizer.tokenize(t) for t in text]

        if add_special_tokens:
            encoded = [self._add_special_tokens(ids) for ids in encoded]

        if padding or truncation:
            encoded = self.pad_and_truncate(encoded, max_length, padding, truncation)

        if return_tensors == "np":
            return np.array(encoded, dtype=np.int32)
        elif return_tensors == "pt":
            import torch

            return torch.tensor(encoded, dtype=torch.long)
        elif return_tensors == "tf":
            import tensorflow as tf

            return tf.constant(encoded, dtype=tf.int32)
        else:
            return encoded if len(encoded) > 1 else encoded[0]

    def decode(
        self,
        token_ids: Union[
            int,
            List[int],
            List[List[int]],
            np.ndarray,
            torch.Tensor,
            Dict[str, Union[List[int], np.ndarray, torch.Tensor]],
        ],
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]

        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            if token_ids.ndim == 2:
                return [
                    self.decode(ids, skip_special_tokens=skip_special_tokens)
                    for ids in token_ids
                ]
            token_ids = token_ids.tolist()

        if (
            isinstance(token_ids, list)
            and len(token_ids) > 0
            and isinstance(token_ids[0], list)
        ):
            return [
                self.decode(ids, skip_special_tokens=skip_special_tokens)
                for ids in token_ids
            ]

        if not isinstance(token_ids, list):
            token_ids = [token_ids]

        # Separate special tokens and regular tokens
        special_tokens = []
        regular_tokens = []
        for id in token_ids:
            if isinstance(id, (int, np.integer)):
                if id in self._all_special_ids:
                    if not skip_special_tokens:
                        special_tokens.append(self._convert_id_to_token(id))
                else:
                    regular_tokens.append(id)
            else:
                raise ValueError(f"Unexpected token id type: {type(id)}")

        # Decode regular tokens all at once
        decoded_text = self.tokenizer.decode(regular_tokens)

        # Add special tokens back in their original positions without extra spaces
        if special_tokens:
            bos_token = self._convert_id_to_token(
                self._special_token_ids[self.bos_token]
            )
            eos_token = self._convert_id_to_token(
                self._special_token_ids[self.eos_token]
            )
            if bos_token in special_tokens:
                decoded_text = f"{bos_token}{decoded_text}"
            if eos_token in special_tokens:
                decoded_text = f"{decoded_text}{eos_token}"

        return decoded_text


if __name__ == "__main__":
    # Register the tokenizer
    AutoTokenizer.register(TokenMonsterConfig, TokenMonster)

    # Usage example
    config = TokenMonsterConfig(vocab_file="englishcode-32000-consistent-v1")
    tokenizer = TokenMonster(vocab_file=config.vocab_file)

    # Test the tokenizer
    text = "Hello, how are you?"
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print("Decoded:", decoded)

    # Test get_vocab
    vocab = tokenizer.get_vocab()
    print("Vocabulary size:", len(vocab))
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
    )["input_ids"]
    print(tokens)
    print("\nTokens shape:", tokens.shape)
    print("First sequence of tokens:", tokens[0])
    print("Number of sequences:", len(tokens))
    for batch in tokens:
        print(batch)
