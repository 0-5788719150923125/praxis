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


class TokenMonsterTokenizer(PreTrainedTokenizer):
    config_class = TokenMonsterConfig

    def __init__(
        self,
        config,
        **kwargs,
    ):

        self.vocab_file = config.vocab_file
        self.add_bos_token = config.add_bos_token
        self.add_eos_token = config.add_eos_token

        tokenmonster.set_local_directory("./build/tokenmonster")
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

        self.split_special_tokens = False
        self._in_target_context_manager = False

        # super().__init__(
        #     pad_token=self.pad_token,
        #     bos_token=self.bos_token,
        #     eos_token=self.eos_token,
        #     unk_token=self.unk_token,
        #     **kwargs,
        # )

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

    def get_vocab(self):
        # return self.tokenizer.get_dictionary()
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def tokenize(self, text: str) -> List[str]:
        # return self.tokenizer.tokenize(text)
        pass

    @property
    def added_tokens_encoder(self):
        # TokenMonster doesn't have a simple mapping of added tokens to IDs
        # We might need to return an empty dict or implement a custom logic
        return {}

    @property
    def added_tokens_decoder(self):
        # Similar to added_tokens_encoder, we might need custom logic here
        return {}

    def _add_tokens(self, new_tokens, special_tokens=False):
        # TokenMonster might handle token addition differently
        # We need to implement this in a way that aligns with TokenMonster's approach
        added_tokens = 0
        for token in new_tokens:
            # Add logic to add token to TokenMonster vocabulary
            # This might involve modifying the underlying TokenMonster object
            added_tokens += 1
        return added_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.tokenizer.token_to_id(tokens)
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        return [self.tokenizer.id_to_token(id) for id in ids]

    def num_special_tokens_to_add(self, pair=False):
        return 4

    def load_vocab(self, vocab_file):
        return tokenmonster.load(vocab_file)

    def decode(
        self,
        token_ids: Union[int, List[int], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = (
                token_ids.astype(np.int64)
                if isinstance(token_ids, np.ndarray)
                else token_ids.to(torch.int64)
            )
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, list):
            token_ids = [np.int64(id) for id in token_ids]
        else:
            token_ids = [np.int64(token_ids)]

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
                ids = np.array(self.tokenizer.tokenize(t), dtype=np.int64)
            else:
                ids = np.array([], dtype=np.int64)
            encoded.append(ids)

        if add_special_tokens:
            if self.add_bos_token:
                bos_token_id = np.array(
                    [self.tokenizer.token_to_id(self.bos_token)], dtype=np.int64
                )
                encoded = [np.concatenate((bos_token_id, ids)) for ids in encoded]

            if self.add_eos_token:
                eos_token_id = np.array(
                    [self.tokenizer.token_to_id(self.eos_token)], dtype=np.int64
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

        outputs = {"input_ids": encoded}
        if return_tensors:
            outputs = self._convert_to_tensors(outputs, return_tensors)

        return outputs["input_ids"]

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

        # Tokenize all texts
        encoded = [np.array(self.tokenizer.tokenize(t), dtype=np.int64) for t in text]

        # Handle text pairs if provided
        if text_pair:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            encoded_pairs = [
                np.array(self.tokenizer.tokenize(t), dtype=np.int64) for t in text_pair
            ]
            encoded = [np.concatenate((e, p)) for e, p in zip(encoded, encoded_pairs)]

        # Handle striding and overflowing tokens
        all_input_ids = []
        for ids in encoded:
            if return_overflowing_tokens and max_length and stride > 0:
                stride_encoded = []
                for i in range(0, len(ids), stride):
                    chunk = ids[i : i + max_length]
                    stride_encoded.append(chunk)
                all_input_ids.extend(stride_encoded)
            else:
                all_input_ids.append(ids)

        # Add special tokens
        if add_special_tokens:
            all_input_ids = [self._add_special_tokens(ids) for ids in all_input_ids]

        # Pad or truncate
        if padding or truncation:
            all_input_ids = [
                self._pad_and_truncate(
                    ids, max_length, padding, truncation, pad_to_multiple_of
                )
                for ids in all_input_ids
            ]

        # Prepare outputs
        outputs = {"input_ids": all_input_ids}

        if return_attention_mask:
            outputs["attention_mask"] = [
                np.where(ids != self.pad_token_id, 1, 0) for ids in all_input_ids
            ]

        if return_token_type_ids:
            if text_pair:
                outputs["token_type_ids"] = [
                    np.concatenate(
                        [
                            np.zeros(len(self._add_special_tokens(ids1))),
                            np.ones(len(self._add_special_tokens(ids2))),
                        ]
                    )
                    for ids1, ids2 in zip(encoded, encoded_pairs)
                ]
            else:
                outputs["token_type_ids"] = [
                    np.zeros_like(ids) for ids in all_input_ids
                ]

        if return_special_tokens_mask:
            special_tokens = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            }
            outputs["special_tokens_mask"] = [
                np.isin(ids, list(special_tokens)).astype(np.int64)
                for ids in all_input_ids
            ]

        if return_length:
            outputs["length"] = [len(ids) for ids in all_input_ids]

        # Convert to tensors if needed
        if return_tensors:
            outputs = self._convert_to_tensors(outputs, return_tensors)

        return BatchEncoding(outputs)

    def _pad_and_truncate(
        self,
        ids: np.ndarray,
        max_length: Optional[int],
        padding: Union[bool, str, PaddingStrategy],
        truncation: Union[bool, str, TruncationStrategy],
        pad_to_multiple_of: Optional[int],
    ) -> np.ndarray:
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        if padding == PaddingStrategy.MAX_LENGTH and max_length:
            pad_length = max_length - len(ids)
            if pad_length > 0:
                ids = np.pad(ids, (0, pad_length), constant_values=self.pad_token_id)
        elif padding == PaddingStrategy.LONGEST:
            # This should be handled in batch processing
            pass

        if pad_to_multiple_of:
            pad_len = (
                pad_to_multiple_of - len(ids) % pad_to_multiple_of
            ) % pad_to_multiple_of
            if pad_len > 0:
                ids = np.pad(ids, (0, pad_len), constant_values=self.pad_token_id)

        return ids

    def _add_special_tokens(self, ids: np.ndarray) -> np.ndarray:
        if self.add_bos_token:
            bos_token_id = self.tokenizer.token_to_id(self.bos_token)
            if ids[0] != bos_token_id:
                ids = np.concatenate(([bos_token_id], ids))
        if self.add_eos_token:
            eos_token_id = self.tokenizer.token_to_id(self.eos_token)
            if ids[-1] != eos_token_id:
                ids = np.concatenate((ids, [eos_token_id]))
        return ids

    def _convert_to_tensors(
        self, outputs: Dict[str, List], return_tensors: str
    ) -> Dict[str, Any]:
        if return_tensors == "pt":
            return self._pad_and_convert_to_pytorch(outputs)
        elif return_tensors == "tf":
            import tensorflow as tf

            return {k: tf.constant(self._pad_sequences(v)) for k, v in outputs.items()}
        elif return_tensors == "np":
            return {k: np.array(self._pad_sequences(v)) for k, v in outputs.items()}
        else:
            return outputs

    def _pad_sequences(self, sequences):
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            padded_seq = np.pad(
                seq, (0, max_length - len(seq)), constant_values=self.pad_token_id
            )
            padded_sequences.append(padded_seq)
        return padded_sequences

    def _pad_and_convert_to_pytorch(self, outputs):
        import torch

        padded_outputs = {}
        for key, value in outputs.items():
            if isinstance(value[0], (list, np.ndarray)):
                padded_value = self._pad_sequences(value)
                padded_outputs[key] = torch.tensor(
                    np.array(padded_value), dtype=torch.int64
                )
            else:
                padded_outputs[key] = torch.tensor(value, dtype=torch.int64)
        return padded_outputs

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # Identify special token IDs
        special_tokens = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        }

        if token_ids_1 is None:
            return [1 if token in special_tokens else 0 for token in token_ids_0]
        return [
            1 if token in special_tokens else 0 for token in token_ids_0 + token_ids_1
        ]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return (
            [self.bos_token_id]
            + token_ids_0
            + [self.eos_token_id]
            + token_ids_1
            + [self.eos_token_id]
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)


if __name__ == "__main__":
    # Register the tokenizer
    AutoTokenizer.register(TokenMonsterConfig, TokenMonsterTokenizer)

    # Usage example
    config = TokenMonsterConfig(vocab_file="englishcode-32000-consistent-nocapcode-v1")
    tokenizer = TokenMonsterTokenizer(config)

    # Test the tokenizer
    text = "Hello, how are you?"
    encoded = tokenizer.encode(text, return_tensors="pt")
    print("Encoded:", encoded)
    print("Input IDs shape:", encoded.shape)

    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print("Decoded:", decoded)
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=False)
    print("Decoded (w/ special tokens):", decoded)

    # Test get_vocab
    print("Vocabulary size:", tokenizer.vocab_size)

    vocab = tokenizer.get_vocab()
    print("First 10 items:", dict(list(vocab.items())[:10]))
    print("get_vocab() length:", len(vocab))

    # Test tokenizer with parameters similar to production use case
    long_text = (
        "This is a longer piece of text. We will use this to test the tokenizer.\n\nBut should we we, or shouldn't we?\nDoes it even matter? "
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
    print("Input IDs dtype:", tokens["input_ids"].dtype)
    print("First sequence of tokens:", tokens["input_ids"][0])
    print("Number of sequences:", len(tokens["input_ids"]))
    print("Decoded first sequence:", tokenizer.decode(tokens["input_ids"][0]))
