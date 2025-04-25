from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
)
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    SPECIAL_TOKENS = {
        "boe_token": ("<|boe|>", BOE_ID),
        "pad_token": ("<|endoftext|>", BOE_ID),  # since PAD_ID is -1
        "bos_token": ("<|im_start|>", BOS_ID),
        "eos_token": ("<|im_end|>", EOS_ID),
        "bpe_token": ("<|bpe|>", BPE_ID),
    }

    def __init__(self, **kwargs: Any) -> None:
        for token_name, (token_value, _) in self.SPECIAL_TOKENS.items():
            if token_name not in kwargs:
                kwargs[token_name] = token_value

        self.vocab_size_unit_1 = 256
        super().__init__(**kwargs)

        self._tokenizer = BltTokenizer(
            vocab_size_unit_1=self.vocab_size_unit_1,
            bpe_delim=False,
            add_bos=False,  # We handle special tokens ourselves
            add_eos=False,
        )

        for token_name, (_, token_id) in self.SPECIAL_TOKENS.items():
            setattr(self._tokenizer, f"{token_name[:-6]}_id", token_id)

        # Create reverse mapping for special tokens
        self._id_to_special_token = {
            token_id: token_value
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }

    def _special_tokens_present(self, text: str) -> bool:
        """Check if special tokens already exist in the text."""
        return any(
            token_value in text for _, (token_value, _) in self.SPECIAL_TOKENS.items()
        )

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """Override encode to properly handle add_special_tokens flag."""

        # Check if special tokens already exist
        # has_special = self._special_tokens_present(text)

        # Basic tokenization
        tokens = self._tokenize(text, **kwargs)

        token_ids = self.convert_tokens_to_ids(tokens)

        # Return with special tokens if requested AND not already present
        # if add_special_tokens and not has_special:
        #     result = [self.bos_token_id] + token_ids + [self.eos_token_id]
        #     return result

        return token_ids

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Minimal implementation that just preserves existing special tokens."""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def _tokenize(self, text: str, **kwargs: Any) -> List[str]:
        """Converts a string into a sequence of tokens."""
        # First check if the text is a special token
        if text in [token for _, (token, _) in self.SPECIAL_TOKENS.items()]:
            return [text]

        # Otherwise, use byte tokenization
        byte_tokens = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        return [
            str(token - self._tokenizer.offsetting_special_char)
            for token in byte_tokens
        ]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens to a single string."""
        byte_tokens = []
        result = ""

        for token in tokens:
            if self._is_special_token(token):
                # If we have accumulated byte tokens, decode them first
                if byte_tokens:
                    decoded = self._tokenizer.decode(
                        [
                            int(t) + self._tokenizer.offsetting_special_char
                            for t in byte_tokens
                        ]
                    )
                    if decoded:
                        result += decoded
                    byte_tokens = []
                # Add the special token directly - no spaces
                result += token
            else:
                byte_tokens.append(token)

        # Handle any remaining byte tokens
        if byte_tokens:
            decoded = self._tokenizer.decode(
                [int(t) + self._tokenizer.offsetting_special_char for t in byte_tokens]
            )
            if decoded:
                result += decoded

        return result if result else " "

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        vocab = {
            token_value: token_id
            for _, (token_value, token_id) in self.SPECIAL_TOKENS.items()
        }

        for i in range(self.vocab_size_unit_1):
            vocab[str(i)] = i + self._token_id_offset

        return vocab

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_unit_1 + len(self.SPECIAL_TOKENS)

    @property
    def _token_id_offset(self) -> int:
        return max(id for _, id in self.SPECIAL_TOKENS.values()) + 1

    def _convert_token_to_id(self, token: str) -> int:
        # Check special tokens first
        for _, (token_value, token_id) in self.SPECIAL_TOKENS.items():
            if token == token_value:
                return token_id
        # Regular tokens are offset by special token count
        return int(token) + self._token_id_offset

    def _convert_id_to_token(self, index: int) -> str:
        # Check if it's a special token ID
        if index in self._id_to_special_token:
            return self._id_to_special_token[index]
        # Regular tokens
        return str(index - self._token_id_offset)

    def _is_special_token(self, token: str) -> bool:
        return any(
            token == special_token for special_token, _ in self.SPECIAL_TOKENS.values()
        )
