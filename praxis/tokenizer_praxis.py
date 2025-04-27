import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from jinja2 import Template
from transformers import PreTrainedTokenizer


class ByteLevelTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer class that wraps the BltTokenizer while conforming to
    the HuggingFace PreTrainedTokenizer interface.
    """

    SPECIAL_TOKENS = {
        "boe_token": ("[BOE]", 0),
        "pad_token": ("[PAD]", 0),  # since original PAD_ID is -1
        "bos_token": ("[BOS]", 1),
        "eos_token": ("[EOS]", 2),
        "sep_token": ("[SEP]", 3),
        "bpe_token": ("[BPE]", 4),
    }

    def __init__(self, **kwargs: Any) -> None:
        # Initialize special tokens from kwargs or use defaults
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

        # Initialize chat_template
        self.chat_template = None
        if "chat_template" in kwargs:
            self.chat_template = kwargs["chat_template"]
        else:
            # Set a default ChatML template
            self.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ bos_token }}system
{{ message['content'] }}
{{ eos_token }}
{% elif message['role'] == 'user' %}
{{ bos_token }}user
{{ message['content'] }}
{{ eos_token }}
{% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{{ message['content'] }}
{{ eos_token }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}"""

    def _special_tokens_present(self, text: str) -> bool:
        """Check if special tokens already exist in the text."""
        return any(
            token_value in text for _, (token_value, _) in self.SPECIAL_TOKENS.items()
        )

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> List[int]:
        """Override encode to properly handle add_special_tokens flag."""
        tokens = self._tokenize(text, **kwargs)
        token_ids = self.convert_tokens_to_ids(tokens)
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

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[str, List[int], Dict[str, Any]]:
        """
        Apply the chat template to a list of messages.

        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            tokenize: Whether to tokenize the result
            add_generation_prompt: Whether to add a generation prompt for the assistant
            return_tensors: If tokenize=True, what type of tensors to return (pt, tf, np)
            return_dict: If True, return a dict with additional information

        Returns:
            If tokenize=False, returns a string with the formatted chat
            If tokenize=True, returns token IDs in the requested format
            If return_dict=True, returns a dict with additional fields
        """
        if not self.chat_template:
            raise ValueError(
                "Cannot use apply_chat_template() because tokenizer.chat_template is not set! "
                "Please set tokenizer.chat_template to a valid Jinja template string."
            )

        # Create a Jinja template from our template string
        template = Template(self.chat_template)

        # Define the template variables
        template_vars = {
            "messages": messages,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }

        # Render the template
        chat_text = template.render(**template_vars)

        # If not tokenizing, just return the text
        if not tokenize:
            return chat_text

        # Tokenize the text
        encoding = self.encode(chat_text, add_special_tokens=False)

        # Convert to the requested tensor format
        if return_tensors == "pt":
            encoding = torch.tensor([encoding])
        elif return_tensors == "np":
            import numpy as np

            encoding = np.array([encoding])
        elif return_tensors == "tf":
            import tensorflow as tf

            encoding = tf.convert_to_tensor([encoding])
        elif return_tensors is not None:
            raise ValueError(f"Unsupported tensor type: {return_tensors}")

        # Return the encoding, either as a list/tensor or as a dict
        if return_dict:
            return {"input_ids": encoding}
        return encoding
