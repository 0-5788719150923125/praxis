"""Character-level tokenizer with lazy Unicode vocab.

- Vocab is pre-allocated to the full Unicode BMP (65,536) so embedding /
  output-head row counts are fixed at init. This keeps shapes stable
  under torch.compile and avoids resize-embedding logic.
- Special token IDs come from ``PraxisConfig`` defaults (or explicit
  constructor kwargs); the character offset is ``max(special_ids) + 1``
  so there is no overlap regardless of how the config is configured.
- Observed codepoints are tracked and persisted so the fraction of
  pre-allocated rows that have received any gradient can be audited.
- Codepoints outside the BMP (emojis, rare scripts) are encoded as a
  UTF-16 surrogate *pair* - two tokens whose IDs live in the reserved
  ``U+D800..U+DFFF`` range. ``convert_tokens_to_string`` pairs them
  back. This gives full Unicode coverage and lets the otherwise-dead
  surrogate vocab rows accumulate gradient. Unpaired surrogates at
  decode time fall back to UNK so generated text is always valid
  UTF-8.
"""

import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from transformers import PreTrainedTokenizer

from praxis.configuration import PraxisConfig

from .base import PraxisTokenizerBase
from .chat_templates import get_chat_template

BMP_SIZE = 0x10000  # 65,536
SURROGATE_LOW = 0xD800
SURROGATE_HIGH_END = 0xDC00  # first low surrogate
SURROGATE_END = 0xE000  # one past last low surrogate


def _is_high_surrogate(cp: int) -> bool:
    return SURROGATE_LOW <= cp < SURROGATE_HIGH_END


def _is_low_surrogate(cp: int) -> bool:
    return SURROGATE_HIGH_END <= cp < SURROGATE_END


def _codepoint_to_surrogates(cp: int) -> tuple:
    """Encode a supplementary codepoint as a UTF-16 surrogate pair."""
    cp -= 0x10000
    hi = SURROGATE_LOW + (cp >> 10)
    lo = SURROGATE_HIGH_END + (cp & 0x3FF)
    return hi, lo


def _surrogates_to_codepoint(hi: int, lo: int) -> int:
    return 0x10000 + ((hi - SURROGATE_LOW) << 10) + (lo - SURROGATE_HIGH_END)


def _praxis_default(name: str) -> int:
    """Pull the default value for a PraxisConfig parameter."""
    return inspect.signature(PraxisConfig.__init__).parameters[name].default


class CharLevelTokenizer(PreTrainedTokenizer, PraxisTokenizerBase):
    """Character-level tokenizer with fixed-shape BMP vocab.

    Token id layout:
        special_ids               -> the 4 special tokens (from config)
        offset..offset+BMP_SIZE-1 -> BMP codepoints (id - offset)
    """

    SPECIAL_TOKEN_STRINGS = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "sep_token": "[SEP]",
    }

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "char_vocab.json"}

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        chat_template: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Token IDs: explicit kwargs > persisted vocab > PraxisConfig defaults.
        persisted_ids: Dict[str, int] = {}
        persisted_observed: Set[int] = set()
        if vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            persisted_ids = data.get("special_token_ids", {})
            persisted_observed = {int(cp) for cp in data.get("observed_codepoints", [])}

        def _resolve(name: str, explicit: Optional[int]) -> int:
            if explicit is not None:
                return explicit
            if name in persisted_ids:
                return int(persisted_ids[name])
            return _praxis_default(name)

        self.PAD_ID = _resolve("pad_token_id", pad_token_id)
        self.BOS_ID = _resolve("bos_token_id", bos_token_id)
        self.EOS_ID = _resolve("eos_token_id", eos_token_id)
        self.SEP_ID = _resolve("sep_token_id", sep_token_id)

        self._special_id_map = {
            self.SPECIAL_TOKEN_STRINGS["pad_token"]: self.PAD_ID,
            self.SPECIAL_TOKEN_STRINGS["bos_token"]: self.BOS_ID,
            self.SPECIAL_TOKEN_STRINGS["eos_token"]: self.EOS_ID,
            self.SPECIAL_TOKEN_STRINGS["sep_token"]: self.SEP_ID,
        }
        self._id_to_special = {v: k for k, v in self._special_id_map.items()}

        # Characters start one past the largest special id to avoid collision.
        self._offset = max(self._special_id_map.values()) + 1

        for name, value in self.SPECIAL_TOKEN_STRINGS.items():
            kwargs.setdefault(name, value)
        # HF requires an unk_token; reuse PAD rather than add a 5th special.
        kwargs.setdefault("unk_token", self.SPECIAL_TOKEN_STRINGS["pad_token"])

        self.add_bos = add_bos
        self.add_eos = add_eos

        self._observed: Set[int] = set(persisted_observed)
        # Seed with printable ASCII so the observability number is honest
        # even before any text has been encoded.
        self._observed.update(range(0x20, 0x7F))

        super().__init__(**kwargs)

        self.chat_template = (
            chat_template if chat_template is not None else get_chat_template("default")
        )

    @property
    def vocab_size(self) -> int:
        return BMP_SIZE + self._offset

    @property
    def offset(self) -> int:
        return self._offset

    def get_vocab(self) -> Dict[str, int]:
        vocab: Dict[str, int] = dict(self._special_id_map)
        for cp in range(BMP_SIZE):
            vocab[chr(cp)] = cp + self._offset
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Break text into per-codepoint tokens, splitting supplementary
        codepoints into a UTF-16 surrogate pair of two tokens."""
        tokens: List[str] = []
        specials = list(self._special_id_map.keys())
        i = 0
        while i < len(text):
            matched = None
            for sp in specials:
                if text.startswith(sp, i):
                    matched = sp
                    break
            if matched is not None:
                tokens.append(matched)
                i += len(matched)
                continue
            ch = text[i]
            cp = ord(ch)
            if cp < BMP_SIZE:
                tokens.append(ch)
            else:
                hi, lo = _codepoint_to_surrogates(cp)
                tokens.append(chr(hi))
                tokens.append(chr(lo))
            i += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_id_map:
            return self._special_id_map[token]
        if len(token) != 1:
            return self.PAD_ID
        cp = ord(token)
        if cp < BMP_SIZE:
            self._observed.add(cp)
            return cp + self._offset
        return self.PAD_ID  # non-BMP should have been split in _tokenize

    def _convert_id_to_token(self, index: int) -> str:
        """Return the raw single-character token for ``index``.

        Surrogate characters are returned as-is. Pairing happens in
        :meth:`convert_tokens_to_string`; unpaired surrogates at that
        stage fall back to UNK.
        """
        if index in self._id_to_special:
            return self._id_to_special[index]
        if self._offset <= index < self._offset + BMP_SIZE:
            return chr(index - self._offset)
        return self.SPECIAL_TOKEN_STRINGS["pad_token"]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Join character tokens, re-pairing UTF-16 surrogates.

        A high surrogate followed by a low surrogate is combined into
        the original supplementary codepoint. Any lone surrogate is
        replaced with UNK so the resulting string is always valid
        UTF-8.
        """
        unk = self.SPECIAL_TOKEN_STRINGS["pad_token"]
        out: List[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if len(tok) == 1:
                cp = ord(tok)
                if _is_high_surrogate(cp):
                    if i + 1 < len(tokens) and len(tokens[i + 1]) == 1 and _is_low_surrogate(ord(tokens[i + 1])):
                        out.append(chr(_surrogates_to_codepoint(cp, ord(tokens[i + 1]))))
                        i += 2
                        continue
                    out.append(unk)
                    i += 1
                    continue
                if _is_low_surrogate(cp):
                    out.append(unk)
                    i += 1
                    continue
            out.append(tok)
            i += 1
        return "".join(out)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        result = list(token_ids_0)
        if self.add_bos:
            result = [self.BOS_ID] + result
        if self.add_eos:
            result = result + [self.EOS_ID]
        if token_ids_1 is not None:
            result = result + [self.SEP_ID] + list(token_ids_1)
            if self.add_eos:
                result = result + [self.EOS_ID]
        return result

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + "char_vocab.json"
        out = save_path / name
        payload = {
            "vocab_size": self.vocab_size,
            "bmp_size": BMP_SIZE,
            "offset": self._offset,
            "special_token_strings": self.SPECIAL_TOKEN_STRINGS,
            "special_token_ids": {
                "pad_token_id": self.PAD_ID,
                "bos_token_id": self.BOS_ID,
                "eos_token_id": self.EOS_ID,
                "sep_token_id": self.SEP_ID,
            },
            "observed_codepoints": sorted(self._observed),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return (str(out),)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, Path], **kwargs
    ) -> "CharLevelTokenizer":
        path = Path(pretrained_model_name_or_path)
        vocab_file = None
        if path.is_dir():
            candidate = path / "char_vocab.json"
            if candidate.exists():
                vocab_file = str(candidate)
        elif path.is_file():
            vocab_file = str(path)
        return cls(vocab_file=vocab_file, **kwargs)

    def train(self, texts, vocab_size: Optional[int] = None, **kwargs) -> None:
        """No-op: vocab is fixed at BMP size; observed set updates lazily on encode."""
        return
