"""TokenMonster pretrained vocabularies as a Praxis tokenizer.

Wraps the tokenmonster client behind the standard Praxis tokenizer
interface. The TM vocab is never modified; Praxis special tokens live in
a reserved low-id block and all TM ids are shifted up by a fixed offset.
"""

import inspect
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer

from praxis.configuration import PraxisConfig
from praxis.integrations.base import BaseIntegration, IntegrationSpec
from praxis.tokenizers.base import PraxisTokenizerBase, PraxisToolTokensMixin
from praxis.tokenizers.chat_templates import get_chat_template

# Prebuilt vocab sizes published by TokenMonster.
PRESET_SIZES = [1024, 2048, 4096, 8000, 16000, 24000, 32000, 40000, 50256, 65536, 100256]

# The supported base/mode profiles, each a registry key. "tokenmonster"
# aliases the englishcode profile.
VARIANTS = {
    "tokenmonster:english-clean": ("english", "clean"),
    "tokenmonster:englishcode-consistent": ("englishcode", "consistent"),
    "tokenmonster:code-consistent": ("code", "consistent"),
}
DEFAULT_DATASET = "englishcode"
DEFAULT_MODE = "consistent"

LOCAL_DIR = "./build/tokenmonster"

# One client process per interpreter; all vocab calls serialize through a
# single pipe, so guard them with one lock. State is rebuilt wholesale in
# forked children: the client's own reconnect path replays a stale vocab id
# and hangs, so we drop the inherited handle and load fresh instead.
_TM_LOCK = threading.Lock()
_TM_PID = os.getpid()
_TM_CONFIGURED = False


def _load_tm_vocab(vocab_name: str):
    """Load a TokenMonster vocab (multiprocess-safe; we never modify it)."""
    global _TM_CONFIGURED
    import tokenmonster

    if not _TM_CONFIGURED:
        tokenmonster.set_local_directory(LOCAL_DIR)
        _TM_CONFIGURED = True
    return tokenmonster.load_multiprocess_safe(vocab_name)


def _ensure_this_process():
    """After a fork, discard the inherited server handle and lock."""
    global _TM_PID, _TM_LOCK
    if os.getpid() != _TM_PID:
        import tokenmonster

        _TM_LOCK = threading.Lock()  # inherited lock may be held by a dead owner
        tokenmonster.Vocab._process = None  # parent's pipe is not ours
        _TM_PID = os.getpid()


def resolve_vocab_name(
    vocab_size: int,
    dataset: str = DEFAULT_DATASET,
    mode: str = DEFAULT_MODE,
) -> str:
    """Pick a prebuilt vocab: exact size, else largest preset <= target."""
    if vocab_size not in PRESET_SIZES:
        fits = [s for s in PRESET_SIZES if s <= vocab_size]
        vocab_size = max(fits) if fits else min(PRESET_SIZES)
    return f"{dataset}-{vocab_size}-{mode}-v1"


def _praxis_default(name: str) -> int:
    return inspect.signature(PraxisConfig.__init__).parameters[name].default


class TokenMonsterTokenizer(
    PreTrainedTokenizer, PraxisToolTokensMixin, PraxisTokenizerBase
):
    """TokenMonster vocab behind the HF slow-tokenizer interface.

    Token id layout:
        0..3   -> [PAD] [BOS] [EOS] [SEP] (PraxisConfig defaults)
        4..7   -> the 4 tool-control tokens
        8..    -> TokenMonster ids, shifted by ``self._offset``

    Internal token strings are decimal ids ("8123"); text only exists at
    the encode/decode boundary because TM owns capcode and byte merges.
    """

    SPECIAL_TOKEN_STRINGS = {
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "sep_token": "[SEP]",
    }

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "tokenmonster_vocab.json"}

    def __init__(
        self,
        vocab_name: str = f"{DEFAULT_DATASET}-32000-{DEFAULT_MODE}-v1",
        vocab_file: Optional[str] = None,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        if vocab_file is not None and Path(vocab_file).exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_name = json.load(f)["vocab_name"]

        self.vocab_name = vocab_name
        self._pid = os.getpid()
        self._tm = _load_tm_vocab(vocab_name)
        self._tm_size = int(self._tm.vocab_size)

        self.PAD_ID = _praxis_default("pad_token_id")
        self.BOS_ID = _praxis_default("bos_token_id")
        self.EOS_ID = _praxis_default("eos_token_id")
        self.SEP_ID = _praxis_default("sep_token_id")

        self._special_id_map = {
            self.SPECIAL_TOKEN_STRINGS["pad_token"]: self.PAD_ID,
            self.SPECIAL_TOKEN_STRINGS["bos_token"]: self.BOS_ID,
            self.SPECIAL_TOKEN_STRINGS["eos_token"]: self.EOS_ID,
            self.SPECIAL_TOKEN_STRINGS["sep_token"]: self.SEP_ID,
        }
        next_id = max(self._special_id_map.values()) + 1
        for idx, tok in enumerate(self.TOOL_SPECIAL_TOKEN_STRINGS):
            self._special_id_map[tok] = next_id + idx
        self._id_to_special = {v: k for k, v in self._special_id_map.items()}
        self._offset = max(self._special_id_map.values()) + 1

        for name, value in self.SPECIAL_TOKEN_STRINGS.items():
            kwargs.setdefault(name, value)
        # HF requires an unk_token; reuse PAD rather than add a 5th named special.
        kwargs.setdefault("unk_token", self.SPECIAL_TOKEN_STRINGS["pad_token"])
        self._inject_tool_tokens_kwargs(kwargs)

        super().__init__(**kwargs)

        self.chat_template = (
            chat_template if chat_template is not None else get_chat_template("default")
        )

    @property
    def vocab_size(self) -> int:
        return self._tm_size + self._offset

    @property
    def offset(self) -> int:
        return self._offset

    def get_vocab(self) -> Dict[str, int]:
        vocab: Dict[str, int] = dict(self._special_id_map)
        for i in range(self._offset, self.vocab_size):
            vocab[str(i)] = i
        return vocab

    def _vocab(self):
        """Current-process vocab handle, reloading after a fork."""
        if os.getpid() != self._pid:
            _ensure_this_process()
            # Disarm the inherited handle: its __del__ would fire an unload
            # job with a stale id into the fresh pipe (ids get reused, so it
            # can unload the replacement vocab).
            if self._tm is not None:
                self._tm._modified_id = -1
            self._tm = _load_tm_vocab(self.vocab_name)
            self._pid = os.getpid()
        return self._tm

    def _tm_encode(self, text: str) -> List[int]:
        if not text:
            return []
        vocab = self._vocab()
        with _TM_LOCK:
            ids = vocab.tokenize(text)
        if ids is None:
            return []
        return [int(i) + self._offset for i in ids.tolist()]

    def _tm_encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encode; segments tokenize in parallel server-side."""
        nonempty = [t for t in texts if t]
        if not nonempty:
            return [[] for _ in texts]
        vocab = self._vocab()
        with _TM_LOCK:
            results = vocab.tokenize(nonempty)
        if len(nonempty) == 1:
            results = [results]
        out: List[List[int]] = []
        it = iter(results)
        for t in texts:
            if not t:
                out.append([])
                continue
            ids = next(it)
            out.append(
                [] if ids is None else [int(i) + self._offset for i in ids.tolist()]
            )
        return out

    def _tm_decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        vocab = self._vocab()
        with _TM_LOCK:
            return vocab.decode([i - self._offset for i in ids])

    def _split_specials(self, text: str) -> List[Tuple[bool, str]]:
        """Split text into (is_special, piece) runs around special tokens."""
        pieces: List[Tuple[bool, str]] = []
        specials = list(self._special_id_map.keys())
        segment_start = 0
        i = 0
        while i < len(text):
            matched = None
            for sp in specials:
                if text.startswith(sp, i):
                    matched = sp
                    break
            if matched is not None:
                if i > segment_start:
                    pieces.append((False, text[segment_start:i]))
                pieces.append((True, matched))
                i += len(matched)
                segment_start = i
            else:
                i += 1
        if segment_start < len(text):
            pieces.append((False, text[segment_start:]))
        return pieces

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Split out special-token strings, TM-tokenize the segments."""
        tokens: List[str] = []
        for is_special, piece in self._split_specials(text):
            if is_special:
                tokens.append(piece)
            else:
                tokens.extend(map(str, self._tm_encode(piece)))
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._special_id_map:
            return self._special_id_map[token]
        try:
            return int(token)
        except ValueError:
            return self.PAD_ID

    def _convert_id_to_token(self, index: int) -> str:
        if index in self._id_to_special:
            return self._id_to_special[index]
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Decode runs of TM ids through the vocab; specials pass through."""
        out: List[str] = []
        run: List[int] = []
        for tok in tokens:
            if tok in self._special_id_map:
                out.append(self._tm_decode(run))
                run = []
                out.append(tok)
            else:
                try:
                    run.append(int(tok))
                except ValueError:
                    continue
        out.append(self._tm_decode(run))
        return "".join(out)

    def apply_chat_template(self, conversation, **kwargs):
        """Build assistant masks without char_to_token (fast-tokenizer only).

        The template's {% generation %} spans are mapped to tokens by
        encoding the rendered text segment-wise at span boundaries. Those
        boundaries match the inference-time prompt/generation split, so
        the tokenization stays in-distribution.
        """
        if not kwargs.get("return_assistant_tokens_mask"):
            return super().apply_chat_template(conversation, **kwargs)
        if not (kwargs.get("tokenize") and kwargs.get("return_dict")):
            raise ValueError(
                "return_assistant_tokens_mask=True requires tokenize=True "
                "and return_dict=True"
            )
        if conversation and not isinstance(conversation[0], dict):
            raise NotImplementedError(
                "TokenMonsterTokenizer assistant masks support a single "
                "conversation, not a batch"
            )

        from transformers.tokenization_utils_base import BatchEncoding
        from transformers.utils.chat_template_utils import render_jinja_template

        template_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "tokenize",
                "return_dict",
                "return_assistant_tokens_mask",
                "return_tensors",
                "padding",
                "truncation",
                "max_length",
            )
        }
        rendered, generation_indices = render_jinja_template(
            conversations=[conversation],
            chat_template=self.chat_template,
            return_assistant_tokens_mask=True,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            sep_token=self.sep_token,
            pad_token=self.pad_token,
            **template_kwargs,
        )
        text, spans = rendered[0], generation_indices[0]

        # Alternating non-generation / generation segments.
        segments: List[Tuple[str, int]] = []
        cursor = 0
        for start, end in spans:
            segments.append((text[cursor:start], 0))
            segments.append((text[start:end], 1))
            cursor = end
        segments.append((text[cursor:], 0))

        input_ids: List[int] = []
        assistant_mask: List[int] = []
        for tokens, (_, flag) in zip(
            self._encode_segments([s for s, _ in segments]), segments
        ):
            input_ids.extend(tokens)
            assistant_mask.extend([flag] * len(tokens))

        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "assistant_masks": assistant_mask,
            }
        )

    def _encode_segments(self, texts: List[str]) -> List[List[int]]:
        """Encode segments with all TM runs batched into one server call."""
        pieces_per_text = [self._split_specials(t) for t in texts]
        tm_runs = [
            piece
            for pieces in pieces_per_text
            for is_special, piece in pieces
            if not is_special
        ]
        encoded_runs = iter(self._tm_encode_batch(tm_runs))
        out: List[List[int]] = []
        for pieces in pieces_per_text:
            ids: List[int] = []
            for is_special, piece in pieces:
                if is_special:
                    ids.append(self._special_id_map[piece])
                else:
                    ids.extend(next(encoded_runs))
            out.append(ids)
        return out

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        result = list(token_ids_0)
        if token_ids_1 is not None:
            result = result + [self.SEP_ID] + list(token_ids_1)
        return result

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        name = (
            filename_prefix + "-" if filename_prefix else ""
        ) + "tokenmonster_vocab.json"
        out = save_path / name
        payload = {
            "vocab_name": self.vocab_name,
            "vocab_size": self.vocab_size,
            "offset": self._offset,
            "special_token_ids": {
                "pad_token_id": self.PAD_ID,
                "bos_token_id": self.BOS_ID,
                "eos_token_id": self.EOS_ID,
                "sep_token_id": self.SEP_ID,
            },
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return (str(out),)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, Path], **kwargs
    ) -> "TokenMonsterTokenizer":
        path = Path(pretrained_model_name_or_path)
        vocab_file = None
        if path.is_dir():
            candidate = path / "tokenmonster_vocab.json"
            if candidate.exists():
                vocab_file = str(candidate)
        elif path.is_file():
            vocab_file = str(path)
        return cls(vocab_file=vocab_file, **kwargs)

    def train(self, texts, vocab_size: Optional[int] = None, **kwargs) -> None:
        """No-op: TokenMonster vocabs are pretrained and immutable here."""
        return


def create_tokenmonster_tokenizer(
    vocab_size: int = 32768,
    dataset: str = DEFAULT_DATASET,
    mode: str = DEFAULT_MODE,
    **kwargs,
):
    return TokenMonsterTokenizer(
        vocab_name=resolve_vocab_name(vocab_size, dataset, mode), **kwargs
    )


def _register() -> None:
    """Register TM variants and extend the valid --vocab-size values."""
    import functools

    from praxis.tokenizers import TOKENIZER_REGISTRY, VOCAB_SIZE_CHOICES

    TOKENIZER_REGISTRY.setdefault("tokenmonster", create_tokenmonster_tokenizer)
    for key, (dataset, mode) in VARIANTS.items():
        TOKENIZER_REGISTRY.setdefault(
            key,
            functools.partial(create_tokenmonster_tokenizer, dataset=dataset, mode=mode),
        )
    for size in PRESET_SIZES:
        if size not in VOCAB_SIZE_CHOICES:
            VOCAB_SIZE_CHOICES.append(size)
    VOCAB_SIZE_CHOICES.sort()


# Module import happens during integration discovery, before CLI argument
# groups are built, so the registry keys land in --tokenizer-type choices
# and the preset sizes land in --vocab-size choices (visible in --help).
_register()


class Integration(BaseIntegration):
    """Registers TokenMonster vocabs in the tokenizer registry."""

    def __init__(self, spec: IntegrationSpec):
        super().__init__(spec)

    def initialize(
        self,
        args: Any,
        cache_dir: str,
        ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        try:
            import tokenmonster

            tokenmonster.disconnect()
        except Exception:
            pass
