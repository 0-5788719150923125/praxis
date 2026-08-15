"""ByteLevel tokenizer implementation for Praxis with HuggingFace compatibility."""

from typing import Any, Dict, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .base import PraxisToolTokensMixin
from .chat_templates import get_chat_template

# Try to import BltTokenizer, but make it optional
try:
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer

    HAS_BLT = True
except ImportError:
    HAS_BLT = False
    BltTokenizer = None

# Import byte constants from our local module
try:
    from praxis.encoders.byte_latent.constants import OFFSET
except ImportError:
    # Fallback if the encoder module isn't available
    OFFSET = 4


class ByteLevelTokenizer(PreTrainedTokenizerFast, PraxisToolTokensMixin):
    """
    ByteLevel tokenizer that operates on raw bytes.

    This tokenizer provides byte-level tokenization with full HuggingFace
    compatibility including chat template support.

    Tool-control tokens (see ``PraxisToolTokensMixin``) are appended
    past the byte range so existing byte ids stay valid in saved
    checkpoints. They are registered only when the chat format actually
    renders them, because ``byte_alphabet_size`` sizes the model's output
    head: an unused control token is a sampleable logit with no training
    signal behind it.

    The four NAMED specials below the byte range follow the same rule, and
    that is what makes a pure 256-byte alphabet possible. A format that
    renders no control token and asks the packer to write no document
    separator (``prose``) gets ``byte_offset == 0``: ids ARE bytes, the
    alphabet is exactly 256, and nothing in the stream is outside the set a
    byte can represent. A format with token boundaries (``default``) gets the
    BLT layout - ``byte_offset == 4``, ids 0-3 reserved for
    ``[PAD]``/``[BOS]``/``[EOS]``/``[SEP]``, bytes at ``b + 4``.

    The offset is therefore an instance property, not a module constant.
    Everything downstream that does byte arithmetic reads it from here
    (via ``ByteLatentConfig.byte_offset``) rather than importing ``OFFSET``.
    """

    # Only 4 unique named special tokens matching BLT IDs exactly
    # PAD_ID/BOE_ID = 0, BOS_ID = 1, EOS_ID = 2, SEP_ID = 3
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",  # ID: 0
        "bos_token": "[BOS]",  # ID: 1
        "eos_token": "[EOS]",  # ID: 2
        "sep_token": "[SEP]",  # ID: 3
        # UNK is not used in ByteLatent
    }

    # No merges: every byte encodes independently, so tokenizing text in
    # pieces and concatenating equals tokenizing the whole. That is what
    # lets the assistant mask be built segment-wise instead of through
    # HuggingFace's character-offset mapping, which slips on multi-byte
    # characters here (see chat_templates.tokenize_with_mask).
    context_free_tokenization = True

    # Special token IDs matching BLT constants
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    SEP_ID = 3

    def __init__(
        self,
        tokenizer_object: Optional[Tokenizer] = None,
        vocab_size_unit_1: int = 256,
        vocab_size: Optional[int] = None,
        bpe_delim: bool = False,
        add_bos: bool = False,  # Don't add BOS by default for byte-level
        add_eos: bool = False,  # Don't add EOS by default for byte-level
        chat_template: Optional[str] = None,
        chat_format: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the ByteLevel tokenizer.

        Args:
            tokenizer_object: Pre-built tokenizer object
            vocab_size_unit_1: Size of the byte vocabulary (default 256)
            vocab_size: External (model) vocab size reported by the
                ``vocab_size`` property. The byte-level tokenizer only emits
                ids in ``byte_alphabet_size`` (256 bytes + named specials +
                tool specials), but downstream code (hash embeddings,
                dashboards) treats ``vocab_size`` as the model's logical
                vocabulary. If unset, falls back to the byte alphabet size.
            bpe_delim: Whether to use BPE delimiter (requires BLT)
            add_bos: Whether to add BOS token automatically
            add_eos: Whether to add EOS token automatically
            chat_template: Chat template for conversation formatting
            chat_format: ``ChatFormat`` (or registry name) the tokenizer will
                be used with. Decides whether the tool-control tokens are
                registered at all, and therefore ``byte_alphabet_size``.
                ``None`` resolves to the default format, keeping all 264 ids.
            **kwargs: Additional arguments passed to parent class
        """
        self.vocab_size_unit_1 = vocab_size_unit_1
        self._configured_vocab_size = vocab_size
        self.bpe_delim = bpe_delim
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Named control tokens exist only for formats that actually use them.
        # When they do not, the alphabet is pure bytes and the offset is 0.
        self._wants_named = self._wants_named_specials(chat_format)
        self._offset = OFFSET if self._wants_named else 0

        # Tool-control token ids land immediately past the byte range - but
        # only when the format renders them. Empty map = the ids don't exist,
        # which every vocab/encode/decode path below reads from.
        self._tool_token_start_id = vocab_size_unit_1 + self._offset
        self._tool_special_id_map: Dict[str, int] = (
            {
                tok: self._tool_token_start_id + idx
                for idx, tok in enumerate(self.TOOL_SPECIAL_TOKEN_STRINGS)
            }
            if self._wants_tool_tokens(chat_format)
            else {}
        )

        # Initialize named-special tokens from kwargs or use defaults. Skipped
        # entirely in the pure-byte layout: registering a token HuggingFace can
        # then hand back as `pad_token_id` would put an id outside 0..255 back
        # into play, which is the whole thing this layout removes.
        if self._wants_named:
            for token_name, token_value in self.SPECIAL_TOKENS.items():
                if token_name not in kwargs:
                    kwargs[token_name] = token_value

            # HuggingFace expects unk_token - use PAD as fallback
            if "unk_token" not in kwargs:
                kwargs["unk_token"] = "[PAD]"  # Use PAD as unknown token

        # Register tool tokens so ``skip_special_tokens`` recognises them.
        self._inject_tool_tokens_kwargs(kwargs, chat_format)

        # Create or use provided tokenizer object
        if tokenizer_object is None:
            tokenizer_object = self._create_tokenizer(vocab_size_unit_1)

        # Initialize PreTrainedTokenizerFast (this gives us apply_chat_template!)
        PreTrainedTokenizerFast.__init__(
            self, tokenizer_object=tokenizer_object, **kwargs
        )

        # Set chat template
        if chat_template is None:
            self.chat_template = get_chat_template("byte_level")
        else:
            self.chat_template = chat_template

        # Legacy BLT tokenizer for compatibility (if needed)
        self._blt_tokenizer = None
        if bpe_delim and HAS_BLT:
            self._blt_tokenizer = BltTokenizer(
                vocab_size_unit_1=vocab_size_unit_1,
                bpe_delim=bpe_delim,
                add_bos=add_bos,
                add_eos=add_eos,
            )

    def _create_tokenizer(self, vocab_size: int) -> Tokenizer:
        """
        Create a tokenizer object for BLT-compatible byte-level encoding.

        Args:
            vocab_size: Size of the byte vocabulary

        Returns:
            Tokenizer object configured for byte-level encoding
        """
        # Build vocab with BLT-compatible IDs
        vocab = {}
        merges = []

        # Named special tokens use IDs 0-3, when the layout has them at all
        if self._wants_named:
            vocab["[PAD]"] = self.PAD_ID
            vocab["[BOS]"] = self.BOS_ID
            vocab["[EOS]"] = self.EOS_ID
            vocab["[SEP]"] = self.SEP_ID

        # Byte tokens start at the offset (4 with named specials, else 0)
        for i in range(vocab_size):
            byte_char = chr(i) if i < 256 else f"<unk>"
            vocab[byte_char] = i + self._offset

        # Tool-control tokens appended past the byte range.
        for tok, tid in self._tool_special_id_map.items():
            vocab[tok] = tid

        # Create BPE model with vocab and empty merges
        tokenizer = Tokenizer(
            models.BPE(vocab=vocab, merges=merges, byte_fallback=False)
        )

        # Use simple pre-tokenizer that preserves everything
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        tokenizer.decoder = decoders.Sequence([])

        return tokenizer

    def train(self, texts, vocab_size: int = 32768, **kwargs) -> None:
        """
        Train the tokenizer on a corpus of texts.

        Note: ByteLevel tokenizer doesn't need training as it operates
        on raw bytes. This method is a no-op but provided for interface
        compatibility.
        """
        pass  # ByteLevel tokenizer doesn't need training

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> List[int]:
        """
        Encode text to token IDs using BLT-compatible offset system.

        Bytes are offset by OFFSET (4), special tokens use IDs 0-3.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            **kwargs: Additional arguments

        Returns:
            List of token IDs
        """
        # For testing: handle special token strings in text
        # This allows tests to pass "[BOS]Hello[EOS]" and have it work
        special_token_map = {**self._named_special_map(), **self._tool_special_id_map}

        tokens = []
        i = 0
        while i < len(text):
            # Check if we're at a special token string
            found_special = False
            for token_str, token_id in special_token_map.items():
                if text[i : i + len(token_str)] == token_str:
                    tokens.append(token_id)
                    i += len(token_str)
                    found_special = True
                    break

            if not found_special:
                # Convert character to UTF-8 bytes and add offset for each byte
                char_bytes = text[i].encode("utf-8")
                for b in char_bytes:
                    tokens.append(b + self._offset)
                i += 1

        # Add BOS/EOS if requested (and not already present)
        if add_special_tokens and self._wants_named:
            # When add_special_tokens=True, always add special tokens
            # (this overrides the tokenizer's default settings).
            #
            # Ignored outright in the pure-byte layout: ids 1 and 2 are the
            # bytes 0x01 and 0x02 there, so "adding BOS/EOS" would splice two
            # arbitrary control characters into the text.
            if not tokens or tokens[0] != self.BOS_ID:
                tokens.insert(0, self.BOS_ID)
            if not tokens or tokens[-1] != self.EOS_ID:
                tokens.append(self.EOS_ID)

        return tokens

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Decode token IDs back to text using BLT-compatible offset system.

        For testing compatibility, special tokens are decoded to their string form.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments

        Returns:
            Decoded text
        """
        # Handle tensors
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]

        # Special token ID to string mapping (named specials + tool specials)
        special_id_map = {
            **{tid: tok for tok, tid in self._named_special_map().items()},
            **{tid: tok for tok, tid in self._tool_special_id_map.items()},
        }

        result = []
        byte_buffer = bytearray()
        for tok in token_ids:
            if tok in special_id_map:
                # Flush any pending bytes before the special token
                if byte_buffer:
                    result.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer = bytearray()
                if not skip_special_tokens:
                    result.append(special_id_map[tok])
            else:
                # Regular byte token - reverse the offset
                byte_val = tok - self._offset
                if 0 <= byte_val < 256:
                    byte_buffer.append(byte_val)

        # Flush remaining bytes
        if byte_buffer:
            result.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(result)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text into a list of token strings.

        Args:
            text: Text to tokenize
            **kwargs: Additional arguments

        Returns:
            List of token strings
        """
        token_ids = self.encode(text, add_special_tokens=False)

        # Map special token IDs to strings (named + tool specials)
        special_id_map = {
            **{tid: tok for tok, tid in self._named_special_map().items()},
            **{tid: tok for tok, tid in self._tool_special_id_map.items()},
        }

        tokens = []
        for token_id in token_ids:
            if token_id in special_id_map:
                tokens.append(special_id_map[token_id])
            else:
                # Regular byte token - represent as hex byte string
                byte_val = token_id - self._offset
                if 0 <= byte_val < 256:
                    tokens.append(f"<0x{byte_val:02X}>")

        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back to a string. Hex byte tokens
        (e.g. ``<0x48>``) are decoded back to their original UTF-8 bytes.
        """
        byte_buf = bytearray()
        out: List[str] = []

        def flush() -> None:
            if byte_buf:
                out.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tok in tokens:
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    byte_buf.append(int(tok[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush()
            out.append(tok)

        flush()
        return "".join(out)

    # UTF-8 lead byte -> total sequence length. A byte outside these ranges is
    # either ASCII (complete on its own) or a continuation.
    _UTF8_LEAD_LENGTHS = ((0xC0, 0xE0, 2), (0xE0, 0xF0, 3), (0xF0, 0xF8, 4))

    def _byte_at(self, token_ids: List[int], index: int) -> Optional[int]:
        """The raw byte a position holds, or None if it holds a special id."""
        if index < 0 or index >= len(token_ids):
            return None
        value = int(token_ids[index]) - self._offset
        return value if 0 <= value < 256 else None

    def incomplete_tail(self, token_ids: List[int]) -> bool:
        """Whether the sequence stops partway through a UTF-8 character.

        The rolling contexts (StreamingContext) are TEXT buffers: each round
        decodes the whole sequence, hands the string back as the next prompt,
        and re-encodes it. A generation step commits only a few bytes, so it
        regularly ends on the lead byte of a multi-byte character - which
        ``decode`` turns into U+FFFD, and re-encoding turns that single
        replacement character into three real bytes (EF BF BD) that stay in the
        prompt for as long as the buffer lives. One truncated character becomes
        a permanent artifact the model then conditions on and imitates.

        The generator asks for a few more tokens whenever this is True (see
        ``Generator._process_single_request``), so the character completes
        instead. Only a VALID prefix counts: bytes that cannot be finished no
        matter what follows are reported complete, or the strip loop below would
        eat the whole buffer chasing a tail that can never resolve.
        """
        # 4 bytes is the longest UTF-8 sequence, so the lead is within reach.
        for back in range(min(4, len(token_ids))):
            byte = self._byte_at(token_ids, len(token_ids) - 1 - back)
            if byte is None or byte < 0x80:
                return False  # special token or ASCII: a complete boundary
            if byte < 0xC0:
                continue  # continuation byte; keep walking back to the lead
            for low, high, length in self._UTF8_LEAD_LENGTHS:
                if low <= byte < high:
                    return back + 1 < length
            return False  # 0xF8..0xFF is not a legal lead
        return False

    def strip_incomplete_tail(self, token_ids: List[int]) -> List[int]:
        """Drop a trailing partial character so the sequence ends on a boundary.

        The fallback for when asking for more tokens did not complete it.
        Bounded by construction: a UTF-8 prefix is at most 3 bytes short.
        """
        ids = list(token_ids)
        while ids and self.incomplete_tail(ids):
            ids.pop()
        return ids

    def align_left_cut(self, token_ids: List[int], start: int) -> int:
        """Move a left-truncation point forward onto a character boundary.

        Prompts are truncated by TOKEN index (``Generator._prepare_inputs``),
        and here a token is one byte - so an unaligned cut lands inside a
        multi-byte character about a third of the time on text that has any.
        The severed continuation bytes decode to U+FFFD, which the rolling
        context re-encodes into three real bytes; those make the buffer longer
        in bytes than in characters, which guarantees the NEXT round truncates
        too. That is the ratchet: once one replacement character appears, the
        context keeps minting more.

        Advancing past at most three continuation bytes costs a character and
        ends it.
        """
        limit = min(start + 4, len(token_ids))
        while start < limit:
            byte = self._byte_at(token_ids, start)
            if byte is None or byte < 0x80 or byte >= 0xC0:
                break  # special token, ASCII, or a lead byte: a valid start
            start += 1
        return start

    @staticmethod
    def _wants_named_specials(chat_format: Any = None) -> bool:
        """Whether this format needs the four named control ids to exist.

        True when the format writes control tokens into the rendered text
        (token boundaries), when it renders the atomic tool tokens, or when it
        asks the packer for a document separator. False leaves a pure 256-byte
        alphabet, which is only sound because document boundaries now travel
        out-of-band (MessageQueueManager emits ``block_ids``) rather than as a
        separator id the model has to see.
        """
        from .chat_templates import get_chat_format

        fmt = get_chat_format(chat_format)
        return (
            not fmt.text_boundaries
            or fmt.uses_tool_tokens
            or fmt.document_separator is not None
        )

    def _named_special_map(self) -> Dict[str, int]:
        """``{token string: id}`` for the named specials, empty in byte mode."""
        if not self._wants_named:
            return {}
        return {
            "[PAD]": self.PAD_ID,
            "[BOS]": self.BOS_ID,
            "[EOS]": self.EOS_ID,
            "[SEP]": self.SEP_ID,
        }

    @property
    def byte_offset(self) -> int:
        """Amount added to a raw byte to get its id: 4 (BLT layout) or 0.

        Everything downstream that does byte arithmetic - the space patcher's
        character classes, the byte-latent embedding tables - must read this
        rather than assume a constant.
        """
        return self._offset

    @property
    def byte_alphabet_size(self) -> int:
        """Number of distinct ids the tokenizer can emit (bytes + named
        specials + tool specials ACTUALLY registered). Use this to size
        byte-level embedding tables; ``vocab_size`` is the model's external
        vocab and may be much larger (e.g. for hash-embedding bucket counts).

        Counts the live tool map rather than the class-level string list, so a
        format that skips the tool tokens gets a 260-wide alphabet - and the
        output head shrinks with it (praxis/encoders/byte_latent/config.py)."""
        return self.vocab_size_unit_1 + self._offset + len(self._tool_special_id_map)

    @property
    def vocab_size(self) -> int:
        """Model-facing vocabulary size. Defaults to the byte alphabet
        when no external vocab was configured."""
        if self._configured_vocab_size is not None:
            return int(self._configured_vocab_size)
        return self.byte_alphabet_size
