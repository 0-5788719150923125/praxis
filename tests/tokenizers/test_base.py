"""Tool-calling tests.

Tool-call boundaries are atomic special tokens
(``[TOOL_CALL]``/``[/TOOL_CALL]``/``[TOOL_RESULT]``/``[/TOOL_RESULT]``).
The tests exercise both the string-form helpers (format, parse, regex
patterns) and the token-ID helpers used by the generator at runtime.
"""

import pytest

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.char_level import CharLevelTokenizer
from praxis.tools import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
)

# ---------------------------------------------------------------------------
# Token-ID helpers - the generator's runtime path.
# ---------------------------------------------------------------------------


def test_tokenizer_encodes_tool_tokens_as_single_ids_byte_level():
    tok = ByteLevelTokenizer()
    for marker in (
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
        TOOL_RESULT_OPEN,
        TOOL_RESULT_CLOSE,
    ):
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, f"{marker} not atomic: {ids}"


def test_tokenizer_encodes_tool_tokens_as_single_ids_char_level():
    tok = CharLevelTokenizer()
    for marker in (
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
        TOOL_RESULT_OPEN,
        TOOL_RESULT_CLOSE,
    ):
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, f"{marker} not atomic: {ids}"


def test_skip_special_tokens_strips_tool_markers():
    tok = ByteLevelTokenizer()
    text = f"hello {TOOL_CALL_OPEN}body{TOOL_CALL_CLOSE} done"
    ids = tok.encode(text, add_special_tokens=False)
    with_specials = tok.decode(ids, skip_special_tokens=False)
    without_specials = tok.decode(ids, skip_special_tokens=True)
    assert TOOL_CALL_OPEN in with_specials
    assert TOOL_CALL_OPEN not in without_specials
