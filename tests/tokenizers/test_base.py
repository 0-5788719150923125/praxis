"""``PraxisToolTokensMixin``: tool-call boundaries as atomic special tokens."""

import pytest

from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.char_level import CharLevelTokenizer
from praxis.tools import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
)


@pytest.mark.parametrize("cls", [ByteLevelTokenizer, CharLevelTokenizer])
@pytest.mark.parametrize(
    "marker", [TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE]
)
def test_tool_markers_encode_as_single_ids(cls, marker):
    ids = cls().encode(marker, add_special_tokens=False)
    assert len(ids) == 1, f"{marker} not atomic: {ids}"


def test_skip_special_tokens_strips_tool_markers():
    tok = ByteLevelTokenizer()
    text = f"hello {TOOL_CALL_OPEN}body{TOOL_CALL_CLOSE} done"
    ids = tok.encode(text, add_special_tokens=False)
    with_specials = tok.decode(ids, skip_special_tokens=False)
    without_specials = tok.decode(ids, skip_special_tokens=True)
    assert TOOL_CALL_OPEN in with_specials
    assert TOOL_CALL_OPEN not in without_specials
