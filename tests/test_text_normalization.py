"""Tests for whitespace-escape normalization in training data.

Some upstream datasets flatten strings during export so real newlines get
serialized as literal backslash-n pairs. Without this step the tokenizer
happily encodes ``"\\n\\n"`` as four characters and the model learns to
emit the literal sequence instead of a paragraph break.
"""

from praxis.data.formatters import normalize_escaped_whitespace, text_formatter


def test_flattened_paragraphs_are_unescaped():
    got = normalize_escaped_whitespace("one\\n\\ntwo")
    assert got == "one\n\ntwo"


def test_text_with_real_newlines_is_left_alone():
    # Mixed content is ambiguous (often code discussing escapes), so the
    # normalizer stays out of the way once a real newline is present.
    src = "real\nnewline\\nliteral"
    assert normalize_escaped_whitespace(src) == src


def test_plain_text_without_escapes_is_unchanged():
    assert normalize_escaped_whitespace("plain text") == "plain text"


def test_double_backslash_is_preserved():
    # ``\\n`` in source is an intentional literal backslash followed by n;
    # must not collapse into a newline.
    src = "code: \\\\n means newline"
    assert normalize_escaped_whitespace(src) == src


def test_tabs_and_carriage_returns_are_unescaped():
    assert normalize_escaped_whitespace("col1\\tcol2") == "col1\tcol2"
    assert normalize_escaped_whitespace("a\\rb") == "a\rb"


def test_empty_input_returns_empty():
    assert normalize_escaped_whitespace("") == ""


def test_text_formatter_applies_normalization_first():
    # A fully flattened paragraph should come out with real newlines even
    # though the caller passed in escape sequences.
    out = text_formatter("First paragraph.\\n\\nSecond paragraph.")
    assert "\n\n" in out
    assert "\\n" not in out
