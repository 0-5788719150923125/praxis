"""``StreamingContext``: the growing text buffer with reset-on-degeneracy.

Shared by the Lightning ``TerminalInterface`` callback and the Mono-Forward
live-inference hook. These tests cover the degeneracy
heuristics, the stuck-output reset, and the display copy the web renders.
"""

from __future__ import annotations

import pytest

from praxis.generation import StreamingContext


def test_unchanged_text_triggers_reset_after_threshold():
    ctx = StreamingContext(initial_text="<s>", unchanged_threshold=3)
    # First call: previous_texts is empty so no unchanged increment.
    ctx.update("<s>abc")
    assert ctx.unchanged_count == 0
    # Each subsequent identical update increments the counter; reset
    # fires once ``unchanged_count >= unchanged_threshold``.
    ctx.update("<s>abc")
    assert ctx.unchanged_count == 1
    ctx.update("<s>abc")
    assert ctx.unchanged_count == 2
    did_reset = ctx.update("<s>abc")
    assert did_reset is True
    assert ctx.text == "<s>"
    assert ctx.unchanged_count == 0


@pytest.mark.parametrize(
    "kwargs,text",
    [
        # 10 copies of "abc": the "abc" n-gram count is 8, over the threshold.
        (dict(repetition_n_gram_size=3, repetition_frequency=5), "abc" * 10),
        # "foo" * 5: pattern length 3 repeated 5 times, 15 >= min_segment_length.
        ({}, "foofoofoofoofoo"),
        # Bracket-pipe lines ("[tag]|" with >= 4 brackets) on 2 of 3 lines.
        (
            dict(repetition_frequency=1000),
            "[a]|[b]|[c]|[d]\n[e]|[f]|[g]|[h]\nnormal line",
        ),
        ({}, "     \n\n\t  "),
    ],
    ids=["ngram", "sequential", "bracket_pipe", "whitespace"],
)
def test_degenerate_buffer_resets(kwargs, text):
    ctx = StreamingContext(initial_text="seed", **kwargs)
    assert ctx.update(text) is True
    assert ctx.text == "seed"


def test_max_length_left_truncates_buffer():
    ctx = StreamingContext(
        initial_text="<s>",
        max_length=20,
        repetition_frequency=1000,  # disable repetition detection
    )
    long_text = "abcdefghijklmnopqrstuvwxyz0123456789"
    did_reset = ctx.update(long_text)
    assert did_reset is False
    assert len(ctx.text) == 20
    assert ctx.text == long_text[-20:]


def test_healthy_growth_does_not_reset():
    ctx = StreamingContext(initial_text="<s>")
    # A realistic-looking incremental build-up should never reset.
    passages = [
        "<s>The",
        "<s>The quick",
        "<s>The quick brown",
        "<s>The quick brown fox",
        "<s>The quick brown fox jumps",
    ]
    for p in passages:
        assert ctx.update(p) is False
    assert ctx.text == passages[-1]


def test_explicit_reset_clears_history():
    ctx = StreamingContext(initial_text="<s>")
    ctx.update("<s>something")
    ctx.reset()
    assert ctx.text == "<s>"
    assert ctx.unchanged_count == 0


# ---------------------------------------------- display vs prompt line breaks
#
# The CLI dashboard wraps with str.splitlines(); the browser's
# `white-space: pre-wrap` only breaks on LF / CR / CRLF. Python's set is
# strictly larger, so a byte-level model emitting \v, \f, U+0085 or U+2028
# produced a line break in the terminal that silently vanished in the web
# Terminal tab. The DISPLAY copy is normalized so both agree; the PROMPT copy
# must not be, or the model conditions on bytes it never produced.

EXOTIC_BREAKS = ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


@pytest.mark.parametrize("sep", EXOTIC_BREAKS)
def test_display_text_normalizes_breaks_the_browser_ignores(sep):
    """Both renderers must agree on the line count: the CLI counts with
    splitlines(), the browser counts LF (having already collapsed CRLF)."""
    ctx = StreamingContext(initial_text="a")
    ctx.update(f"alpha{sep}beta{sep}gamma")

    # The prompt copy is untouched - byte-exact is the contract - and it is
    # where the two renderers disagree.
    assert ctx.text == f"alpha{sep}beta{sep}gamma"
    assert len(ctx.text.splitlines()) != len(ctx.text.split("\n"))

    cli_lines = ctx.display_text.splitlines()
    browser_lines = ctx.display_text.replace("\r\n", "\n").split("\n")
    assert cli_lines == browser_lines == ["alpha", "beta", "gamma"]


def test_display_text_leaves_ordinary_whitespace_alone():
    ctx = StreamingContext(initial_text="a")
    ctx.update("keep\nthese\r\nand\ttabs  and spaces")
    assert ctx.display_text == "keep\nthese\r\nand\ttabs  and spaces"
