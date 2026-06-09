"""Unit tests for :class:`praxis.generation.StreamingContext`.

The streaming context is the shared helper that drives the
"growing text buffer with reset-on-degeneracy" pattern used by both
the backprop Lightning ``TerminalInterface`` callback and the Ray
Mono-Forward live-inference hook. These tests cover the degeneracy
heuristics and the stuck-output reset path in isolation.
"""

from __future__ import annotations

import pytest

from praxis.generation import StreamingContext


def test_update_appends_and_stores_current_text():
    ctx = StreamingContext(initial_text="<s>")
    assert ctx.text == "<s>"
    ctx.update("<s>hello")
    assert ctx.text == "<s>hello"
    ctx.update("<s>hello world")
    assert ctx.text == "<s>hello world"


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


def test_character_ngram_repetition_triggers_reset():
    ctx = StreamingContext(
        initial_text="seed",
        repetition_n_gram_size=3,
        repetition_frequency=5,
    )
    # 10 copies of "abc" -> "abc" n-gram count is 8, exceeds threshold=5.
    did_reset = ctx.update("abc" * 10)
    assert did_reset is True
    assert ctx.text == "seed"


def test_sequential_repetition_triggers_reset():
    ctx = StreamingContext(initial_text="<s>")
    # "foofoofoofoofoo" = "foo" * 5, pattern_length=3, repeat_count=5,
    # total segment length = 15 >= min_segment_length(8).
    did_reset = ctx.update("foofoofoofoofoo")
    assert did_reset is True


def test_bracket_pipe_pattern_triggers_reset():
    ctx = StreamingContext(initial_text="<s>", repetition_frequency=1000)
    # The bracket-pipe heuristic looks for ``[tag]`` items followed
    # immediately by ``|`` or end-of-line, with >= 4 brackets and
    # >= 1 pipe per matching line. Two lines of that pattern out of
    # three (67% >= 50% threshold) trips the reset.
    text = "\n".join(
        [
            "[a]|[b]|[c]|[d]",
            "[e]|[f]|[g]|[h]",
            "normal line",
        ]
    )
    did_reset = ctx.update(text)
    assert did_reset is True
    assert ctx.text == "<s>"


def test_all_whitespace_triggers_reset():
    ctx = StreamingContext(initial_text="<s>")
    did_reset = ctx.update("     \n\n\t  ")
    assert did_reset is True
    assert ctx.text == "<s>"


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


# --- ContextStreams: the anchored cohort ------------------------------------


def _cohort(reseed_threshold=2, n=3):
    """Three single-char-anchored paths with a deterministic anchor mint."""
    from praxis.generation.context_blocks import ContextBlock, ContextStreams

    seeds = iter(["A", "B", "C", "D"])
    temps = [1.0 / 3.0, 0.5, 1.0][:n]
    blocks = [ContextBlock(f"b{i}", "", temps[i], 1.0) for i in range(n)]
    return ContextStreams(
        lambda b: StreamingContext(unchanged_threshold=3),
        blocks=blocks,
        seed_factory=lambda: next(seeds),
        reseed_threshold=reseed_threshold,
    )


def test_cohort_shares_one_anchor():
    streams = _cohort()
    assert streams.anchor == "A"
    assert [c.text for c in streams.contexts] == ["A", "A", "A"]
    assert streams.quorum == 2


def test_single_path_degeneracy_keeps_anchor():
    streams = _cohort(reseed_threshold=2)
    # Only the first path degenerates (whitespace); the others grow cleanly.
    for _ in range(5):
        streams.step(lambda t, temp: "   " if temp < 0.5 else t + "x")
    assert streams.anchor == "A"  # below quorum, seed stands
    assert streams.contexts[0].text == "A"  # degenerate path snaps back to anchor


def test_quorum_degeneracy_reanchors_all_paths():
    streams = _cohort(reseed_threshold=2)
    # Two of three paths degenerate each step; once both clear the threshold the
    # shared anchor re-rolls and every path snaps to the new seed.
    for _ in range(2):
        streams.step(lambda t, temp: "   " if temp <= 0.5 else t + "x")
    assert streams.anchor == "B"
    assert [c.text for c in streams.contexts] == ["B", "B", "B"]
