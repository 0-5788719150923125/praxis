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


# ---------------------------------------------- display vs prompt line breaks
#
# The CLI dashboard wraps with str.splitlines(); the browser's
# `white-space: pre-wrap` only breaks on LF / CR / CRLF. Python's set is
# strictly larger, so a byte-level model emitting \v, \f, U+0085 or U+2028
# produced a line break in the terminal that silently vanished in the web
# Terminal tab. The DISPLAY copy is normalized so both agree; the PROMPT copy
# must not be, or the model conditions on bytes it never produced.

EXOTIC_BREAKS = ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]


@pytest.mark.parametrize("sep", EXOTIC_BREAKS)
def test_display_text_normalizes_breaks_the_browser_ignores(sep):
    ctx = StreamingContext(initial_text="a")
    ctx.update(f"one{sep}two")

    # The prompt copy is untouched - byte-exact is the contract.
    assert ctx.text == f"one{sep}two"

    # The display copy breaks where the CLI already did.
    assert ctx.display_text == "one\ntwo"


@pytest.mark.parametrize("sep", EXOTIC_BREAKS)
def test_both_renderers_agree_on_line_count(sep):
    """The actual invariant: same number of lines in the terminal and the web.

    The CLI counts with splitlines(); the browser counts LF (having already
    collapsed CRLF). Before normalizing, these disagreed by one per separator.
    """
    ctx = StreamingContext(initial_text="a")
    ctx.update(f"alpha{sep}beta{sep}gamma")

    cli_lines = ctx.display_text.splitlines()
    browser_lines = ctx.display_text.replace("\r\n", "\n").split("\n")
    assert cli_lines == browser_lines == ["alpha", "beta", "gamma"]

    # And the raw buffer is where they diverged.
    assert len(ctx.text.splitlines()) != len(ctx.text.split("\n"))


def test_display_text_leaves_ordinary_whitespace_alone():
    ctx = StreamingContext(initial_text="a")
    ctx.update("keep\nthese\r\nand\ttabs  and spaces")
    assert ctx.display_text == "keep\nthese\r\nand\ttabs  and spaces"


def test_context_payload_ships_the_display_copy():
    """The web reads ContextStreams.payload(); that is the copy that must be
    normalized, while token counts stay measured against the raw buffer."""
    from praxis.generation.context_blocks import ContextBlock, ContextStreams

    blocks = [
        ContextBlock(name="Primary", description="test", temperature=0.5, chance=1.0)
    ]
    streams = ContextStreams(
        context_factory=lambda b: StreamingContext(initial_text="a"),
        blocks=blocks,
        token_counter=len,
    )
    streams.contexts[0].text = "one two"

    entry = streams.payload()[0]
    assert entry["text"] == "one\ntwo"
    # len() stands in for the tokenizer here: counted on the raw buffer, whose
    # U+2028 is several bytes to a byte-level tokenizer where \n is one.
    assert entry["tokens"] == len("one two")


# ── the prompt the buffer feeds back ─────────────────────────────────────
#
# The buffer is text, so every round re-encodes it. On a byte-level tokenizer
# a token is a byte, and the generator truncates prompts by token index - so an
# unaligned cut severs a multi-byte character, decode yields U+FFFD, and the
# next encode turns that one character into three real bytes. Those bytes make
# the buffer longer in bytes than in characters, which guarantees the next
# round truncates too. The context ends up minting replacement characters
# indefinitely and prompting the model with a sequence its training data
# essentially never contains.


def _prose_byte_tokenizer():
    """The run's tokenizer: byte-level ids under the text-boundary format."""
    from praxis.tokenizers.byte_level import ByteLevelTokenizer
    from praxis.tokenizers.chat_templates import PROSE_FORMAT, apply_chat_format

    tok = ByteLevelTokenizer(chat_format=PROSE_FORMAT)
    apply_chat_format(tok, PROSE_FORMAT)
    return tok


def _generator_with(tokenizer, max_positions=4096):
    """A Generator wired to a stub backend; only _prepare_inputs is exercised."""
    from types import SimpleNamespace

    from praxis.generation.generator import Generator

    backend = SimpleNamespace(
        model=None,
        device="cpu",
        max_positions=max_positions,
        tokenizer=tokenizer,
        default_sampling_temperature=None,
    )
    return Generator(tokenizer=tokenizer, device="cpu", backend=backend)


def test_prompt_truncation_lands_on_a_character_boundary():
    from praxis.generation.request import GenerationRequest

    tok = _prose_byte_tokenizer()
    gen = _generator_with(tok)
    text = "日本語のテキストです"

    for budget in range(4, len(tok.encode(text))):
        request = GenerationRequest(
            id="t",
            prompt=text,
            kwargs=dict(
                max_new_tokens=4, truncate_to=budget, skip_special_tokens=False
            ),
        )
        input_ids, _, _, _ = gen._prepare_inputs(request)
        decoded = tok.decode(input_ids[0].tolist(), skip_special_tokens=False)
        assert "�" not in decoded, f"severed character at truncate_to={budget}"


def test_prompt_truncation_still_respects_the_budget():
    """Alignment moves the cut forward, never backward - the prompt must not
    grow past what the positional budget allows."""
    from praxis.generation.request import GenerationRequest

    tok = _prose_byte_tokenizer()
    gen = _generator_with(tok)
    for budget in range(4, 30):
        request = GenerationRequest(
            id="t",
            prompt="日本語のテキストです",
            kwargs=dict(
                max_new_tokens=4, truncate_to=budget, skip_special_tokens=False
            ),
        )
        input_ids, _, _, _ = gen._prepare_inputs(request)
        assert input_ids.size(1) <= budget


def test_truncation_is_unchanged_for_tokenizers_without_the_hook():
    """One token = whole characters there, so there is nothing to align and
    the cut must stay exactly where the budget puts it."""
    from types import SimpleNamespace

    from praxis.generation.request import GenerationRequest

    from praxis.tokenizers.chat_templates import PROSE_FORMAT

    class _Whole:
        chat_format = PROSE_FORMAT
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    tok = _Whole()
    assert not hasattr(tok, "align_left_cut")
    gen = _generator_with(tok)
    request = GenerationRequest(
        id="t",
        prompt="abcdefghij",
        kwargs=dict(max_new_tokens=1, truncate_to=4, skip_special_tokens=False),
    )
    input_ids, _, _, _ = gen._prepare_inputs(request)
    assert input_ids[0].tolist() == [ord(c) for c in "ghij"]
