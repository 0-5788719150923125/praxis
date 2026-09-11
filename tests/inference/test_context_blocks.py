"""``ContextStreams``: a cohort of rolling contexts sharing one anchor."""

from __future__ import annotations

from praxis.inference import StreamingContext
from praxis.inference.context_blocks import ContextBlock, ContextStreams


def _cohort(reseed_threshold=2, n=3):
    """Three single-char-anchored paths with a deterministic anchor mint."""
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


def test_context_payload_ships_the_display_copy():
    """The web reads ContextStreams.payload(); that is the copy that must be
    normalized, while token counts stay measured against the raw buffer."""
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
