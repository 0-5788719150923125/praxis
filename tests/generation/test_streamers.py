"""Publishing a reply while it is still being written.

The contract that matters is agreement: whatever the streamer published, joined
together, has to equal what a caller who simply waited would have received.
That holds because both go through the same ``extract_assistant_reply`` - there
is no streaming-flavored second copy of the extraction logic. What streaming
adds is the hold-back, and most of what is worth testing is that it never lets
half a boundary out.
"""

import pytest
import torch

from praxis.generation.reply import extract_assistant_reply
from praxis.generation.request import GenerationResult
from praxis.generation.streamers import ReplyStreamer
from tests.generation.scripted import Sink


def _feed(streamer, tokenizer, text):
    """Publish ``text`` one token at a time, the way a decode loop would."""
    for token_id in tokenizer.encode(text):
        streamer.put(torch.tensor([token_id]))


# ---------------------------------------------------------------------------
# agreement with the extractor
# ---------------------------------------------------------------------------


def test_streamed_text_equals_the_waited_for_reply(prose_tokenizer):
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    reply = "Hello there, this is the whole answer."
    _feed(streamer, prose_tokenizer, reply)
    streamer.finish()

    assert len(sink.chunks) > 1, "published as one snapshot, not deltas"
    assert sink.joined == reply
    # ...which is exactly what the caller who waited would have gotten.
    assert sink.joined == extract_assistant_reply(
        GenerationResult(reply, 0), prose_tokenizer
    )


def _tokens_until_first_text(tokenizer, reply):
    sink = Sink()
    streamer = ReplyStreamer(tokenizer, sink.text)
    for i, token_id in enumerate(tokenizer.encode(reply), start=1):
        streamer.put(torch.tensor([token_id]))
        if sink.chunks:
            return i
    return None


@pytest.mark.parametrize("fmt", ["prose", "default"])
def test_the_first_character_ships_immediately(request, fmt):
    """The hold-back is only the tail that could still become a boundary. A
    flat "longest terminator" taken from the first token held 16 bytes back
    before ANY text reached the reader, which reads as the model never having
    started - and every test that calls `finish()` hides it."""
    tokenizer = request.getfixturevalue(f"{fmt}_tokenizer")
    assert _tokens_until_first_text(tokenizer, "The capital of France.") == 1


def test_only_a_real_partial_match_is_held_back(prose_tokenizer):
    """Every terminator starts with a newline or a bracket, so ordinary prose
    withholds nothing - and a tail that could still become a boundary withholds
    exactly itself, no more."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "Hello there")
    assert sink.joined == "Hello there", "ordinary prose was held back"

    # "\n\nus" could still become "\n\nuser\n\n": hold those five, ship nothing more.
    _feed(streamer, prose_tokenizer, "\n\nus")
    assert sink.joined == "Hello there"

    # A character that rules every terminator out releases the lot.
    _feed(streamer, prose_tokenizer, "!")
    assert sink.joined == "Hello there\n\nus!"


def test_the_head_strip_is_not_half_published(default_tokenizer):
    """`extract_assistant_reply` strips a leading `#RESPONSE`, so shipping half
    of it would have to be retracted - and a retraction is what the streamer
    cannot do."""
    sink = Sink()
    streamer = ReplyStreamer(default_tokenizer, sink.text)
    _feed(streamer, default_tokenizer, "#RESP")
    assert sink.joined == ""
    _feed(streamer, default_tokenizer, "ONSE the answer")
    streamer.finish()
    assert sink.joined == "the answer"


def test_half_a_boundary_is_never_published(prose_tokenizer):
    """The hold-back's whole job. Feeding text that ends mid-boundary, the
    partial must stay private - published text cannot be retracted, and one
    more token would turn it into a cut."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "Hi there.\n\nuse")

    assert "\n\nuse" not in sink.joined
    assert sink.joined == "Hi there."[: len(sink.joined)]

    # One more token completes the boundary, and the cut lands where the
    # extractor would have put it.
    _feed(streamer, prose_tokenizer, "r\n\n")
    streamer.finish()
    assert sink.joined == "Hi there."


@pytest.mark.parametrize(
    "after",
    [
        "what the model kept going",
        pytest.param(
            "trailing junk",
            marks=pytest.mark.xfail(
                strict=True,
                reason="ReplyStreamer publishes the '\\n\\nuser' boundary when the "
                "text after it starts like the 'tool' role boundary",
            ),
        ),
    ],
)
def test_every_prefix_of_the_stream_is_a_prefix_of_the_answer(prose_tokenizer, after):
    """Monotonicity: a consumer appends, so anything published early has to
    still be right later. The turn ends at the next speaker's name, and that
    boundary and everything after it is plumbing the client never sees."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    final = "One two three four five."
    seen = []
    for token_id in prose_tokenizer.encode(final + "\n\nuser\n\n" + after):
        streamer.put(torch.tensor([token_id]))
        seen.append(sink.joined)
    streamer.finish()

    assert sink.joined == final
    for partial in seen:
        assert final.startswith(partial), f"published {partial!r}, answer {final!r}"


def test_an_empty_turn_publishes_nothing(prose_tokenizer):
    """The placeholder a finished empty turn shows is a presentation choice for
    a completed turn; streaming it would put literal parenthetical text into
    the middle of a reply."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "\n\nuser\n\n")
    streamer.finish()
    assert sink.joined == ""


def test_control_tokens_do_not_leak_under_the_default_format(default_tokenizer):
    """The default format ends a turn on a control TOKEN rather than a role
    line, and its text (`[EOS]`) must not reach the client either."""
    sink = Sink()
    streamer = ReplyStreamer(default_tokenizer, sink.text)
    _feed(streamer, default_tokenizer, "answer text")
    streamer.put(torch.tensor([default_tokenizer.eos_token_id]))
    streamer.finish()
    assert sink.joined == "answer text"


# ---------------------------------------------------------------------------
# driver hooks
# ---------------------------------------------------------------------------


def test_the_step_prompt_is_not_republished(prose_tokenizer):
    """Every `generate` call publishes the whole sequence so far before it
    starts. Only what follows is new, and `begin_step` is what says so."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)

    streamer.begin_step()
    prompt = torch.tensor([prose_tokenizer.encode("user\n\nhi\n\nassistant\n\n")])
    streamer.put(prompt)
    _feed(streamer, prose_tokenizer, "the reply")
    streamer.finish()

    assert sink.joined == "the reply"


def test_a_muted_streamer_absorbs_nothing(prose_tokenizer):
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "visible ")
    streamer.mute()
    _feed(streamer, prose_tokenizer, "INVISIBLE")
    streamer.mute(False)
    _feed(streamer, prose_tokenizer, "tail")
    streamer.finish()

    assert "INVISIBLE" not in sink.joined
    assert sink.joined == "visible tail"


def test_restart_drops_what_is_no_longer_part_of_the_answer(prose_tokenizer):
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text, on_reset=sink.reset)
    _feed(streamer, prose_tokenizer, "thinking out loud")
    streamer.finish()
    assert sink.joined == "thinking out loud"

    streamer.restart()
    assert sink.resets == 1
    assert sink.joined == ""

    _feed(streamer, prose_tokenizer, "the real answer")
    streamer.finish()
    assert sink.joined == "the real answer"


def test_restart_without_a_reset_callback_is_still_safe(prose_tokenizer):
    """A consumer with nowhere to put a reset can omit it and lean on the final
    result instead; the streamer must not require one."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "abc")
    streamer.restart()
    _feed(streamer, prose_tokenizer, "xyz")
    streamer.finish()
    assert sink.joined.endswith("xyz")


def test_end_is_a_step_boundary_not_the_end_of_the_turn(prose_tokenizer):
    """transformers calls end() at the close of every decode, and a Praxis turn
    is several of them - so end() must not release the hold-back."""
    sink = Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "partial\n\nuse")
    streamer.end()
    assert "\n\nuse" not in sink.joined

    streamer.begin_step()
    streamer.put(torch.tensor([prose_tokenizer.encode("partial\n\nuse")]))
    _feed(streamer, prose_tokenizer, "ful text")
    streamer.finish()
    assert sink.joined == "partial\n\nuseful text"
