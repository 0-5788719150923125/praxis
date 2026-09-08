"""Publishing a reply while it is still being written.

The contract that matters is agreement: whatever the streamer published, joined
together, has to equal what a caller who simply waited would have received.
That holds because both go through the same ``extract_assistant_reply`` - there
is no streaming-flavored second copy of the extraction logic. What streaming
adds is the hold-back, and most of what is worth testing is that it never lets
half a boundary out.
"""

import contextlib
import time

import pytest
import torch

from praxis.generation.generator import Generator
from praxis.generation.reply import extract_assistant_reply
from praxis.generation.request import GenerationResult
from praxis.generation.streamers import ReplyStreamer
from praxis.tokenizers import create_tokenizer


@pytest.fixture(scope="module")
def prose_tokenizer():
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )


@pytest.fixture(scope="module")
def default_tokenizer():
    return create_tokenizer(tokenizer_type="byte_level", vocab_size=1024)


class _Sink:
    def __init__(self):
        self.chunks = []
        self.resets = 0

    def text(self, delta):
        self.chunks.append(delta)

    def reset(self):
        self.resets += 1
        self.chunks.clear()

    @property
    def joined(self):
        return "".join(self.chunks)


def _feed(streamer, tokenizer, text):
    """Publish ``text`` one token at a time, the way a decode loop would."""
    for token_id in tokenizer.encode(text):
        streamer.put(torch.tensor([token_id]))


# ---------------------------------------------------------------------------
# agreement with the extractor
# ---------------------------------------------------------------------------


def test_streamed_text_equals_the_waited_for_reply(prose_tokenizer):
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    reply = "Hello there, this is the whole answer."
    _feed(streamer, prose_tokenizer, reply)
    streamer.finish()

    assert sink.joined == reply
    # ...which is exactly what the caller who waited would have gotten.
    assert sink.joined == extract_assistant_reply(
        GenerationResult(reply, 0), prose_tokenizer
    )


def test_the_turn_boundary_is_never_published(prose_tokenizer):
    """A prose turn ends at the next speaker's name. The reply stops there and
    the boundary itself is plumbing the client must not see."""
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "Hi there.\n\nuser\n\nwhat the model kept going")
    streamer.finish()

    assert sink.joined == "Hi there."


def _tokens_until_first_text(tokenizer, reply):
    sink = _Sink()
    streamer = ReplyStreamer(tokenizer, sink.text)
    for i, token_id in enumerate(tokenizer.encode(reply), start=1):
        streamer.put(torch.tensor([token_id]))
        if sink.chunks:
            return i
    return None


def test_the_first_character_ships_immediately(prose_tokenizer):
    """The latency regression, and it was invisible in every other test here
    because they all call `finish()`.

    The hold-back used to be a flat "longest terminator" - 16 characters under
    both shipped formats - taken from the very first token. On a byte-level
    model decoding inside the training loop that is 16 bytes before ANY text
    reaches the reader, however fast the model runs, and it reads as the model
    never having started.
    """
    assert _tokens_until_first_text(prose_tokenizer, "The capital of France.") == 1


def test_the_first_character_ships_immediately_under_token_boundaries(
    default_tokenizer,
):
    assert _tokens_until_first_text(default_tokenizer, "The capital of France.") == 1


def test_only_a_real_partial_match_is_held_back(prose_tokenizer):
    """Every terminator starts with a newline or a bracket, so ordinary prose
    withholds nothing - and a tail that could still become a boundary withholds
    exactly itself, no more."""
    sink = _Sink()
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
    sink = _Sink()
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
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "Hi there.\n\nuse")

    assert "\n\nuse" not in sink.joined
    assert sink.joined == "Hi there."[: len(sink.joined)]

    # One more token completes the boundary, and the cut lands where the
    # extractor would have put it.
    _feed(streamer, prose_tokenizer, "r\n\n")
    streamer.finish()
    assert sink.joined == "Hi there."


def test_every_prefix_of_the_stream_is_a_prefix_of_the_answer(prose_tokenizer):
    """Monotonicity: a consumer appends, so anything published early has to
    still be right later."""
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    final = "One two three four five."
    seen = []
    for token_id in prose_tokenizer.encode(final + "\n\nuser\n\n"):
        streamer.put(torch.tensor([token_id]))
        seen.append(sink.joined)
    streamer.finish()

    assert sink.joined == final
    for partial in seen:
        assert final.startswith(partial), f"published {partial!r}, answer {final!r}"


def test_the_stream_is_deltas_not_snapshots(prose_tokenizer):
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "abcdefghijklmnopqrstuvwxyz")
    streamer.finish()
    assert len(sink.chunks) > 1
    assert sink.joined == "abcdefghijklmnopqrstuvwxyz"


def test_an_empty_turn_publishes_nothing(prose_tokenizer):
    """The placeholder a finished empty turn shows is a presentation choice for
    a completed turn; streaming it would put literal parenthetical text into
    the middle of a reply."""
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "\n\nuser\n\n")
    streamer.finish()
    assert sink.joined == ""


def test_control_tokens_do_not_leak_under_the_default_format(default_tokenizer):
    """The default format ends a turn on a control TOKEN rather than a role
    line, and its text (`[EOS]`) must not reach the client either."""
    sink = _Sink()
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
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)

    streamer.begin_step()
    prompt = torch.tensor([prose_tokenizer.encode("user\n\nhi\n\nassistant\n\n")])
    streamer.put(prompt)
    _feed(streamer, prose_tokenizer, "the reply")
    streamer.finish()

    assert sink.joined == "the reply"


def test_a_muted_streamer_absorbs_nothing(prose_tokenizer):
    sink = _Sink()
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
    sink = _Sink()
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
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "abc")
    streamer.restart()
    _feed(streamer, prose_tokenizer, "xyz")
    streamer.finish()
    assert sink.joined.endswith("xyz")


def test_end_is_a_step_boundary_not_the_end_of_the_turn(prose_tokenizer):
    """transformers calls end() at the close of every decode, and a Praxis turn
    is several of them - so end() must not release the hold-back."""
    sink = _Sink()
    streamer = ReplyStreamer(prose_tokenizer, sink.text)
    _feed(streamer, prose_tokenizer, "partial\n\nuse")
    streamer.end()
    assert "\n\nuse" not in sink.joined

    streamer.begin_step()
    streamer.put(torch.tensor([prose_tokenizer.encode("partial\n\nuse")]))
    _feed(streamer, prose_tokenizer, "ful text")
    streamer.finish()
    assert sink.joined == "partial\n\nuseful text"


# ---------------------------------------------------------------------------
# end to end, through the Generator
# ---------------------------------------------------------------------------


class _ScriptBackend:
    """Emits a fixed string one byte per step, the way a real decode would."""

    model = None
    default_sampling_temperature = None
    max_positions = None

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.script = tokenizer.encode(script)
        self.device = "cpu"

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        # Mirror what transformers does: publish the step's prompt first, then
        # each produced token.
        if streamer is not None:
            streamer.put(tokens)
        produced = tokens.shape[1] - self.prompt_len
        budget = int(step_kwargs.get("max_new_tokens", 100))
        for i in range(produced, min(len(self.script), produced + budget)):
            nxt = torch.tensor([[self.script[i]]], dtype=torch.long)
            tokens = torch.cat([tokens, nxt], dim=-1)
            if streamer is not None:
                streamer.put(nxt[0])
        if streamer is not None:
            streamer.end()
        return tokens


def test_generator_publishes_deltas_that_match_its_own_result(prose_tokenizer):
    """The end-to-end promise: streaming is additive. The deltas joined equal
    the reply the caller gets back, and a caller that passes no callback is
    completely unaffected."""
    prompt = "user\n\nhi\n\nassistant\n\n"
    backend = _ScriptBackend(prose_tokenizer, "Hello!\n\nuser\n\nnot mine")
    backend.prompt_len = len(prose_tokenizer.encode(prompt))

    gen = Generator(backend=backend, tokenizer=prose_tokenizer)
    gen.tools = {}

    sink = _Sink()
    rid = gen.request_generation(prompt, {"max_new_tokens": 64}, on_text=sink.text)
    gen.fulfill_requests(max_requests=1)
    result = gen.get_result(rid)

    assert sink.joined == extract_assistant_reply(result, prose_tokenizer)
    assert sink.joined == "Hello!"


def test_a_request_without_a_callback_streams_nothing(prose_tokenizer):
    prompt = "user\n\nhi\n\nassistant\n\n"
    backend = _ScriptBackend(prose_tokenizer, "Hello!")
    backend.prompt_len = len(prose_tokenizer.encode(prompt))

    gen = Generator(backend=backend, tokenizer=prose_tokenizer)
    gen.tools = {}
    rid = gen.request_generation(prompt, {"max_new_tokens": 64})
    gen.fulfill_requests(max_requests=1)
    assert extract_assistant_reply(gen.get_result(rid), prose_tokenizer) == "Hello!"


# ---------------------------------------------------------------------------
# the tool flow, where "what the model said" changes mid-turn
# ---------------------------------------------------------------------------


class _PendingBackend:
    """Emits a fixed script one token per step, halting on the real criteria.

    Modelled on the scripted backend in ``test_chat_formats``; it exists here so
    the tool state machine can be driven with a streamer attached.
    """

    model = None
    default_sampling_temperature = None
    max_positions = None

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.pending = list(tokenizer.encode(script, add_special_tokens=False))

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def _criteria(self, step_kwargs):
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
        )

        criteria = StoppingCriteriaList()
        stops = step_kwargs.get("stop_strings")
        if stops:
            criteria.append(
                StopStringCriteria(stop_strings=list(stops), tokenizer=self.tokenizer)
            )
        eos = step_kwargs.get("eos_token_id")
        if eos:
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            criteria.append(EosTokenCriteria(eos_token_id=torch.tensor(eos)))
        return criteria

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        from praxis.generation.decoding import first_halt

        criteria = self._criteria(step_kwargs)
        budget = int(step_kwargs.get("max_new_tokens", 100))
        start = tokens.shape[1]
        if streamer is not None:
            streamer.put(tokens)
        ids = tokens[0].tolist()
        produced = 0
        while self.pending and produced < budget:
            nxt = self.pending.pop(0)
            ids.append(nxt)
            produced += 1
            if streamer is not None:
                streamer.put(torch.tensor([nxt]))
            if first_halt(torch.tensor([ids]), criteria, start) is not None:
                break
        if streamer is not None:
            streamer.end()
        return torch.tensor([ids], dtype=torch.long)


def test_a_tool_call_never_reaches_the_consumer(default_tokenizer):
    """The JSON body is not the reply, and neither is anything written before
    the call: the runtime's turn anchor moves past the spliced result, so the
    final answer is only what follows it. The consumer is told to drop the
    rest rather than being left holding it.
    """
    script = (
        "Checking now.\n[TOOL_CALL]\n"
        '{"name": "get_time", "arguments": {}}\n[/TOOL_CALL]\n'
        "It is noon.[SEP]"
    )
    gen = Generator(
        backend=_PendingBackend(default_tokenizer, script),
        tokenizer=default_tokenizer,
        synchronous=True,
    )
    gen.tools = {"get_time": {}}
    gen.call_tool = lambda name, args: "noon"

    sink = _Sink()
    rid = gen.request_generation(
        "[BOS]user\nwhat time is it?[SEP][BOS]assistant\n",
        {"max_new_tokens": 200},
        on_text=sink.text,
        on_reset=sink.reset,
    )
    result = gen.get_result(rid)

    assert extract_assistant_reply(result, default_tokenizer) == "It is noon."
    # The pre-call chatter was published, then invalidated when the runtime
    # spliced the result and moved the turn anchor past it.
    assert sink.resets == 1
    assert sink.joined == "It is noon."
    # The JSON body never went out at all - the streamer was muted for it.
    assert "TOOL_CALL" not in sink.joined
    assert "get_time" not in sink.joined


def test_the_consumer_is_told_when_a_tool_call_invalidates_what_it_has():
    """Without the reset signal a consumer that appended the pre-call chatter
    would show it beside an answer that no longer includes it."""
    tokenizer = create_tokenizer(tokenizer_type="byte_level", vocab_size=1024)
    script = (
        "Checking now.\n[TOOL_CALL]\n"
        '{"name": "get_time", "arguments": {}}\n[/TOOL_CALL]\n'
        "It is noon.[SEP]"
    )
    gen = Generator(
        backend=_PendingBackend(tokenizer, script),
        tokenizer=tokenizer,
        synchronous=True,
    )
    gen.tools = {"get_time": {}}
    gen.call_tool = lambda name, args: "noon"

    seen = []
    sink = _Sink()

    def record(delta):
        seen.append(delta)
        sink.text(delta)

    rid = gen.request_generation(
        "[BOS]user\nwhat time is it?[SEP][BOS]assistant\n",
        {"max_new_tokens": 200},
        on_text=record,
    )
    gen.get_result(rid)

    assert sink.resets == 0  # this sink registered no reset handler
    assert "".join(seen).endswith("It is noon.")
