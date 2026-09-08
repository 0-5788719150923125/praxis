"""Telling the client that a tool ran.

The reply is the wrong place to look. Under either chat format the extractor
strips the whole call/result exchange out of it (see
``praxis.generation.reply``), and under ``tool_style="roles"`` the streamer is
muted for the duration of the call on top of that - so a turn that consulted a
tool and a turn that made the same claim up produce byte-identical text. The
only account of the difference is ``on_tool``, fired from the branch that
executes the tool, and these are its properties:

- it names the tool that actually RAN, resolved the same way the executor
  resolves it,
- a call that never ran is never announced,
- and the announcement survives the reset that follows it, because the reset
  retracts the model's pre-call chatter and not the fact of the call.
"""

import contextlib
import json

import pytest
import torch

from praxis.generation.generator import Generator
from praxis.tokenizers import create_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format="prose"
    )


class _ScriptedBackend:
    """Appends one canned chunk of text per ``generate_until_halt`` call.

    A real decode reaches these boundaries by sampling; scripting them isolates
    the reporting from whether a model can be coaxed into a well-formed call.
    The chunks are the exact text ``prose`` puts on the wire, so the generator's
    own halt classification is exercised rather than stubbed.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, tokenizer, chunks):
        self.tokenizer = tokenizer
        self.chunks = list(chunks)
        self.device = "cpu"
        self.max_positions = None

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        if not self.chunks:
            return tokens
        chunk = self.chunks.pop(0)
        ids = self.tokenizer.encode(chunk, add_special_tokens=False)
        nxt = torch.tensor([ids], dtype=torch.long, device=tokens.device)
        extended = torch.cat([tokens, nxt], dim=-1)
        if streamer is not None:
            streamer.put(nxt)
        return extended


PROMPT = "user\n\nhi\n\nassistant\n\n"


def _call(name, arguments):
    """One call/answer exchange, as the two chunks a decode halts between."""
    return [
        "let me check\n\ncall\n\n",
        json.dumps({"name": name, "arguments": arguments}) + "\n\ntool\n\n",
    ]


def _run(tokenizer, chunks, **callbacks):
    gen = Generator(backend=_ScriptedBackend(tokenizer, chunks), tokenizer=tokenizer)
    rid = gen.request_generation(PROMPT, {"max_new_tokens": 4000}, **callbacks)
    gen.fulfill_requests()
    return gen.get_result(rid)


def test_a_tool_that_runs_is_announced_by_name(tokenizer):
    seen = []
    result = _run(
        tokenizer,
        _call("calc", {"values": [2, 3], "op": "add"}) + ["five\n\nuser\n\n"],
        on_tool=seen.append,
    )

    assert seen == ["calc"]
    # ...and the result really did come back through the tool path.
    assert "5" in result


def test_each_execution_is_announced_separately(tokenizer):
    """The chips count executions, so the callback has to fire per call rather
    than once per distinct tool. The arguments differ because an identical
    repeat is stopped as a duplicate - which is the next test."""
    seen = []
    _run(
        tokenizer,
        _call("calc", {"values": [2, 3], "op": "add"})
        + _call("calc", {"values": [4, 5], "op": "mul"})
        + ["twenty\n\nuser\n\n"],
        on_tool=seen.append,
    )

    assert seen == ["calc", "calc"]


def test_a_duplicate_call_is_not_announced(tokenizer):
    """`execute_tool_call` stops the loop on a repeat rather than running the
    tool again, so counting it would report work that never happened."""
    seen = []
    _run(
        tokenizer,
        _call("calc", {"values": [2, 3], "op": "add"})
        + _call("calc", {"values": [2, 3], "op": "add"})
        + ["five\n\nuser\n\n"],
        on_tool=seen.append,
    )

    assert seen == ["calc"]


def test_a_nameless_call_is_not_announced(tokenizer):
    """A malformed body still gets an error result spliced back for the model
    to recover from, but there is no tool to name - and captioning the turn
    with a tool that never ran would be worse than saying nothing."""
    seen = []
    _run(
        tokenizer,
        ["\n\ncall\n\n", "not json at all\n\ntool\n\n", "sorry\n\nuser\n\n"],
        on_tool=seen.append,
    )

    assert seen == []


def test_the_announcement_precedes_the_reset_it_causes(tokenizer):
    """Ordering the client depends on. The splice moves the turn anchor, so
    every call is followed by "drop what you have"; a chip that arrived after
    that would look like something the reset should have taken back."""
    events = []
    _run(
        tokenizer,
        _call("calc", {"values": [2, 3], "op": "add"}) + ["five\n\nuser\n\n"],
        on_text=lambda text: events.append(("text", text)),
        on_reset=lambda: events.append(("reset", None)),
        on_tool=lambda name: events.append(("tool", name)),
    )

    kinds = [kind for kind, _ in events]
    assert "tool" in kinds and "reset" in kinds
    assert kinds.index("tool") < kinds.index("reset")


def test_a_raising_callback_never_reaches_the_training_loop(tokenizer):
    """This fires from inside ``on_train_batch_end`` for a queued request. A
    chat-UI affordance must not be able to take the run down."""

    def explode(name):
        raise RuntimeError("the browser went away")

    result = _run(
        tokenizer,
        _call("calc", {"values": [2, 3], "op": "add"}) + ["five\n\nuser\n\n"],
        on_tool=explode,
    )

    assert "5" in result


def test_no_callback_is_the_unchanged_path(tokenizer):
    """A caller that does not ask is unaffected - the same bargain the deltas
    already make."""
    result = _run(
        tokenizer, _call("calc", {"values": [2, 3], "op": "add"}) + ["five\n\nuser\n\n"]
    )

    assert "5" in result


def test_the_reported_name_is_the_one_the_executor_resolved(tokenizer):
    """Three spellings reach the executor (``name``, ``tool``, and OpenAI's
    nested ``function.name``). The badge shares its resolver, so it cannot name
    a different tool than the one that ran."""
    from praxis.tools import tool_call_name

    for call in (
        {"name": "calc"},
        {"tool": "calc"},
        {"function": {"name": "calc"}},
    ):
        assert tool_call_name(call) == "calc"
    assert tool_call_name({"arguments": {}}) is None
    assert tool_call_name("not a dict") is None
