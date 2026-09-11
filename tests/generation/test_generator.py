from __future__ import annotations

import contextlib
import json
import time
from unittest.mock import MagicMock

import pytest
import torch

from praxis import PraxisConfig
from praxis.generation.decoding import first_halt
from praxis.generation.generator import Generator
from praxis.generation.reply import extract_assistant_reply
from praxis.generation.request import GenerationRequest
from praxis.modeling import PraxisForCausalLM
from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.chat_templates import apply_chat_format, chat_format_of
from praxis.tools import calc, call_tool, find_unprocessed_tool_call_ids

# ------------------------------------------------------------------------------
# tool_reporting
# ------------------------------------------------------------------------------
# Telling the client that a tool ran.
#
# The reply is the wrong place to look. Under either chat format the extractor strips
# the whole call/result exchange out of it (see ``praxis.generation.reply``), and under
# ``tool_style="roles"`` the streamer is muted for the duration of the call on top of
# that - so a turn that consulted a tool and a turn that made the same claim up produce
# byte-identical text. The only account of the difference is ``on_tool``, fired from the
# branch that executes the tool, and these are its properties:
#
# - it names the tool that actually RAN, resolved the same way the executor resolves it,
# - a call that never ran is never announced, - and the announcement survives the reset
# that follows it, because the reset retracts the model's pre-call chatter and not the
# fact of the call.


class _ChunkBackend:
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
    gen = Generator(backend=_ChunkBackend(tokenizer, chunks), tokenizer=tokenizer)
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


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


# ------------------------------------------------- reply anchoring end-to-end


class _ScriptedBackend:
    """Emits a fixed byte script, halting exactly as the real decode loops do.

    Drives the real ``Generator``, so these exercise the whole endpoint path -
    prompt construction, the halt contract, the tool state machine, and reply
    extraction - with the model's output pinned instead of sampled.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, tokenizer, script):
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.max_positions = None
        self.pending = list(tokenizer.encode(script, add_special_tokens=False))

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def _criteria(self, step_kwargs):
        """The criteria transformers would build for these kwargs.

        Assembled here rather than approximated, so the scripted backend halts
        on exactly what the real one halts on.
        """
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


def _scripted_reply(
    tokenizer, script, max_new_tokens=200, tools=None, call_tool=None, messages=None
):
    from praxis.generation.generator import Generator
    from praxis.web.utils.formatters import generate_from_messages

    generator = Generator(
        backend=_ScriptedBackend(tokenizer, script),
        tokenizer=tokenizer,
        synchronous=True,
    )
    generator.tools = {} if tools is None else tools
    if call_tool is not None:
        generator.call_tool = call_tool
    return generate_from_messages(
        messages or [{"role": "user", "content": "What is 2+2?"}],
        generator,
        tokenizer,
        max_new_tokens=max_new_tokens,
        timeout=10.0,
    )


@pytest.mark.parametrize(
    "script,expected",
    [
        ("It is 4.\n\nuser\n\n", "It is 4."),
        # A second assistant turn ends the first one. It used to ANCHOR the
        # reply, so everything the model actually said was discarded.
        (
            "Sure, here goes.\n\nassistant\n\nSecond thought.\n\nuser\n\n",
            "Sure, here goes.",
        ),
        # ...and when the second turn ran out of budget, the whole reply was
        # reported as an empty turn.
        ("The answer is 4.\n\nassistant\n\n", "The answer is 4."),
    ],
)
def test_prose_reply_is_anchored_where_the_runtime_wrote(
    prose_tokenizer, script, expected
):
    assert _scripted_reply(prose_tokenizer, script) == expected


def test_default_reply_is_anchored_where_the_runtime_wrote(default_tokenizer):
    """Same defect, same fix, under control-token boundaries."""
    reply = _scripted_reply(
        default_tokenizer, "first part[BOS]assistant\nsecond part[SEP]"
    )
    assert reply == "first part"


@pytest.mark.parametrize(
    "script,expected",
    [
        # The model repeating the boundary the prompt just wrote is noise. The
        # cut treats a reply-role boundary as end-of-turn, so without skipping
        # the repetition first it zeroed the reply that followed.
        ("assistant\n\nHere is the answer.\n\nuser\n\n", "Here is the answer."),
        ("\n\nassistant\n\nHere is the answer.\n\nuser\n\n", "Here is the answer."),
    ],
)
def test_prose_seam_repetition_does_not_eat_the_reply(
    prose_tokenizer, script, expected
):
    assert _scripted_reply(prose_tokenizer, script) == expected


def test_default_turn_opener_repetition_does_not_eat_the_reply(default_tokenizer):
    """`[BOS]` is samplable under `default`, so the model can restate the opener.

    Anchoring on the runtime offset put that BOS at offset 0 of the slice, where
    the end-of-turn cut read it as an immediately-empty turn.
    """
    reply = _scripted_reply(default_tokenizer, "[BOS]assistant\nIt is 4.[SEP]")
    assert reply == "It is 4."


def test_default_other_role_opener_still_ends_the_turn(default_tokenizer):
    """Only the REPLY role's opener is noise; `[BOS]user` genuinely ends it."""
    reply = _scripted_reply(default_tokenizer, "It is 4.[BOS]user\nnext question[SEP]")
    assert reply == "It is 4."


def test_stale_prompt_call_does_not_drag_the_anchor_backwards(default_tokenizer):
    """A splice landing MID-sequence must not move the anchor past the reply.

    `find_unprocessed_tool_call_ids` scans from the front, so an unanswered
    `[TOOL_CALL]` sitting in the PROMPT gets its result spliced in the middle.
    Anchoring at the end of that splice threw away everything the model had
    already written.
    """
    messages = [
        {"role": "user", "content": "time?"},
        {
            "role": "assistant",
            "content": '[TOOL_CALL]\n{"name": "get_time", "arguments": {}}\n[/TOOL_CALL]',
        },
        {"role": "user", "content": "and now?"},
    ]
    script = (
        "The time is definitely noon.\n[TOOL_CALL]\n"
        '{"name": "get_time", "arguments": {}}\n[/TOOL_CALL]'
    )
    reply = _scripted_reply(
        default_tokenizer,
        script,
        tools={"get_time": {}},
        call_tool=lambda name, args: "noon",
        messages=messages,
    )
    assert "The time is definitely noon." in reply


def test_generation_stays_inside_the_positional_capacity(prose_tokenizer):
    """Budgeting on model output alone dropped the total-length bound.

    `_prepare_inputs` caps the prompt at `mpe - max_new_tokens` precisely so the
    context can never overflow learned positions. A tool result spliced in is
    not model output, so it does not spend the budget - but it does spend the
    context, and the loop has to notice.
    """
    from praxis.generation.generator import Generator

    script = (
        '\n\ncall\n\n{"name": "get_time", "arguments": {}}\n\ntool\n\n'
        "and here is a long tail.\n\nuser\n\n"
    )
    backend = _ScriptedBackend(prose_tokenizer, script)
    backend.max_positions = 256
    generator = Generator(backend=backend, tokenizer=prose_tokenizer)
    generator.tools = {"get_time": {}}
    generator.call_tool = lambda name, args: "R" * 900
    request = GenerationRequest(
        id="t",
        prompt="user\n\nhi\n\nassistant\n\n",
        kwargs={"max_new_tokens": 128},
    )
    out = generator._process_single_request(request)
    assert len(prose_tokenizer.encode(str(out), add_special_tokens=False)) <= 256


def test_spliced_tool_result_does_not_spend_the_caller_budget(prose_tokenizer):
    """The model did not write the tool result, so it must not be charged for it.

    `remaining` used to be derived from the sequence length, which the splice
    grows: one fat result against the web default of 256 drove it negative and
    broke the loop before the model ever spoke.
    """
    result = "x" * 800
    script = (
        '\n\ncall\n\n{"name": "get_time", "arguments": {}}\n\ntool\n\n'
        + "I looked it up."
        + "\n\nuser\n\n"
    )
    reply = _scripted_reply(
        prose_tokenizer,
        script,
        max_new_tokens=64,
        tools={"get_time": {}},
        call_tool=lambda name, args: result,
    )
    assert reply.startswith("I looked")


def test_generator_passes_suppression_to_the_sampler(default_tokenizer):
    """A declared suppression that never reaches generate() is decoration.

    Exercised on `default`, the layout that still HAS unproducible ids; prose
    has none left to suppress.
    """
    suppress = chat_format_of(default_tokenizer).suppressed_token_ids(default_tokenizer)
    assert suppress  # guards against the assertions below passing vacuously
    assert default_tokenizer.eos_token_id not in suppress
    ids = default_tokenizer.encode("user\nhi\n")
    assert all(0 <= t < default_tokenizer.byte_alphabet_size for t in suppress)
    assert all(t not in suppress for t in ids)


# ------------------------------------------------------------------------------
# reply_streamer
# ------------------------------------------------------------------------------
# Publishing a reply while it is still being written.
#
# The contract that matters is agreement: whatever the streamer published, joined
# together, has to equal what a caller who simply waited would have received. That holds
# because both go through the same ``extract_assistant_reply`` - there is no streaming-
# flavored second copy of the extraction logic. What streaming adds is the hold-back,
# and most of what is worth testing is that it never lets half a boundary out.


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


# ------------------------------------------------------------------------------
# streaming_context
# ------------------------------------------------------------------------------
# Unit tests for :class:`praxis.generation.StreamingContext`.
#
# The streaming context is the shared helper that drives the "growing text buffer with
# reset-on-degeneracy" pattern used by both the backprop Lightning ``TerminalInterface``
# callback and the Ray Mono-Forward live-inference hook. These tests cover the
# degeneracy heuristics and the stuck-output reset path in isolation.


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


# ------------------------------------------------------------------------------
# generation_deadline
# ------------------------------------------------------------------------------
# The request deadline, which is what bounds a chat request's cost to the run.
#
# Queued generations are served by ``GenerationQueueCallback`` from inside
# ``on_train_batch_end``, so they hold the training loop's turn. The wait in
# ``generate_from_messages`` is client-side only: before the deadline existed, giving up
# after 60s stopped us listening but did not stop the loop from decoding the whole turn.
# Measured on ``abstractinator-r``, where the encoder stack cannot cache and every
# decode step is a full forward: one 512-byte Discord turn stalled training for 208
# seconds, ~148 of them after the client had already timed out and thrown the eventual
# reply away.
#
# Two things have to hold, and neither is visible from the caller's side:
#
# - a request that expires BEFORE it is served must never run, - a request served just
# under the wire must stop decoding when it expires, rather than running to
# ``max_new_tokens``.


class _SlowBackend:
    """Spends ``delay`` seconds per token and never halts on a boundary.

    Not halting is the point: it isolates the deadline as the only thing that
    can end the decode, so a passing test cannot be passing because the model
    happened to stop. The per-token check mirrors what the real backend gets
    from transformers - ``ModelBackend`` turns the deadline into
    ``GenerationConfig.max_time``, which ``_get_stopping_criteria`` builds into
    a ``MaxTimeCriteria`` evaluated after every token - because the caller only
    calls this ONCE for a turn with no tool in it.
    """

    model = None
    default_sampling_temperature = None

    def __init__(self, delay=0.01):
        self.delay = delay
        self.device = "cpu"
        self.max_positions = None
        self.calls = 0
        self.tokens_emitted = 0

    @contextlib.contextmanager
    def eval_mode(self):
        yield

    def generate_until_halt(self, tokens, step_kwargs, deadline=None, streamer=None):
        self.calls += 1
        budget = int(step_kwargs.get("max_new_tokens", 100))
        for _ in range(budget):
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.delay)
            # One ordinary byte ('a'), which is not a boundary under prose.
            nxt = torch.tensor([[ord("a")]], dtype=torch.long)
            tokens = torch.cat([tokens, nxt], dim=-1)
            self.tokens_emitted += 1
        return tokens


def _generator(tokenizer, backend):
    gen = Generator(backend=backend, tokenizer=tokenizer)
    gen.tools = {}
    return gen


def test_expired_request_is_never_served(tokenizer):
    """The stall the deadline exists to prevent: nobody is listening, so the
    training loop must not spend a single forward on it."""
    backend = _SlowBackend()
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 5000},
        deadline=time.time() - 1.0,
    )
    served = gen.fulfill_requests(max_requests=1)

    assert backend.calls == 0, "an abandoned request still ran the model"
    # Dropped, not silently forgotten: a late poller gets a falsy answer rather
    # than waiting out its own timeout on a request that will never run.
    assert gen.get_result(rid) == ""
    assert served == 0, "a drop must not spend the per-step generation budget"


def test_drops_do_not_consume_the_request_budget(tokenizer):
    """``max_requests`` bounds GENERATION per step. Expired requests run none,
    so a burst of them must clear in one drain rather than one step each."""
    backend = _SlowBackend()
    gen = _generator(tokenizer, backend)

    expired = time.time() - 1
    stale = [
        gen.request_generation("user\n\nhi\n\nassistant\n\n", {}, deadline=expired)
        for _ in range(5)
    ]
    live = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 2},
        deadline=time.time() + 30,
    )

    assert gen.fulfill_requests(max_requests=1) == 1
    assert all(gen.get_result(r) == "" for r in stale)
    assert gen.get_result(live) is not None


def test_decode_stops_at_the_deadline(tokenizer):
    """Served under the wire, then expires mid-decode.

    This is the case the between-steps check alone did NOT cover: a turn with
    no tool call enters ``generate_until_halt`` exactly once, so the deadline
    has to reach inside the decode or the request keeps the training loop for
    the full ``max_new_tokens`` regardless.
    """
    backend = _SlowBackend(delay=0.01)
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation(
        "user\n\nhi\n\nassistant\n\n",
        {"max_new_tokens": 5000},
        deadline=time.time() + 0.3,
    )
    started = time.time()
    gen.fulfill_requests(max_requests=1)
    elapsed = time.time() - started

    assert elapsed < 5.0, f"decode ran {elapsed:.1f}s past a 0.3s deadline"
    assert backend.calls == 1, "the turn should be one decode call"
    assert backend.tokens_emitted < 5000, "decode spent the whole budget anyway"
    # The partial turn still comes back rather than being discarded: reply_start
    # is the runtime's own offset and does not depend on halting cleanly.
    result = gen.get_result(rid)
    assert result is not None and result.reply_start is not None


def test_no_deadline_means_no_limit(tokenizer):
    """``/input`` polls forever and passes no deadline; that must keep working
    exactly as before."""
    backend = _SlowBackend(delay=0.0)
    gen = _generator(tokenizer, backend)

    rid = gen.request_generation("user\n\nhi\n\nassistant\n\n", {"max_new_tokens": 8})
    gen.fulfill_requests(max_requests=1)

    assert backend.tokens_emitted == 8
    assert gen.get_result(rid) is not None


# ------------------------------------------------------------------------------
# modeling
# ------------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# Lossless multi-token (speculative) inference for the byte-latent stack.
#
# The byte-latent core patches non-causally within a partial patch, so a single
# verify forward over ``committed + drafts`` reads contaminated earlier
# positions. The fix reads each truncated prefix at its OWN last real position
# (causal) and batches them behind an attention mask. Two properties make that
# lossless, and these tests pin both so a regression in either is caught:
#   1. padding invariance - a right-padded, mask-gated prefix predicts the same
#      last-real-position token as its unpadded form (incl. the prismatic4
#      CrystalVearHead router, which must route per-sequence and mask pads);
#   2. greedy speculative decoding reproduces byte-by-byte greedy exactly, up to
#      floating-point argmax ties (batched-GEMM reduction order) where greedy is
#      itself ill-defined.
# --------------------------------------------------------------------------- #


@pytest.fixture
def spec_config():
    """Byte-latent + prismatic4 head + dual memory + VEAR MTP (drafting stack)."""
    return PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
        memory_type="mal_energy_dual",
        mtp_type="vear",
        mtp_depth=4,
    )


def test_draft_window_from_mtp_depth(spec_config):
    """The terminal sizes its per-step budget off the ADAPTIVE draft window, so a
    step exercises MTP without over-drafting; without live MTP it collapses to a
    single token."""
    from praxis.generation.generator import Generator

    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    gen = Generator(model=model, tokenizer=None, device="cpu")
    # The window tracks the adaptive width (draft_width + 1), which starts
    # conservative: a fresh model drafts narrowly and widens only as runs land,
    # so a large mtp_depth costs nothing extra until acceptance earns it.
    assert gen.draft_window == model.mtp.draft_width + 1
    assert model.mtp.draft_width < spec_config.mtp_depth  # conservative at init

    saved = model.mtp
    model.mtp = None
    try:
        assert gen.draft_window == 1  # no MTP -> single-token throttle
    finally:
        model.mtp = saved


# ------------------------------------------------------------------------------
# tools
# ------------------------------------------------------------------------------
# Tool-calling tests.
#
# Tool-call boundaries are atomic special tokens
# (``[TOOL_CALL]``/``[/TOOL_CALL]``/``[TOOL_RESULT]``/``[/TOOL_RESULT]``). The tests
# exercise both the string-form helpers (format, parse, regex patterns) and the token-ID
# helpers used by the generator at runtime.


# ---------------------------------------------------------------------------
# Generator plumbing.
# ---------------------------------------------------------------------------


def test_generator_eos_token_id_list_includes_tool_close():
    """The generator must halt on [/TOOL_CALL] via the eos list."""
    from praxis.generation.generator import Generator

    mock_model = MagicMock()
    mock_model.training = False
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    tok = ByteLevelTokenizer()

    gen = Generator(mock_model, tok, device="cpu")
    eos_list = gen._eos_token_id_list()
    assert eos_list is not None
    assert tok.eos_token_id in eos_list
    assert tok.sep_token_id in eos_list
    assert tok.tool_call_end_token_id in eos_list
