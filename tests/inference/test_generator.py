"""The ``Generator``: preparing a request, bounding its cost, and serving a turn.

Most of these drive the real ``Generator`` over a ``ScriptedBackend``, so the whole
endpoint path runs - prompt construction, the halt contract, the tool state machine,
reply extraction, streaming - with the model's output pinned instead of sampled.
"""

from __future__ import annotations

import json
import time

import pytest
import torch

from praxis.inference.generator import Generator
from praxis.inference.reply import extract_assistant_reply
from praxis.inference.request import GenerationRequest
from praxis.modeling import PraxisForCausalLM
from praxis.tokenizers.chat_templates import PROSE_FORMAT, chat_format_of
from praxis.web.utils.formatters import generate_from_messages
from tests.inference.scripted import ScriptedBackend, Sink
from tests.stubs import _SlowBackend

PROMPT = "user\n\nhi\n\nassistant\n\n"


def _generator(tokenizer, backend=None, **kwargs):
    gen = Generator(
        backend=backend or ScriptedBackend(tokenizer, ""), tokenizer=tokenizer, **kwargs
    )
    gen.tools = {}
    return gen


def _prepare(gen, prompt, **kwargs):
    """``(input_ids, gen_kwargs)`` for a request, without decoding anything."""
    input_ids, gen_kwargs, _, _ = gen._prepare_inputs(
        GenerationRequest(id="t", prompt=prompt, kwargs=kwargs)
    )
    return input_ids, gen_kwargs


# ---------------------------------------------------------------------------
# preparing a request
# ---------------------------------------------------------------------------


def test_prepared_kwargs_carry_the_formats_halt_and_suppression_contract(
    default_tokenizer, prose_tokenizer
):
    """A declared halt or suppression that never reaches generate() is
    decoration, so this reads the kwargs the decode actually receives."""
    tok = default_tokenizer
    _, kwargs = _prepare(_generator(tok), "hi")
    # default halts on ids - including the tool-call close, so the tool can run.
    assert {tok.eos_token_id, tok.sep_token_id, tok.tool_call_end_token_id} <= set(
        kwargs["eos_token_id"]
    )
    # Its only unproducible id is PAD; BOS and SEP are rendered.
    assert kwargs["suppress_tokens"] == [tok.pad_token_id]
    assert "stop_strings" not in kwargs

    # prose halts on text and has no id left to suppress.
    _, kwargs = _prepare(_generator(prose_tokenizer), "hi")
    assert kwargs["stop_strings"] == list(
        chat_format_of(prose_tokenizer).stop_strings()
    )
    assert "eos_token_id" not in kwargs
    assert "suppress_tokens" not in kwargs


def test_prompt_truncation_lands_on_a_character_boundary(prose_tokenizer):
    """A byte-level token is a byte, so a raw token-index cut can sever a
    multi-byte character. Decode then yields U+FFFD, a rolling context
    re-encodes that as three real bytes, and every later round truncates again
    - minting replacement characters for as long as the buffer lives. The cut
    is aligned forward, never backward, so it also stays inside the budget."""
    gen = _generator(prose_tokenizer)
    text = "日本語のテキストです"
    for budget in range(4, len(prose_tokenizer.encode(text))):
        input_ids, _ = _prepare(
            gen, text, max_new_tokens=4, truncate_to=budget, skip_special_tokens=False
        )
        decoded = prose_tokenizer.decode(
            input_ids[0].tolist(), skip_special_tokens=False
        )
        assert "�" not in decoded, f"severed character at truncate_to={budget}"
        assert input_ids.size(1) <= budget


def test_truncation_is_unchanged_for_tokenizers_without_the_hook():
    """One token = whole characters there, so there is nothing to align and
    the cut must stay exactly where the budget puts it."""

    class _Whole:
        chat_format = PROSE_FORMAT
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

    tok = _Whole()
    assert not hasattr(tok, "align_left_cut")
    input_ids, _ = _prepare(
        _generator(tok),
        "abcdefghij",
        max_new_tokens=1,
        truncate_to=4,
        skip_special_tokens=False,
    )
    assert input_ids[0].tolist() == [ord(c) for c in "ghij"]


def test_draft_window_from_mtp_depth(spec_config):
    """The terminal sizes its per-step budget off the ADAPTIVE draft window, so a
    step exercises MTP without over-drafting; without live MTP it collapses to a
    single token."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(spec_config).eval()
    gen = Generator(model=model, tokenizer=None, device="cpu")
    # The window tracks the adaptive width (draft_width + 1), which starts
    # conservative: a fresh model drafts narrowly and widens only as runs land,
    # so a large mtp_depth costs nothing extra until acceptance earns it.
    assert gen.draft_window == model.mtp.draft_width + 1
    assert model.mtp.draft_width < spec_config.mtp_depth

    model.mtp = None
    assert gen.draft_window == 1


# ---------------------------------------------------------------------------
# the request deadline
# ---------------------------------------------------------------------------
#
# Queued generations are served from inside ``on_train_batch_end``, so they
# hold the training loop's turn. Measured on abstractinator-r, one 512-byte
# Discord turn stalled training for 208s, ~148 of them after the client had
# already timed out. A request that expires before it is served must never
# run, and one served under the wire must stop decoding when it expires.


def test_expired_request_is_never_served(prose_tokenizer):
    """Nobody is listening, so the training loop must not spend a single
    forward on it."""
    backend = _SlowBackend()
    gen = _generator(prose_tokenizer, backend)

    rid = gen.request_generation(
        PROMPT, {"max_new_tokens": 5000}, deadline=time.time() - 1.0
    )
    served = gen.fulfill_requests(max_requests=1)

    assert backend.calls == 0, "an abandoned request still ran the model"
    # Dropped, not silently forgotten: a late poller gets a falsy answer rather
    # than waiting out its own timeout on a request that will never run.
    assert gen.get_result(rid) == ""
    assert served == 0, "a drop must not spend the per-step generation budget"


def test_drops_do_not_consume_the_request_budget(prose_tokenizer):
    """``max_requests`` bounds GENERATION per step. Expired requests run none,
    so a burst of them must clear in one drain rather than one step each."""
    gen = _generator(prose_tokenizer, _SlowBackend())

    expired = time.time() - 1
    stale = [gen.request_generation(PROMPT, {}, deadline=expired) for _ in range(5)]
    live = gen.request_generation(
        PROMPT, {"max_new_tokens": 2}, deadline=time.time() + 30
    )

    assert gen.fulfill_requests(max_requests=1) == 1
    assert all(gen.get_result(r) == "" for r in stale)
    assert gen.get_result(live) is not None


def test_decode_stops_at_the_deadline(prose_tokenizer):
    """Served under the wire, then expires mid-decode.

    A turn with no tool call enters ``generate_until_halt`` exactly once, so a
    between-steps check alone would not cover it: the deadline has to reach
    inside the decode or the request keeps the loop for the full budget.
    """
    backend = _SlowBackend(delay=0.01)
    gen = _generator(prose_tokenizer, backend)

    rid = gen.request_generation(
        PROMPT, {"max_new_tokens": 5000}, deadline=time.time() + 0.3
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


def test_no_deadline_means_no_limit(prose_tokenizer):
    """``/input`` polls forever and passes no deadline; that must keep working."""
    backend = _SlowBackend(delay=0.0)
    gen = _generator(prose_tokenizer, backend)

    rid = gen.request_generation(PROMPT, {"max_new_tokens": 8})
    gen.fulfill_requests(max_requests=1)

    assert backend.tokens_emitted == 8
    assert gen.get_result(rid) is not None


# ---------------------------------------------------------------------------
# where the reply starts
# ---------------------------------------------------------------------------


def _scripted_reply(
    tokenizer, script, max_new_tokens=200, tools=None, call_tool=None, messages=None
):
    """The reply the web layer returns for a turn the model writes as ``script``."""
    generator = _generator(
        tokenizer, ScriptedBackend(tokenizer, script), synchronous=True
    )
    if tools is not None:
        generator.tools = tools
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
        # A second assistant turn ends the first one rather than anchoring the
        # reply, which discarded everything the model actually said.
        (
            "Sure, here goes.\n\nassistant\n\nSecond thought.\n\nuser\n\n",
            "Sure, here goes.",
        ),
        # ...and a second turn that ran out of budget is not an empty reply.
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
    "script",
    [
        # The model repeating the boundary the prompt just wrote is noise. The
        # cut treats a reply-role boundary as end-of-turn, so without skipping
        # the repetition first it zeroed the reply that followed.
        "assistant\n\nHere is the answer.\n\nuser\n\n",
        "\n\nassistant\n\nHere is the answer.\n\nuser\n\n",
    ],
)
def test_prose_seam_repetition_does_not_eat_the_reply(prose_tokenizer, script):
    assert _scripted_reply(prose_tokenizer, script) == "Here is the answer."


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
    """A spliced tool result is not model output, so it does not spend the
    budget - but it does spend the context, and the loop has to notice.
    `_prepare_inputs` caps the prompt at `mpe - max_new_tokens` so the context
    can never overflow learned positions."""
    script = (
        '\n\ncall\n\n{"name": "get_time", "arguments": {}}\n\ntool\n\n'
        "and here is a long tail.\n\nuser\n\n"
    )
    backend = ScriptedBackend(prose_tokenizer, script)
    backend.max_positions = 256
    generator = _generator(prose_tokenizer, backend)
    generator.tools = {"get_time": {}}
    generator.call_tool = lambda name, args: "R" * 900
    request = GenerationRequest(id="t", prompt=PROMPT, kwargs={"max_new_tokens": 128})
    out = generator._process_single_request(request)
    assert len(prose_tokenizer.encode(str(out), add_special_tokens=False)) <= 256


def test_spliced_tool_result_does_not_spend_the_caller_budget(prose_tokenizer):
    """The model did not write the tool result, so it must not be charged for
    it: one fat result against the web default of 256 used to drive the budget
    negative and break the loop before the model ever spoke."""
    script = (
        '\n\ncall\n\n{"name": "get_time", "arguments": {}}\n\ntool\n\n'
        "I looked it up.\n\nuser\n\n"
    )
    reply = _scripted_reply(
        prose_tokenizer,
        script,
        max_new_tokens=64,
        tools={"get_time": {}},
        call_tool=lambda name, args: "x" * 800,
    )
    assert reply.startswith("I looked")


# ---------------------------------------------------------------------------
# streaming a turn as it is written
# ---------------------------------------------------------------------------


def test_generator_publishes_deltas_that_match_its_own_result(prose_tokenizer):
    """Streaming is additive: the deltas joined equal the reply the caller gets
    back."""
    gen = _generator(
        prose_tokenizer, ScriptedBackend(prose_tokenizer, "Hello!\n\nuser\n\nnot mine")
    )
    sink = Sink()
    rid = gen.request_generation(PROMPT, {"max_new_tokens": 64}, on_text=sink.text)
    gen.fulfill_requests(max_requests=1)
    result = gen.get_result(rid)

    assert sink.joined == extract_assistant_reply(result, prose_tokenizer)
    assert sink.joined == "Hello!"


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
    gen = _generator(
        default_tokenizer, ScriptedBackend(default_tokenizer, script), synchronous=True
    )
    gen.tools = {"get_time": {}}
    gen.call_tool = lambda name, args: "noon"

    sink = Sink()
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


# ---------------------------------------------------------------------------
# announcing a tool that ran
# ---------------------------------------------------------------------------
#
# The reply is the wrong place to look: the extractor strips the whole
# call/result exchange out of it, so a turn that consulted a tool and a turn
# that made the same claim up produce byte-identical text. ``on_tool``, fired
# from the branch that executes the tool, is the only account of the
# difference. It names the tool that actually RAN, never a call that did not,
# and it survives the reset that follows the splice.


def _call(name, arguments):
    """One prose call/answer exchange, halting at both tool boundaries."""
    body = json.dumps({"name": name, "arguments": arguments})
    return f"let me check\n\ncall\n\n{body}\n\ntool\n\n"


def _run(tokenizer, script, **callbacks):
    gen = Generator(backend=ScriptedBackend(tokenizer, script), tokenizer=tokenizer)
    rid = gen.request_generation(PROMPT, {"max_new_tokens": 4000}, **callbacks)
    gen.fulfill_requests()
    return gen.get_result(rid)


ADD = {"values": [2, 3], "op": "add"}


def test_a_tool_that_runs_is_announced_by_name(prose_tokenizer):
    seen = []
    result = _run(
        prose_tokenizer, _call("calc", ADD) + "five\n\nuser\n\n", on_tool=seen.append
    )

    assert seen == ["calc"]
    # ...and the result really did come back through the tool path.
    assert "5" in result


def test_each_execution_is_announced_separately(prose_tokenizer):
    """The chips count executions, so the callback fires per call rather than
    once per distinct tool. The arguments differ because an identical repeat
    is stopped as a duplicate - which is the next test."""
    seen = []
    _run(
        prose_tokenizer,
        _call("calc", ADD)
        + _call("calc", {"values": [4, 5], "op": "mul"})
        + "twenty\n\nuser\n\n",
        on_tool=seen.append,
    )

    assert seen == ["calc", "calc"]


def test_a_duplicate_call_is_not_announced(prose_tokenizer):
    """`execute_tool_call` stops the loop on a repeat rather than running the
    tool again, so counting it would report work that never happened."""
    seen = []
    _run(
        prose_tokenizer,
        _call("calc", ADD) + _call("calc", ADD) + "five\n\nuser\n\n",
        on_tool=seen.append,
    )

    assert seen == ["calc"]


def test_a_nameless_call_is_not_announced(prose_tokenizer):
    """A malformed body still gets an error result spliced back for the model
    to recover from, but there is no tool to name."""
    seen = []
    _run(
        prose_tokenizer,
        "\n\ncall\n\nnot json at all\n\ntool\n\nsorry\n\nuser\n\n",
        on_tool=seen.append,
    )

    assert seen == []


def test_the_announcement_precedes_the_reset_it_causes(prose_tokenizer):
    """The splice moves the turn anchor, so every call is followed by "drop
    what you have"; a chip that arrived after that would look like something
    the reset should have taken back."""
    events = []
    _run(
        prose_tokenizer,
        _call("calc", ADD) + "five\n\nuser\n\n",
        on_text=lambda text: events.append(("text", text)),
        on_reset=lambda: events.append(("reset", None)),
        on_tool=lambda name: events.append(("tool", name)),
    )

    kinds = [kind for kind, _ in events]
    assert "tool" in kinds and "reset" in kinds
    assert kinds.index("tool") < kinds.index("reset")


def test_a_raising_callback_never_reaches_the_training_loop(prose_tokenizer):
    """This fires from inside ``on_train_batch_end`` for a queued request. A
    chat-UI affordance must not be able to take the run down."""

    def explode(name):
        raise RuntimeError("the browser went away")

    result = _run(
        prose_tokenizer, _call("calc", ADD) + "five\n\nuser\n\n", on_tool=explode
    )

    assert "5" in result
