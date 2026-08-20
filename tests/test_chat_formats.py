"""Tests for CHAT_FORMAT_REGISTRY and the text-boundary (prose) format.

The invariants worth pinning are the ones that silently produce a broken run
rather than an exception:

- the `default` profile must stay byte-identical, since every existing
  checkpoint's data pipeline depends on it,
- the boundary that ENDS a generated turn must be a trained target (the defect
  `prose` exists to remove),
- a stop-string scan must not re-halt on the boundary it resumed from, or the
  tool loop returns zero new tokens forever,
- the tool flow's three boundaries must classify unambiguously.
"""

import contextlib

import pytest
import torch

from praxis.data.formatters.tools import format_tool_calling
from praxis.data.validators import ChatTemplateValidator
from praxis.generation.request import GenerationRequest
from praxis.generation.stopping import (
    find_stop_cut,
    normalize_stop_strings,
    split_at_stop,
    trailing_stop,
)
from praxis.tokenizers import create_tokenizer
from praxis.tokenizers.chat_templates import (
    CHAT_FORMAT_REGISTRY,
    DEFAULT_CHAT_TEMPLATE,
    apply_chat_format,
    chat_format_of,
    get_chat_template,
    resolve_chat_format,
    tokenize_with_mask,
)
from praxis.tools import (
    build_result_splice_text,
    classify_boundary_halt,
    find_pending_call_text,
    format_call_body,
)

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]


def tokenizer_for(chat_format):
    return create_tokenizer(
        tokenizer_type="byte_level", vocab_size=1024, chat_format=chat_format
    )


@pytest.fixture(scope="module")
def prose_tokenizer():
    return tokenizer_for("prose")


@pytest.fixture(scope="module")
def default_tokenizer():
    return tokenizer_for("default")


# ---------------------------------------------------------------- registry


def test_registry_keys():
    assert set(CHAT_FORMAT_REGISTRY) == {"default", "prose"}


def test_unknown_format_is_a_hard_error():
    with pytest.raises(ValueError, match="Unknown chat_format"):
        resolve_chat_format("chatml-ish")
    with pytest.raises(ValueError, match="Unknown chat_format"):
        create_tokenizer(
            tokenizer_type="byte_level", vocab_size=1024, chat_format="chatml-ish"
        )


def test_default_template_unchanged(default_tokenizer):
    """Existing runs depend on this string; the registry must not perturb it."""
    assert default_tokenizer.chat_template == DEFAULT_CHAT_TEMPLATE
    assert get_chat_template("default") == DEFAULT_CHAT_TEMPLATE
    # Pre-registry call sites pass a tokenizer TYPE, which is not a format name.
    assert get_chat_template("byte_level") == DEFAULT_CHAT_TEMPLATE


def test_format_recovered_from_template_when_attribute_is_lost(prose_tokenizer):
    """`chat_format` is a plain attribute and does not survive
    save_pretrained; `chat_template` does. Losing the pairing would leave a
    prose template with the default halting contract, which never terminates."""
    tok = tokenizer_for("prose")
    del tok.chat_format
    assert chat_format_of(tok).name == "prose"


def test_apply_chat_format_sets_both_halves(default_tokenizer):
    tok = tokenizer_for("default")
    apply_chat_format(tok, "prose")
    assert chat_format_of(tok).name == "prose"
    assert tok.chat_template == CHAT_FORMAT_REGISTRY["prose"].template


# ------------------------------------------------------------- rendering


def test_prose_render_has_no_control_tokens(prose_tokenizer):
    text = prose_tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    for token in ("[BOS]", "[EOS]", "[SEP]", "[PAD]"):
        assert token not in text
    assert text.startswith("system\n\nYou are a helpful assistant.\n\nuser\n\n")
    assert text.endswith("Tokyo.\n\n")

    ids = prose_tokenizer.encode(text)
    assert all(i >= 4 for i in ids), "no id below OFFSET may survive"


def test_prose_generation_prompt(prose_tokenizer):
    text = prose_tokenizer.apply_chat_template(
        CONVERSATION[:2], tokenize=False, add_generation_prompt=True
    )
    assert text.endswith("assistant\n\n")


def test_prose_trains_the_boundary_that_ends_the_turn(prose_tokenizer):
    """The whole point: an assistant turn's mask must cover the boundary
    naming the next speaker, so the halt signal is a trained target."""
    enc = prose_tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    assert len(mask) == len(ids)

    text = prose_tokenizer.decode(ids, skip_special_tokens=False)
    trained = "".join(prose_tokenizer.decode([t]) for t, m in zip(ids, mask) if m)
    # First assistant turn ends by naming the next speaker.
    assert "Paris is the capital of France.\n\nuser\n\n" in trained
    # The user's own words are never a target.
    assert "And of Japan?" not in trained
    assert "And of Japan?" in text


def test_default_leaves_its_turn_opener_untrained(default_tokenizer):
    """The measured defect, pinned so a template edit cannot reintroduce it
    silently: under `default` the BOS opening a turn has zero gradient."""
    enc = default_tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    bos = default_tokenizer.bos_token_id
    bos_positions = [i for i, t in enumerate(ids) if t == bos]
    assert bos_positions, "sanity: the default format uses BOS"
    assert not any(mask[i] for i in bos_positions)


def test_prose_supervises_more_of_the_sequence(prose_tokenizer, default_tokenizer):
    def trained_fraction(tok):
        enc = tok.apply_chat_template(
            CONVERSATION,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        mask = enc["assistant_masks"]
        return sum(mask) / len(mask)

    assert trained_fraction(prose_tokenizer) > trained_fraction(default_tokenizer)


# ---------------------------------------------------------- patch budget


def test_prose_boundary_costs_fewer_patches(prose_tokenizer):
    """A control token cuts a patch unconditionally (the `|= tokens < OFFSET`
    in find_space_patch_start_ids runs after the run-collapse), so each one
    buys its own patch. Text boundaries fold into the newline run."""
    from praxis.encoders.byte_latent.patcher import (
        find_space_patch_start_ids,
        patch_lengths_from_start_ids,
    )

    def patch_count(text):
        ids = prose_tokenizer.encode(text)
        t = torch.tensor([ids])
        lengths = patch_lengths_from_start_ids(
            find_space_patch_start_ids(t), t.shape[1]
        )
        return int((lengths[0] > 0).sum())

    assert patch_count("France.\n\nuser\n\nAnd of Japan?") < patch_count(
        "France.\n[SEP]\n[BOS]user\nAnd of Japan?"
    )


# --------------------------------------------------------------- halting


def test_prose_halts_on_strings_plus_the_one_retained_id(
    prose_tokenizer, default_tokenizer
):
    """Turn boundaries are strings, and there is no halt ID at all.

    prose keeps no control token: the template emits none and the packer
    appends none, so an id-based halt would be a logit the data never makes a
    target. Halting is entirely by stop string, which is a trained target
    because the boundary sits inside the generated turn's span.
    """
    prose = chat_format_of(prose_tokenizer)
    assert prose.stop_token_ids(prose_tokenizer) == []
    assert "\n\nuser\n\n" in prose.stop_strings()

    default = chat_format_of(default_tokenizer)
    assert default.stop_token_ids(default_tokenizer) == [
        default_tokenizer.eos_token_id,
        default_tokenizer.sep_token_id,
    ]
    assert default.stop_strings() == ()


def test_reply_boundary_is_not_a_stop_string(prose_tokenizer):
    """Both the generation prompt and the post-tool splice END with the reply
    boundary; treating it as a stop string would halt every resumed step
    before it produced a token."""
    fmt = chat_format_of(prose_tokenizer)
    assert fmt.boundary(fmt.reply_role) not in fmt.stop_strings()


def test_stop_cut_lands_exactly_on_the_boundary(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    stops = fmt.stop_strings()
    prompt = "user\n\nhi\n\nassistant\n\n"
    full = prompt + "Hello there!\n\nuser\n\nbytes the model kept drafting"
    ids = prose_tokenizer.encode(full)
    start = len(prose_tokenizer.encode(prompt))

    keep = find_stop_cut(prose_tokenizer, ids, start, stops)
    assert keep is not None
    assert prose_tokenizer.decode(ids[:keep]) == prompt + "Hello there!\n\nuser\n\n"


def test_stop_cut_ignores_the_boundary_it_resumed_from(prose_tokenizer):
    """Resumption depends on this: after halting on the call boundary the
    sequence already ends with a stop string."""
    fmt = chat_format_of(prose_tokenizer)
    stops = fmt.stop_strings()
    ids = prose_tokenizer.encode("assistant\n\nlet me look\n\ncall\n\n")
    assert find_stop_cut(prose_tokenizer, ids, len(ids), stops) is None
    ids_plus = ids + prose_tokenizer.encode("{")
    assert find_stop_cut(prose_tokenizer, ids_plus, len(ids), stops) is None


def test_stop_cut_is_inert_without_stop_strings(default_tokenizer):
    ids = default_tokenizer.encode("anything at all\n\nuser\n\n")
    assert find_stop_cut(default_tokenizer, ids, 0, ()) is None


def test_stopping_helpers():
    assert normalize_stop_strings(None) == ()
    assert normalize_stop_strings("\n\nuser\n\n") == ("\n\nuser\n\n",)
    assert normalize_stop_strings(["a", "b"]) == ("a", "b")
    assert split_at_stop("hello\n\nuser\n\nrest", ("\n\nuser\n\n",)) == "hello"
    assert split_at_stop("no boundary", ("\n\nuser\n\n",)) == "no boundary"
    assert trailing_stop("x\n\ntool\n\n", ("\n\ntool\n\n", "\n\nuser\n\n")) == (
        "\n\ntool\n\n"
    )
    assert trailing_stop("x", ("\n\ntool\n\n",)) is None


# ------------------------------------------------------------- tool flow


def test_prose_tool_boundaries_classify(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    call_open = "user\n\n2+2?\n\nassistant\n\nlet me check.\n\ncall\n\n"
    assert classify_boundary_halt(call_open, fmt) == "call_open"

    body = format_call_body("calc", {"values": [2, 2], "op": "add"})
    call_close = call_open + body + "\n\ntool\n\n"
    assert classify_boundary_halt(call_close, fmt) == "call_close"

    # A plain turn terminator is neither; the caller treats it as done.
    assert classify_boundary_halt(call_open + body + "\n\nuser\n\n", fmt) is None


def test_prose_pending_call_round_trip(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    body = format_call_body("calc", {"values": [750, 485], "op": "mul"})
    text = f"assistant\n\nok\n\ncall\n\n{body}\n\ntool\n\n"
    assert find_pending_call_text(text, fmt) == {
        "name": "calc",
        "arguments": {"values": [750, 485], "op": "mul"},
    }
    assert build_result_splice_text("363750.0", fmt) == "363750.0\n\nassistant\n\n"


def test_prose_malformed_call_is_surfaced_not_guessed(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    call = find_pending_call_text("call\n\nnot json at all\n\ntool\n\n", fmt)
    assert call is not None and call["_malformed"] is True

    # A JSON scalar is valid JSON but cannot be a call.
    scalar = find_pending_call_text("call\n\n5\n\ntool\n\n", fmt)
    assert scalar is not None and scalar["_malformed"] is True


def test_pending_call_requires_the_result_boundary(prose_tokenizer):
    fmt = chat_format_of(prose_tokenizer)
    assert find_pending_call_text("call\n\n{}\n\n", fmt) is None
    assert find_pending_call_text("assistant\n\nhi\n\ntool\n\n", fmt) is None


def test_tool_training_data_matches_the_format(prose_tokenizer, default_tokenizer):
    """The formatter must ask the format for the LAYOUT, not just rendering."""
    prose_doc = format_tool_calling({}, [], prose_tokenizer)
    roles = [m["role"] for m in prose_doc["messages"]]
    assert "call" in roles and "tool" in roles
    call_msg = next(m for m in prose_doc["messages"] if m["role"] == "call")
    assert "[TOOL_CALL]" not in call_msg["content"]
    tool_msg = next(m for m in prose_doc["messages"] if m["role"] == "tool")
    assert "[TOOL_RESULT]" not in tool_msg["content"]

    default_doc = format_tool_calling({}, [], default_tokenizer)
    assert "call" not in [m["role"] for m in default_doc["messages"]]
    rendered = "".join(m["content"] for m in default_doc["messages"])
    assert "[TOOL_CALL]" in rendered and "[TOOL_RESULT]" in rendered


def test_prose_call_turn_is_supervised(prose_tokenizer):
    """A tool call is model-produced, so it has to be inside the mask - else
    the format reintroduces exactly the defect it exists to remove."""
    doc = format_tool_calling({}, [], prose_tokenizer)
    enc = prose_tokenizer.apply_chat_template(
        doc["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    trained = "".join(
        prose_tokenizer.decode([t])
        for t, m in zip(enc["input_ids"], enc["assistant_masks"])
        if m
    )
    assert '"name": "calc"' in trained
    # The call turn hands off to the tool, and that handoff is trained.
    assert "\n\ntool\n\n" in trained


def test_prose_call_boundary_is_supervised(prose_tokenizer):
    """The boundary that OPENS a call must be trained, not just its body.

    Every boundary is the tail of the turn before it, and a tail is supervised
    only when that turn is generated. Running `user` straight into `call` puts
    `\\n\\ncall\\n\\n` in the user turn's tail, so the model could continue a
    call it was handed but never decide to open one - the same defect as an
    untrainable `[BOS]`, relocated. The empty `assistant` turn is what moves
    the boundary inside a generated span.
    """
    doc = format_tool_calling({}, [], prose_tokenizer)
    ids, mask = tokenize_with_mask(prose_tokenizer, doc["messages"])
    trained = "".join(prose_tokenizer.decode([t]) for t, m in zip(ids, mask) if m)
    assert "\n\ncall\n\n" in trained


def test_prose_call_follows_the_generation_prompt(prose_tokenizer):
    """Training has to show the call from the position inference starts at.

    `_prepare_inputs` renders with `add_generation_prompt=True`, so a request
    ends at the reply boundary. If no assistant turn preceded the call in
    training, the model would have to open one from a context it never saw.
    """
    fmt = chat_format_of(prose_tokenizer)
    doc = format_tool_calling({}, [], prose_tokenizer)
    rendered = prose_tokenizer.apply_chat_template(doc["messages"], tokenize=False)

    prompt_tail = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
    )
    assert prompt_tail.endswith(f"{fmt.reply_role}\n\n")

    # The continuation inference asks for: reply boundary, then the call.
    assert f"{fmt.reply_role}\n\n{fmt.boundary(fmt.call_role)}" in rendered


# ------------------------------------------------------------- validation


def test_validator_accepts_prose_documents(prose_tokenizer):
    """The BOS-role check would reject every prose doc, silently draining the
    training stream, so the validator has to switch modes with the format."""
    validator = ChatTemplateValidator(tokenizer=prose_tokenizer)
    enc = prose_tokenizer.apply_chat_template(
        CONVERSATION, tokenize=True, return_dict=True
    )
    ids = torch.as_tensor(enc["input_ids"], dtype=torch.long)
    is_valid, report = validator.validate_and_report(ids, messages=CONVERSATION)
    assert is_valid, report


def test_validator_flags_a_missing_prose_boundary(prose_tokenizer):
    validator = ChatTemplateValidator(tokenizer=prose_tokenizer)
    ids = torch.as_tensor(
        prose_tokenizer.encode("system\n\nhi\n\nuser\n\nthere\n\n"), dtype=torch.long
    )
    # Claim an assistant turn the render does not contain.
    messages = CONVERSATION[:3]
    is_valid, report = validator.validate_and_report(ids, messages=messages)
    assert not is_valid
    assert "Missing boundary" in report


def test_validator_still_checks_bos_under_default(default_tokenizer):
    validator = ChatTemplateValidator(tokenizer=default_tokenizer)
    good = default_tokenizer.apply_chat_template(
        CONVERSATION, tokenize=True, return_dict=True
    )
    ids = torch.as_tensor(good["input_ids"], dtype=torch.long)
    assert validator.validate_token_sequence(ids)[0]

    bad = torch.as_tensor(
        default_tokenizer.encode("[BOS]not-a-role\nbody\n[SEP]\n"), dtype=torch.long
    )
    assert not validator.validate_token_sequence(bad)[0]


# ------------------------------------------------------------ web layer


def test_reply_extraction_under_prose(prose_tokenizer):
    from praxis.web.utils.formatters import extract_assistant_reply

    text = (
        "user\n\nWhat is 750 times 485?\n\nassistant\n\nlet me check.\n\n"
        'call\n\n{"name": "calc", "arguments": {}}\n\ntool\n\n363750.0\n\n'
        "assistant\n\n750 times 485 equals 363750.\n\nuser\n\n"
    )
    assert extract_assistant_reply(text, prose_tokenizer) == (
        "750 times 485 equals 363750."
    )


@pytest.mark.parametrize("role", ["user", "system", "developer", "tool", "call"])
def test_empty_prose_turn_does_not_leak_the_next_speaker(prose_tokenizer, role):
    """The seam case: the prompt already supplied the boundary's blank line.

    A generation prompt ends `...assistant\\n\\n`, so a model that writes
    nothing and goes straight to naming the next speaker emits only
    `<role>\\n\\n`. The full sequence ends in a real stop string and generation
    halts correctly - but the slice handed to the extractor is missing the
    leading `\\n\\n`, so a cut that only looks for the full `\\n\\n<role>\\n\\n`
    form returns the bare role word as if it were the model's answer.
    """
    from praxis.web.utils.formatters import (
        _EMPTY_REPLY_PLACEHOLDER,
        extract_assistant_reply,
    )

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}{role}\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == _EMPTY_REPLY_PLACEHOLDER


def test_prose_reply_keeps_a_role_word_that_is_only_prose(prose_tokenizer):
    """The cut needs the boundary's trailing blank line, not just the word."""
    from praxis.web.utils.formatters import extract_assistant_reply

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = f"{prompt}call me back later.\n\nuser\n\n"
    assert extract_assistant_reply(text, prose_tokenizer) == "call me back later."


def test_prose_has_no_separator_to_decode(prose_tokenizer):
    """Prose defines no control token, so none can appear in a decode.

    The reply extractor therefore only ever has to cut on role boundaries -
    there is no literal `[EOS]` string for it to trip over, because the id does
    not exist in this tokenizer at all.
    """
    from praxis.web.utils.formatters import (
        _EMPTY_REPLY_PLACEHOLDER,
        extract_assistant_reply,
    )

    assert prose_tokenizer.eos_token_id is None
    assert chat_format_of(prose_tokenizer).document_separator is None

    prompt = prose_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert extract_assistant_reply(
        f"{prompt}Hello there.\n\nuser\n\n", prose_tokenizer
    ) == ("Hello there.")
    assert (
        extract_assistant_reply(f"{prompt}\n\nuser\n\n", prose_tokenizer)
        == _EMPTY_REPLY_PLACEHOLDER
    )


def test_answered_call_is_not_pending_again(prose_tokenizer):
    """A spliced result ends the call. Re-finding it fabricates an error.

    Halting on the result boundary a second time in one request would rfind the
    same `call` turn and hand json.loads its body PLUS the spliced result and
    the reply that followed - which fails to parse, so an error result gets
    spliced over a turn that was already answered correctly.
    """
    fmt = chat_format_of(prose_tokenizer)
    answered = (
        'user\n\nWhat is 2+2?\n\nassistant\n\n\n\ncall\n\n{"name": "calc", '
        '"arguments": {}}\n\ntool\n\n4\n\nassistant\n\nIt is 4.\n\ntool\n\n'
    )
    assert find_pending_call_text(answered, fmt) is None


def test_reply_extraction_strips_tool_plumbing_under_default(default_tokenizer):
    from praxis.web.utils.formatters import extract_assistant_reply

    text = (
        "[BOS]assistant\n[TOOL_CALL]\n"
        '{"name": "calc", "arguments": {}}\n[/TOOL_CALL]\n'
        "The answer is 4.\n[SEP]\n"
    )
    assert extract_assistant_reply(text, default_tokenizer) == "The answer is 4."


def test_api_roles_exclude_runtime_injected_turns(prose_tokenizer):
    """A client must not be able to fabricate a tool result."""
    from praxis.web.utils.formatters import format_messages_to_chatml

    with pytest.raises(ValueError, match="Invalid role"):
        format_messages_to_chatml([{"role": "tool", "content": "999"}], prose_tokenizer)
    with pytest.raises(ValueError, match="Invalid role"):
        format_messages_to_chatml([{"role": "call", "content": "{}"}], prose_tokenizer)
    # Ordinary roles still render.
    assert format_messages_to_chatml(
        [{"role": "user", "content": "hi"}], prose_tokenizer
    ).endswith("assistant\n\n")


# ---------------------------------------------------------------- packing


def test_prose_survives_the_packer(prose_tokenizer):
    """The packer passes `omit_leading_bos` for docs appended mid-sequence.
    In prose the boundary IS the separator, so the template must ignore that
    flag rather than run one document's role name into the previous one's last
    word."""
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(tokenizer=prose_tokenizer, block_size=512)
    for _ in range(6):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    batch = manager.get_batch(batch_size=2)

    assert len(batch["batch"]) == 2
    for seq, mask in zip(batch["batch"], batch["assistant_mask"]):
        assert seq.shape == mask.shape
        text = prose_tokenizer.decode(seq, skip_special_tokens=False)
        assert "[BOS]" not in text
        assert "[SEP]" not in text
        assert "[TOOL_CALL]" not in text
        assert "[EOS]" not in text
        # Doc-to-doc seams read as plain prose. Nothing marks them in the
        # stream; the seam is carried by block_ids instead.
        assert "Tokyo.\n\nsystem\n\n" in text
    assert manager.get_validation_stats()["documents_skipped"] == 0


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

    def generate_until_halt(self, tokens, step_kwargs, deadline=None):
        stops = normalize_stop_strings(step_kwargs.get("stop_strings"))
        eos = step_kwargs.get("eos_token_id") or []
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        budget = int(step_kwargs.get("max_new_tokens", 100))
        start = tokens.shape[1]
        ids = tokens[0].tolist()
        produced = 0
        while self.pending and produced < budget:
            ids.append(self.pending.pop(0))
            produced += 1
            if ids[-1] in eos:
                break
            if stops and find_stop_cut(self.tokenizer, ids, start, stops) is not None:
                break
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


def test_seam_cut_needs_the_boundary_to_be_the_whole_turn(prose_tokenizer):
    """A role word merely OPENING ordinary prose is not an empty turn.

    Unreachable from the decode loop - every role here is a stop string, so
    generation halts at the seam and the slice really is just the boundary -
    but `extract_assistant_reply` is public and also takes bare strings.
    """
    from praxis.web.utils.formatters import (
        _EMPTY_REPLY_PLACEHOLDER,
        _extract_reply_text_boundaries,
    )

    fmt = chat_format_of(prose_tokenizer)
    empty = _extract_reply_text_boundaries("user\n\n", fmt, prose_tokenizer, 0)
    assert empty == ""
    kept = _extract_reply_text_boundaries(
        "call\n\nme back later.", fmt, prose_tokenizer, 0
    )
    assert kept == "call\n\nme back later."
    assert _EMPTY_REPLY_PLACEHOLDER  # the caller substitutes it for `empty`


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


def test_tool_splice_anchors_the_turn_it_opens(default_tokenizer):
    """The splice ends with a role transition, so it opens a new reply turn.

    `build_result_splice_ids` writes `\\n[SEP]\\n[BOS]assistant\\n` after the
    result, which is the boundary the reply belongs to.
    """
    script = (
        "Checking now.\n[TOOL_CALL]\n"
        '{"name": "get_time", "arguments": {}}\n[/TOOL_CALL]\n'
        "It is noon.[SEP]"
    )
    reply = _scripted_reply(
        default_tokenizer,
        script,
        tools={"get_time": {}},
        call_tool=lambda name, args: "noon",
    )
    assert reply == "It is noon."


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


def test_bare_string_still_falls_back_to_scanning(prose_tokenizer):
    """Callers holding plain text (not a GenerationResult) keep working."""
    from praxis.web.utils.formatters import extract_assistant_reply

    raw = "user\n\nWhat is 2+2?\n\nassistant\n\nIt is 4.\n\nuser\n\n"
    assert extract_assistant_reply(raw, prose_tokenizer) == "It is 4."


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


# ------------------------------------------------- speculative decode halt


class _ScriptedMTP:
    """MTP stub that drafts nothing, so each step commits the main forward's
    token plus the bonus token - a two-byte commit, which is the case where a
    boundary can complete mid-run."""

    byte_level = True

    def __init__(self):
        self.accepted = []

    def draft_next_tokens(self, hidden, token_0, embed_fn, head):
        return token_0.new_zeros((1, 0))

    def note_accepted(self, n):
        self.accepted.append(n)


class _ScriptedModel:
    """The minimum surface ``_speculative_generate`` touches, driven by a fixed
    list of byte ids so the loop's output is deterministic."""

    encoder = object()  # truthy: take the byte-latent branch
    embeds = None
    head = None

    def __init__(self, script, prompt_len, vocab_size=264):
        self.script = list(script)
        self.prompt_len = prompt_len
        self.vocab_size = vocab_size
        self.mtp = _ScriptedMTP()

    def _one_hot(self, index):
        logits = torch.full((1, self.vocab_size), -10.0)
        if 0 <= index < len(self.script):
            logits[0, self.script[index]] = 10.0
        else:
            logits[0, 0] = 10.0  # past the script: emit PAD
        return logits

    def _spec_logits_and_hidden(self, generated, attention_mask=None):
        produced = generated.size(1) - self.prompt_len
        logits = torch.full((1, generated.size(1), self.vocab_size), -10.0)
        logits[:, -1, :] = self._one_hot(produced)
        return logits, torch.zeros(1, generated.size(1), 8)

    def _verify_prefixes_batched(self, generated, candidates):
        produced = generated.size(1) - self.prompt_len
        return self._one_hot(produced + candidates.size(1))


def _run_scripted(tokenizer, prompt, continuation, stop_strings, max_new_tokens=200):
    from transformers import GenerationConfig

    from praxis.modeling import PraxisForCausalLM

    prompt_ids = tokenizer.encode(prompt)
    script = tokenizer.encode(continuation)
    model = _ScriptedModel(script, prompt_len=len(prompt_ids))
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        stop_strings=list(stop_strings) or None,
    )
    out = PraxisForCausalLM._speculative_generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long),
        config,
        tokenizer=tokenizer,
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def test_speculative_decode_halts_on_a_text_boundary(prose_tokenizer):
    """The path abstractinator-g actually decodes through. It owns its own
    sampling, so transformers' StopStringCriteria never runs and without an
    explicit check every prose turn would run to max_new_tokens."""
    fmt = chat_format_of(prose_tokenizer)
    prompt = "user\n\nhi\n\nassistant\n\n"
    text = _run_scripted(
        prose_tokenizer,
        prompt,
        "Hello there!\n\nuser\n\nand plenty more drafted bytes past the boundary",
        fmt.stop_strings(),
    )
    assert text == prompt + "Hello there!\n\nuser\n\n"


def test_speculative_decode_halts_mid_commit(prose_tokenizer):
    """Each step commits two bytes here, so the boundary's last byte can land
    on the first of a pair; the cut must still be exact."""
    fmt = chat_format_of(prose_tokenizer)
    prompt = "user\n\nhi\n\nassistant\n\n"
    for filler in ("ok", "yes"):  # shifts the boundary's parity
        text = _run_scripted(
            prose_tokenizer, prompt, f"{filler}\n\nuser\n\nXXXXXX", fmt.stop_strings()
        )
        assert text == prompt + f"{filler}\n\nuser\n\n"


def test_speculative_decode_without_stop_strings_runs_on(prose_tokenizer):
    """Control: the halt comes from the stop strings, not from the script."""
    prompt = "user\n\nhi\n\nassistant\n\n"
    text = _run_scripted(
        prose_tokenizer,
        prompt,
        "Hello there!\n\nuser\n\nkeeps going",
        (),
        max_new_tokens=40,
    )
    assert text.startswith(prompt + "Hello there!\n\nuser\n\nkeeps going")
    assert len(text) > len(prompt + "Hello there!\n\nuser\n\n")


# ------------------------------------------- assistant mask on non-ASCII text


MULTIBYTE = [
    {"role": "system", "content": "“quoted”"},
    {"role": "user", "content": "Calculate √1156 \U0001f600"},
    {"role": "assistant", "content": "— the answer is 34."},
    {"role": "user", "content": "and été?"},
    {"role": "assistant", "content": "Summer."},
]


def _trained_text(tok, messages):
    from praxis.tokenizers.chat_templates import tokenize_with_mask

    ids, mask = tokenize_with_mask(tok, messages)
    return tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_mask_is_exact_on_multibyte_text(fmt_name):
    """HuggingFace's return_assistant_tokens_mask maps CHARACTER offsets to token
    spans, which slips wherever one character is several tokens. Measured on the
    byte tokenizer before the fix: every multi-byte character before a span shifted
    the prose mask two tokens (cumulatively, so 'The answer' became 'r is'), and a
    multi-byte character starting a span shifted the default mask by its byte
    length. That is a silently corrupted SFT objective on any text with a curly
    quote, accent, em dash or emoji - which is most real text."""
    tok = tokenizer_for(fmt_name)
    trained = _trained_text(tok, MULTIBYTE)
    # Every assistant turn, whole and unshifted.
    assert "— the answer is 34." in trained
    assert "Summer." in trained
    # And nothing from a prompt turn.
    assert "quoted" not in trained
    assert "1156" not in trained
    assert "été" not in trained


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_segment_join_is_byte_identical_to_the_template(fmt_name):
    """The segment split only stays safe while it renders exactly what Jinja
    does - otherwise the fix silently changes the training data."""
    tok = tokenizer_for(fmt_name)
    fmt = chat_format_of(tok)
    cases = [CONVERSATION, MULTIBYTE, [{"role": "user", "content": "solo"}]]
    for messages in cases:
        for add_gen in (False, True):
            for omit in (False, True):
                kwargs = {"add_generation_prompt": add_gen}
                if omit:
                    kwargs["omit_leading_bos"] = True
                jinja = tok.apply_chat_template(messages, tokenize=False, **kwargs)
                segments = fmt.render_segments(
                    messages,
                    tok,
                    add_generation_prompt=add_gen,
                    omit_leading_bos=omit,
                )
                assert "".join(text for text, _ in segments) == jinja


@pytest.mark.parametrize("fmt_name", ["default", "prose"])
def test_segment_ids_match_whole_string_encoding(fmt_name):
    """Piece-wise encoding is only equivalent for merge-free tokenizers; this is
    the property that licenses the whole approach."""
    from praxis.tokenizers.chat_templates import tokenize_with_mask

    tok = tokenizer_for(fmt_name)
    ids, mask = tokenize_with_mask(tok, MULTIBYTE)
    whole = tok.encode(
        tok.apply_chat_template(MULTIBYTE, tokenize=False), add_special_tokens=False
    )
    assert ids == whole
    assert len(mask) == len(ids)


def test_bpe_keeps_the_offset_mask():
    """A merge can straddle a segment boundary, so piece-wise encoding would
    change BPE's tokenization. Those tokenizers must decline the segment path -
    and they do not need it: their characters map to tokens cleanly."""
    from praxis.tokenizers.chat_templates import tokenize_with_mask
    from praxis.tokenizers.standard import StandardTokenizer

    bpe = StandardTokenizer(tokenizer_type="bpe", vocab_size=1024)
    assert not getattr(bpe, "context_free_tokenization", False)
    assert tokenize_with_mask(bpe, CONVERSATION) is None


def test_packer_uses_the_exact_mask(prose_tokenizer):
    """End to end through the packer: no assistant content may be dropped and no
    prompt content admitted, on multi-byte text."""
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(tokenizer=prose_tokenizer, block_size=4096)
    manager.add_document({"messages": MULTIBYTE, "metadata": {}})
    batch = manager.get_batch(batch_size=1)
    seq, mask = batch["batch"][0], batch["assistant_mask"][0]
    trained = prose_tokenizer.decode(
        [int(t) for t, m in zip(seq, mask) if m], skip_special_tokens=False
    )
    assert "— the answer is 34." in trained
    assert "Summer." in trained
    assert "1156" not in trained


# ------------------------------------------------- control-token inventory
#
# The head is exactly as wide as the tokenizer's alphabet, so every id the
# format cannot make a target is an id sampling can still pick. These pin the
# inventory in both directions: what must not exist, and what must.


def test_prose_does_not_register_the_tool_tokens(prose_tokenizer, default_tokenizer):
    """No [TOOL_CALL] id under prose - it lays tool calls out as ordinary turns.

    Registering them anyway is not cosmetic: byte_alphabet_size counts them and
    the model's output head is sized from it, so four logits would exist that no
    training example can ever make a target.
    """
    assert not prose_tokenizer.tool_tokens_registered
    assert "[TOOL_CALL]" not in prose_tokenizer.get_vocab()
    assert prose_tokenizer.tool_call_token_id is None
    # The string is now ordinary text, so it encodes to its bytes.
    assert len(prose_tokenizer.encode("[TOOL_CALL]")) == len("[TOOL_CALL]")

    # default keeps them: its template renders them as atomic markers.
    assert default_tokenizer.tool_tokens_registered
    assert default_tokenizer.encode("[TOOL_CALL]") == [
        default_tokenizer.tool_call_token_id
    ]


def test_alphabet_and_head_shrink_together(prose_tokenizer, default_tokenizer):
    """The tokenizer's alphabet IS the byte-latent head width.

    A tokenizer that drops tokens without the encoder following would leave the
    head sized for ids the tokenizer can no longer produce, which is the same
    defect with the sign flipped.
    """
    from types import SimpleNamespace

    from praxis.encoders.byte_latent.config import create_base_config

    def head_width(tok):
        cfg = SimpleNamespace(
            byte_vocab_size=tok.byte_alphabet_size,
            hidden_size=64,
            embed_size=32,
            dropout=0.0,
            epsilon=1e-5,
            max_position_embeddings=1024,
            meta=[],
        )
        return create_base_config(cfg).local_vocab_size

    assert default_tokenizer.byte_alphabet_size == 264  # 256 + 4 named + 4 tool
    assert prose_tokenizer.byte_alphabet_size == 256  # pure bytes, nothing else
    assert head_width(default_tokenizer) == 264
    assert head_width(prose_tokenizer) == 256
    # The offset moves with the alphabet, or byte arithmetic downstream breaks.
    assert default_tokenizer.byte_offset == 4
    assert prose_tokenizer.byte_offset == 0


def test_packer_emits_block_ids_for_every_document(prose_tokenizer, default_tokenizer):
    """Packing needs the seam to be findable, and the packer states it.

    block_ids segment the local encoder's attention so it cannot read across
    unrelated documents. They come from the packer, which is the only step that
    knows where documents meet - so this holds identically for a format that
    writes a separator id and one that writes nothing at all.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    for tok in (prose_tokenizer, default_tokenizer):
        manager = MessageQueueManager(
            tokenizer=tok, block_size=4096, enable_chat_validation=False
        )
        for _ in range(3):
            manager.add_document({"messages": CONVERSATION, "metadata": {}})
        batch = manager.get_batch(batch_size=1)
        seq, blocks = batch["batch"][0], batch["block_ids"][0]

        assert blocks.shape == seq.shape
        # Three documents, each its own block, plus a block for the pad tail.
        assert sorted(set(blocks.tolist())) == [1, 2, 3, 4]
        # Blocks are contiguous runs: a document never resumes after another.
        runs = [int(blocks[0])]
        for prev, cur in zip(blocks[:-1], blocks[1:]):
            if cur != prev:
                runs.append(int(cur))
        assert runs == sorted(set(runs))


def test_prose_writes_no_control_id_into_the_stream(prose_tokenizer):
    """The point of the pure-byte layout: every id is a byte.

    Nothing below 256 is reserved, so there is no id in a packed sequence that
    could not equally have come from the text itself.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(
        tokenizer=prose_tokenizer, block_size=4096, enable_chat_validation=False
    )
    for _ in range(3):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    seq = manager.get_batch(batch_size=1)["batch"][0]

    assert (
        chat_format_of(prose_tokenizer).document_separator_id(prose_tokenizer) is None
    )
    assert int(seq.max()) < 256
    assert int(seq.min()) >= 0


def test_prose_documents_end_on_their_own_text(prose_tokenizer):
    """No separator is appended, so a document ends where its text ends.

    The old layout appended [EOS] and copied the last mask value onto it. With
    the separator gone there is nothing to supervise or to mask: the trailing
    boundary the template already emits is the halt signal, and it is inside
    the generated turn's span.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(
        tokenizer=prose_tokenizer, block_size=4096, enable_chat_validation=False
    )
    ends_generated = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    ends_prompt = [{"role": "user", "content": "unanswered"}]

    ids, mask = manager._tokenize_doc(
        {"messages": ends_generated, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("yo\n\n")
    assert int(mask[-1]) == 1  # the halting boundary is a trained target

    ids, mask = manager._tokenize_doc(
        {"messages": ends_prompt, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("unanswered\n\n")
    assert int(mask[-1]) == 0


def test_unproducible_control_ids_are_suppressed(prose_tokenizer, default_tokenizer):
    """Suppression covers ids a format cannot train but could still sample.

    In the BLT layout ids 0-3 exist whether or not a format uses them, so the
    unused ones are kept out of samples. In the pure-byte layout there is
    nothing to suppress, because there is no id that is not a byte - which is
    the stronger version of the same guarantee.
    """
    prose = chat_format_of(prose_tokenizer)
    assert prose.suppressed_token_ids(prose_tokenizer) == []
    assert prose_tokenizer.byte_alphabet_size == 256

    # default renders BOS and SEP, so only PAD is unreachable there.
    default = chat_format_of(default_tokenizer)
    assert default.suppressed_token_ids(default_tokenizer) == [
        default_tokenizer.pad_token_id
    ]


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
