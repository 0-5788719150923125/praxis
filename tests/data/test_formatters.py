"""Tests for praxis/data/formatters: whitespace normalization and paragraph
reformatting (base.py), the conversation, preference-pair and joke formatters,
and the synthetic tool-calling formatter."""

import random

import pytest

from praxis.data.formatters import (
    format_joke,
    format_preference_pair,
    normalize_escaped_whitespace,
    text_formatter,
)
from praxis.data.formatters.conversation import format_human_assistant
from praxis.data.formatters.tools import format_tool_calling
from praxis.tasks import TaskType
from praxis.tokenizers.chat_templates import chat_format_of, tokenize_with_mask
from praxis.tools import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESULT_CLOSE,
    TOOL_RESULT_OPEN,
)


@pytest.fixture
def seeded_random():
    """Seed the global ``random`` the formatters draw from, then restore it."""
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


# ------------------------------------------------------------------------------
# base.py: escaped whitespace
# ------------------------------------------------------------------------------
# Some upstream datasets flatten strings during export, so real newlines arrive as
# literal backslash-n pairs. Unnormalized, the model learns to emit the literal
# sequence instead of a paragraph break.


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param("one\\n\\ntwo", "one\n\ntwo", id="flattened paragraphs"),
        pytest.param("col1\\tcol2", "col1\tcol2", id="tab"),
        pytest.param("a\\rb", "a\rb", id="carriage return"),
        # Mixed content is ambiguous (often code discussing escapes), so the
        # normalizer stays out of the way once a real newline is present.
        pytest.param(
            "real\nnewline\\nliteral", "real\nnewline\\nliteral", id="real newline"
        ),
        # An escaped backslash followed by n is an intentional literal.
        pytest.param(
            "code: \\\\n means newline",
            "code: \\\\n means newline",
            id="double backslash",
        ),
        pytest.param("plain text", "plain text", id="no escapes"),
        pytest.param("", "", id="empty"),
    ],
)
def test_normalize_escaped_whitespace(text, expected):
    assert normalize_escaped_whitespace(text) == expected


# ------------------------------------------------------------------------------
# base.py: text_formatter
# ------------------------------------------------------------------------------

# (input, expected_output, description)
TEXT_FORMATTER_CASES = [
    # Case 1: Basic paragraph separation (should convert)
    (
        "This is the first paragraph.\nThis is the second paragraph with a capital letter.",
        "This is the first paragraph.\n\nThis is the second paragraph with a capital letter.",
        "Basic paragraph test",
    ),
    # Case 2: Multiple paragraphs with different endings (should convert)
    (
        "First paragraph ends with period.\nSecond paragraph ends with exclamation!\nThird paragraph ends with question?\nFourth paragraph ends with a number 42.\nFifth paragraph.",
        "First paragraph ends with period.\n\nSecond paragraph ends with exclamation!\n\nThird paragraph ends with question?\n\nFourth paragraph ends with a number 42.\n\nFifth paragraph.",
        "Multiple paragraphs test",
    ),
    # Case 3: Paragraph ending with quotes (should convert)
    (
        'He said, "This is a quote."\nThe next paragraph begins here.',
        'He said, "This is a quote."\n\nThe next paragraph begins here.',
        "Quote ending test",
    ),
    # Case 4: Unordered list items (should not convert)
    (
        "List items:\n- Item one\n- Item two\n- Item three",
        "List items:\n\n- Item one\n- Item two\n- Item three",
        "Unordered list test",
    ),
    # Case 5: Ordered list with period format (should not convert)
    (
        "Ordered list:\n1. First item\n2. Second item\n3. Third item",
        "Ordered list:\n\n1. First item\n2. Second item\n3. Third item",
        "Ordered list period format test",
    ),
    # Case 6: Code block with indentation (should not convert)
    (
        "Python code:\ndef hello():\n    print('Hello')\n    return None",
        "Python code:\ndef hello():\n    print('Hello')\n    return None",
        "Code block test",
    ),
    # Case 7: Sentence not starting with capital (should not convert)
    (
        "This is a sentence.\nlowercase beginning shouldn't trigger a double newline.",
        "This is a sentence.\nlowercase beginning shouldn't trigger a double newline.",
        "Lowercase beginning test",
    ),
    # Case 8: Text already with double newlines (should not add more)
    (
        "This paragraph has proper formatting.\n\nThis one too.\n\nAnd this one.",
        "This paragraph has proper formatting.\n\nThis one too.\n\nAnd this one.",
        "Already formatted test",
    ),
    # Case 9: Mixed cases
    (
        "Regular paragraph.\nNew paragraph starts here.\n- List item 1\n- List item 2\nAnother paragraph after list.\ndef code():\n    return True\nFinal paragraph.",
        "Regular paragraph.\n\nNew paragraph starts here.\n\n- List item 1\n- List item 2\n\nAnother paragraph after list.\ndef code():\n    return True\n\nFinal paragraph.",
        "Mixed content test",
    ),
    # Case 10: List with capitalized items (should not convert)
    (
        "Important points:\n- The first point\n- Another critical point\n- The final consideration",
        "Important points:\n\n- The first point\n- Another critical point\n- The final consideration",
        "Capitalized list items test",
    ),
    # Case 12: Simple variable assignments (should not convert)
    (
        "Variable examples:\nmy_var = 100\ntotal = my_var + 50",
        "Variable examples:\nmy_var = 100\ntotal = my_var + 50",
        "Variable assignment test",
    ),
    # Case 13: After backticks - minimal handling
    (
        "Some code example:\n```\ndef hello():\n    print('Hello')\n```\nAnd here is more text.",
        "Some code example:\n```\ndef hello():\n    print('Hello')\n```\n\nAnd here is more text.",
        "After backticks test",
    ),
    # Case 14: Ordered list with parenthesis format (should not convert)
    (
        "Another list format:\n1) First item\n2) Second item\n3) Third item",
        "Another list format:\n\n1) First item\n2) Second item\n3) Third item",
        "Ordered list parenthesis format test",
    ),
    # Case 15: Subtle indentation (should not convert)
    (
        "Consider this code:\n var = 100\n  slightly_indented = 200",
        "Consider this code:\n var = 100\n  slightly_indented = 200",
        "Subtle indentation test",
    ),
    # Case 16: 3 or more newlines should be collapsed into 2
    (
        "First paragraph.\n\n\nSecond paragraph after triple newline.",
        "First paragraph.\n\nSecond paragraph after triple newline.",
        "Triple newline collapsed test",
    ),
    # Case 17: Mix of single, double, triple, and quad newlines
    (
        "Line one.\nLine two with single newline.\n\nLine three after double.\n\n\nLine four after triple.\n\n\n\nLine five after quad.",
        "Line one.\n\nLine two with single newline.\n\nLine three after double.\n\nLine four after triple.\n\nLine five after quad.",
        "Mixed newline sequences test",
    ),
    # Case 18: Paragraph starting with quotation marks (should convert)
    (
        'SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.\n"SmolLM2 demonstrates significant advances over its predecessor SmolLM1, particularly in instruction following, knowledge, reasoning."',
        'SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.\n\n"SmolLM2 demonstrates significant advances over its predecessor SmolLM1, particularly in instruction following, knowledge, reasoning."',
        "Paragraph starting with quotation marks test",
    ),
    # Case 19: Multiple paragraphs with quotation marks (should convert)
    (
        "First paragraph ends normally.\n\"Second paragraph starts with quotes.\" And continues.\n'Third paragraph uses single quotes.' And also continues.",
        "First paragraph ends normally.\n\n\"Second paragraph starts with quotes.\" And continues.\n\n'Third paragraph uses single quotes.' And also continues.",
        "Multiple paragraphs with quotation marks test",
    ),
    # Case 20: Paragraphs following a quote (should convert)
    (
        '"This is a test."\nBut this is also a test.',
        '"This is a test."\n\nBut this is also a test.',
        "Paragraph following a quote test",
    ),
    # Case 21: Tag squeezing - remove blank lines between tags and content
    (
        "<thinking>\n\nThis is my thought process.\nI need to analyze this problem.\n\n</thinking>",
        "<thinking>\nThis is my thought process.\n\nI need to analyze this problem.\n</thinking>",
        "Tag squeezing test",
    ),
    # Case 22: Multiple tag squeezing with different tags
    (
        "<output>\n\nThis is the final answer.\nIt should be formatted properly.\n\n</output>\n\n<step_by_step>\n\nStep 1: Identify the problem\nStep 2: Solve it\n\n</step_by_step>",
        "<output>\nThis is the final answer.\n\nIt should be formatted properly.\n</output>\n\n<step_by_step>\nStep 1: Identify the problem\nStep 2: Solve it\n</step_by_step>",
        "Multiple tag squeezing test",
    ),
    # Case 23: Nested tags should be handled correctly
    (
        "<thinking>\n\nI need to think about this.\n\n<step_by_step>\n\nStep 1: First step\nStep 2: Second step\n\n</step_by_step>\n\nNow I have my answer.\n\n</thinking>",
        "<thinking>\nI need to think about this.\n\n<step_by_step>\nStep 1: First step\nStep 2: Second step\n</step_by_step>\n\nNow I have my answer.\n</thinking>",
        "Nested tag squeezing test",
    ),
    # Case 24: Structured data with colons should not get paragraph breaks
    (
        "Feeling: Good\nEmotional Response: Satisfied\nConfidence: High",
        "Feeling: Good\nEmotional Response: Satisfied\nConfidence: High",
        "Structured data with colons test",
    ),
    # Case 25: Mixed structured data and paragraphs
    (
        "Here is some introduction text. This explains the concept.\nName: John Doe\nAge: 30\nStatus: Active\nThis is a concluding paragraph. It wraps up the discussion.",
        "Here is some introduction text. This explains the concept.\n\nName: John Doe\nAge: 30\nStatus: Active\nThis is a concluding paragraph. It wraps up the discussion.",
        "Mixed structured data and paragraphs test",
    ),
    # Case 26: Numbers and scores should not get paragraph breaks
    (
        "Final results are in. Here are the scores.\nPlayer 1: 95\nPlayer 2: 87\nGame Mode: Tournament\nThe competition was fierce! Everyone played well.",
        "Final results are in. Here are the scores.\n\nPlayer 1: 95\nPlayer 2: 87\nGame Mode: Tournament\nThe competition was fierce! Everyone played well.",
        "Numbers and scores test",
    ),
    # Case 27: Lines ending without sentence punctuation should stay together
    (
        "Configuration Settings\nDatabase Host: localhost\nPort Number: 5432\nUsername: admin\nConnection Timeout: 30\nThese settings are important for the application.",
        "Configuration Settings\nDatabase Host: localhost\nPort Number: 5432\nUsername: admin\nConnection Timeout: 30\nThese settings are important for the application.",
        "Configuration without punctuation test",
    ),
    # Case 28: Consecutive tags should be squished together
    (
        "<conscious_thought>\nI am writing a test.\n</conscious_thought>\n<step_by_step>\n(the steps)\n</step_by_step>",
        "<conscious_thought>\nI am writing a test.\n</conscious_thought>\n<step_by_step>\n(the steps)\n</step_by_step>",
        "Consecutive tags squishing test",
    ),
    # Case 29: Multiple consecutive tags should all stay together
    (
        "<thinking>\nFirst thought\n</thinking>\n<analysis>\nAnalysis here\n</analysis>\n<conclusion>\nFinal thought\n</conclusion>",
        "<thinking>\nFirst thought\n</thinking>\n<analysis>\nAnalysis here\n</analysis>\n<conclusion>\nFinal thought\n</conclusion>",
        "Multiple consecutive tags test",
    ),
    # Case 30: Tags followed by regular content should get proper spacing
    (
        "<conscious_thought>\nI am thinking.\n</conscious_thought>\n<step_by_step>\nStep 1: Do this\n</step_by_step>\nThis is regular content after tags.",
        "<conscious_thought>\nI am thinking.\n</conscious_thought>\n<step_by_step>\nStep 1: Do this\n</step_by_step>\n\nThis is regular content after tags.",
        "Tags followed by content test",
    ),
    # Case 31: Mixed tags and structured data
    (
        "<analysis>\nFeeling: Good\nConfidence: High\n</analysis>\n<next_steps>\nAction: Review data\nDeadline: Tomorrow\n</next_steps>\nThe analysis is complete. We can proceed.",
        "<analysis>\nFeeling: Good\nConfidence: High\n</analysis>\n<next_steps>\nAction: Review data\nDeadline: Tomorrow\n</next_steps>\n\nThe analysis is complete. We can proceed.",
        "Mixed tags and structured data test",
    ),
    # Cases 32-37: typographic quotes (U+201C/D, U+2018/9), escaped so an editor
    # cannot silently normalize them to ASCII.
    (
        "This is a regular sentence.\n\u201cThis starts with a curly left double quote.\u201d",
        "This is a regular sentence.\n\n\u201cThis starts with a curly left double quote.\u201d",
        "Curly left double quote after sentence test",
    ),
    (
        "This is a regular sentence.\n\u2018This starts with a curly left single quote.\u2019",
        "This is a regular sentence.\n\n\u2018This starts with a curly left single quote.\u2019",
        "Curly left single quote after sentence test",
    ),
    (
        "He said \u201cthis is with curly quotes.\u201d\nNext paragraph should have double newline.",
        "He said \u201cthis is with curly quotes.\u201d\n\nNext paragraph should have double newline.",
        "Curly right double quote at end test",
    ),
    (
        "He said \u2018this is with curly single quotes.\u2019\nNext paragraph here.",
        "He said \u2018this is with curly single quotes.\u2019\n\nNext paragraph here.",
        "Curly right single quote at end test",
    ),
    (
        "First with straight quotes.\n"
        "\u201cSecond with curly quotes.\u201d\n"
        "'Third with straight single.'\n"
        "\u2018Fourth with curly single.\u2019",
        "First with straight quotes.\n\n"
        "\u201cSecond with curly quotes.\u201d\n\n"
        "'Third with straight single.'\n\n"
        "\u2018Fourth with curly single.\u2019",
        "Mixed quote types test",
    ),
    (
        "```\ncode block\n```\n\u201cText starting with curly quote after code.\u201d",
        "```\ncode block\n```\n\n\u201cText starting with curly quote after code.\u201d",
        "Curly quote after code block test",
    ),
    # Case 38: Quote then closing parenthesis
    (
        'End of quote.")\nNext paragraph starts here.',
        'End of quote.")\n\nNext paragraph starts here.',
        "Quote with closing parenthesis test",
    ),
    # Case 39: Paragraphs ending with colon should get double newlines
    (
        "Here are the main points:\nThis is the first point about something important.",
        "Here are the main points:\n\nThis is the first point about something important.",
        "Paragraph ending with colon test",
    ),
    # Case 40: Multiple paragraphs ending with colons
    (
        "Introduction to the topic:\nThis covers basic concepts and ideas.\nAdvanced techniques:\nThese require more experience and knowledge.",
        "Introduction to the topic:\n\nThis covers basic concepts and ideas.\n\nAdvanced techniques:\n\nThese require more experience and knowledge.",
        "Multiple paragraphs ending with colons test",
    ),
    # Case 41: Paragraph ending with colon followed by structured data (should be treated differently)
    (
        "The configuration values are:\nHost: localhost\nPort: 8080\nThis completes the setup process.",
        "The configuration values are:\nHost: localhost\nPort: 8080\nThis completes the setup process.",
        "Colon followed by structured data test",
    ),
    # Case 42: Question ending with colon (narrative context)
    (
        "The question we need to ask ourselves is this:\nHow can we improve the system for everyone?",
        "The question we need to ask ourselves is this:\n\nHow can we improve the system for everyone?",
        "Question ending with colon test",
    ),
    # Case 43: Statement ending with colon followed by explanation
    (
        "The reason is simple:\nWe need better communication between teams.",
        "The reason is simple:\n\nWe need better communication between teams.",
        "Statement ending with colon followed by explanation test",
    ),
    # Case 44: Colon with quotes around it
    (
        'She said: "This is important."\nThe next paragraph continues the thought.',
        'She said: "This is important."\n\nThe next paragraph continues the thought.',
        "Colon with quotes test",
    ),
    # Case 45: Header/title without punctuation followed by paragraph
    (
        "Diagnosis of HIVE infection\nHIVE infection is often associated with kidney disease.",
        "Diagnosis of HIVE infection\n\nHIVE infection is often associated with kidney disease.",
        "Header without punctuation test",
    ),
    # Case 46: Multiple headers without punctuation
    (
        "First Section Title\nThis is content for the first section.\nSecond Section Title\nThis is content for the second section.",
        "First Section Title\n\nThis is content for the first section.\n\nSecond Section Title\n\nThis is content for the second section.",
        "Multiple headers without punctuation test",
    ),
    # Case 47: List items followed by paragraph
    (
        "Common types of HIVE infection include:\n- Type 1 diabetes\n- Alcohol consumption\n- Variations in blood sugar\nYou can also be treated with medicine based on your child's health history.",
        "Common types of HIVE infection include:\n\n- Type 1 diabetes\n- Alcohol consumption\n- Variations in blood sugar\n\nYou can also be treated with medicine based on your child's health history.",
        "List items followed by paragraph test",
    ),
    # Case 48: Ordered list followed by paragraph
    (
        "Steps to follow:\n1. First step\n2. Second step\n3. Third step\nThese steps are important for success.",
        "Steps to follow:\n\n1. First step\n2. Second step\n3. Third step\n\nThese steps are important for success.",
        "Ordered list followed by paragraph test",
    ),
    # Case 49: Quoted paragraph after list items
    (
        'Key points:\n- First point\n- Second point\n"This is a quoted statement after the list."',
        'Key points:\n\n- First point\n- Second point\n\n"This is a quoted statement after the list."',
        "Quote after list items test",
    ),
    # Case 50: Header ending with lowercase letter followed by a quoted paragraph
    (
        'What Are HIVE Early Childhood\n"Common types of HIVE infection include the following."',
        'What Are HIVE Early Childhood\n\n"Common types of HIVE infection include the following."',
        "Header with lowercase ending followed by quote test",
    ),
    # Escaped whitespace is normalized before paragraph detection runs.
    (
        "First paragraph.\\n\\nSecond paragraph.",
        "First paragraph.\n\nSecond paragraph.",
        "Escaped newlines normalized first test",
    ),
]


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(text, expected, id=desc)
        for text, expected, desc in TEXT_FORMATTER_CASES
    ],
)
def test_text_formatter(text, expected):
    assert text_formatter(text) == expected


# ------------------------------------------------------------------------------
# conversation.py
# ------------------------------------------------------------------------------


HH_TRANSCRIPT = (
    "\n\nHuman: What are some good ways to stay focused?"
    "\n\nAssistant: A few that work well:\n\nShort sessions, regular breaks..."
    "\n\nHuman: Which one matters most?"
    "\n\nAssistant: Consistency beats any single trick."
)


def test_human_assistant_parses_turns_and_roles():
    out = format_human_assistant({"chosen": HH_TRANSCRIPT}, ["chosen"], tokenizer=None)
    msgs = out["messages"]
    convo = [m for m in msgs if m["role"] in ("user", "assistant")]
    assert [m["role"] for m in convo] == ["user", "assistant", "user", "assistant"]
    assert convo[0]["content"].startswith("What are some good ways")
    # Embedded blank lines inside a turn must stay within that turn.
    assert "Short sessions" in convo[1]["content"]
    # Unified system + developer prompts come from the messages pipeline.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "developer"


def test_human_assistant_empty_and_garbage_inputs():
    assert format_human_assistant({"chosen": ""}, ["chosen"], None)["messages"] == []
    assert format_human_assistant({}, ["chosen"], None)["messages"] == []
    out = format_human_assistant({"chosen": "no markers here"}, ["chosen"], None)
    assert out["messages"] == []


def test_preference_pair_emits_one_side_per_call(seeded_random):
    """Each call returns a single side, picked 50/50, so a pair's two halves are
    never co-resident in a batch (see next/rl.md section 4.2). This pins that
    contract, so a switch to paired emission shows up here."""
    chosen, rejected = int(TaskType.PREF_CHOSEN), int(TaskType.PREF_REJECTED)
    doc = {
        "chosen": "\n\nHuman: Hi there\n\nAssistant: Good answer",
        "rejected": "\n\nHuman: Hi there\n\nAssistant: Bad answer",
    }
    seen = set()
    for _ in range(40):
        out = format_preference_pair(doc, ["chosen", "rejected"], tokenizer=None)
        assert out["messages"], "pair side must parse to messages"
        tag = out["metadata"]["task_type"]
        seen.add(tag)
        text = out["messages"][-1]["content"]
        if tag == chosen:
            assert "Good" in text
        else:
            assert tag == rejected and "Bad" in text
    assert seen == {chosen, rejected}


# ------------------------------------------------------------------------------
# joke.py
# ------------------------------------------------------------------------------


def test_format_joke_quality_filters():
    keys = ["jokeText", "rating"]
    good = format_joke({"jokeText": "knock knock", "rating": 5.0}, keys, None)
    assert [m["role"] for m in good["messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
    ]
    # The assistant turn ends with the normalized want->need score on its own
    # line - the dense grounding for the calibration loop mode.
    assert good["messages"][-1]["content"] == "knock knock\n+0.5"
    # Below-median (disliked) jokes are skipped (empty -> sampler retries).
    assert (
        format_joke({"jokeText": "bad", "rating": -2.0}, keys, None)["messages"] == []
    )


# ------------------------------------------------------------------------------
# tools.py
# ------------------------------------------------------------------------------


def test_tool_training_data_matches_the_format(
    prose_tokenizer, default_tokenizer, seeded_random
):
    """The formatter asks the chat format for the layout, not just rendering.

    prose puts the call and result in their own roles with no markers; default
    keeps them inside assistant/tool messages as atomic tool-boundary tokens.
    """
    prose_doc = format_tool_calling({}, [], prose_tokenizer)
    roles = [m["role"] for m in prose_doc["messages"]]
    assert "call" in roles and "tool" in roles
    call_msg = next(m for m in prose_doc["messages"] if m["role"] == "call")
    assert TOOL_CALL_OPEN not in call_msg["content"]
    tool_msg = next(m for m in prose_doc["messages"] if m["role"] == "tool")
    assert TOOL_RESULT_OPEN not in tool_msg["content"]

    tok = default_tokenizer
    default_doc = format_tool_calling({}, [], tok)
    assert "call" not in [m["role"] for m in default_doc["messages"]]
    call_msg = next(
        m
        for m in default_doc["messages"]
        if m["role"] == "assistant" and TOOL_CALL_OPEN in m["content"]
    )
    result_msg = next(m for m in default_doc["messages"] if m["role"] == "tool")
    assert TOOL_CALL_CLOSE in call_msg["content"]
    assert TOOL_RESULT_OPEN in result_msg["content"]
    assert TOOL_RESULT_CLOSE in result_msg["content"]
    # Each boundary encodes to its single atomic id, never to its characters.
    call_ids = tok.encode(call_msg["content"], add_special_tokens=False)
    result_ids = tok.encode(result_msg["content"], add_special_tokens=False)
    assert tok.tool_call_token_id in call_ids
    assert tok.tool_call_end_token_id in call_ids
    assert tok.tool_result_token_id in result_ids
    assert tok.tool_result_end_token_id in result_ids


def test_prose_call_turn_and_its_boundaries_are_supervised(prose_tokenizer):
    """A tool call is model-produced, so the call body and both boundaries
    around it must be trained targets.

    Every boundary is the tail of the turn before it, and a tail is supervised
    only when that turn is generated. Running `user` straight into `call` would
    put `\\n\\ncall\\n\\n` in the user turn's tail, so the model could continue a
    call it was handed but never decide to open one. The empty `assistant` turn
    is what moves that boundary inside a generated span.
    """
    doc = format_tool_calling({}, [], prose_tokenizer)
    ids, mask = tokenize_with_mask(prose_tokenizer, doc["messages"])
    trained = "".join(prose_tokenizer.decode([t]) for t, m in zip(ids, mask) if m)
    assert '"name": "calc"' in trained
    assert "\n\ncall\n\n" in trained  # opening the call
    assert "\n\ntool\n\n" in trained  # handing off to the tool


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
