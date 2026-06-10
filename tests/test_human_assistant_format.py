from praxis.data.formatters.conversation import format_human_assistant

TRANSCRIPT = (
    "\n\nHuman: What are some good ways to stay focused?"
    "\n\nAssistant: A few that work well:\n\nShort sessions, regular breaks..."
    "\n\nHuman: Which one matters most?"
    "\n\nAssistant: Consistency beats any single trick."
)


def test_parses_turns_and_roles():
    out = format_human_assistant({"chosen": TRANSCRIPT}, ["chosen"], tokenizer=None)
    msgs = out["messages"]
    convo = [m for m in msgs if m["role"] in ("user", "assistant")]
    assert [m["role"] for m in convo] == ["user", "assistant", "user", "assistant"]
    assert convo[0]["content"].startswith("What are some good ways")
    # Embedded blank lines inside a turn must stay within that turn.
    assert "Short sessions" in convo[1]["content"]
    # Unified system + developer prompts come from the messages pipeline.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "developer"


def test_empty_and_garbage_inputs():
    assert format_human_assistant({"chosen": ""}, ["chosen"], None)["messages"] == []
    assert format_human_assistant({}, ["chosen"], None)["messages"] == []
    out = format_human_assistant({"chosen": "no markers here"}, ["chosen"], None)
    assert out["messages"] == []
