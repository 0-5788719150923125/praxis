"""Teaching a published chat template to report its assistant spans.

The transform has one hard requirement: the rendered text must not change. A
template that drifted would train the model on a format it has never seen, and
the drift would be invisible - which is why every test here checks the rendering
as well as the mask.
"""

import pytest

from praxis.tokenizers.generation_spans import (
    GenerationSpanError,
    has_generation_spans,
    rewrite_template,
)

# The ChatML shape almost every published instruct model ships, verbatim from
# HuggingFaceTB/SmolLM2-135M-Instruct minus its system-prompt injection.
CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

# A template that branches on the role itself, and uses attribute access and a
# differently-named loop variable.
BRANCHED = (
    "{% for turn in loop_messages %}"
    "{% if turn.role == 'assistant' %}A: {{ turn.content }}\n"
    "{% else %}U: {{ turn.content }}\n{% endif %}"
    "{% endfor %}"
)

MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
]


def _render(template, **kwargs):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {}
    # The markers render nothing; Jinja does not know them, so they are stripped
    # before rendering exactly as HuggingFace's extension would emit them.
    body = template.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    return env.from_string(body).render(**kwargs)


@pytest.mark.parametrize(
    "template,variables",
    [
        (CHATML, dict(messages=MESSAGES, add_generation_prompt=False)),
        (BRANCHED, dict(loop_messages=MESSAGES)),
    ],
    ids=["chatml", "branched"],
)
def test_rewrite_preserves_the_rendered_text(template, variables):
    rewritten = rewrite_template(template, ["assistant"])
    assert has_generation_spans(rewritten)
    assert _render(rewritten, **variables) == _render(template, **variables)


def test_rewrite_wraps_the_loop_body_in_a_role_test():
    rewritten = rewrite_template(CHATML, ["assistant"])
    assert "{% if message['role'] == 'assistant' %}{% generation %}" in rewritten


def test_rewrite_uses_the_templates_own_loop_variable():
    rewritten = rewrite_template(BRANCHED, ["assistant"])
    assert "turn['role'] == 'assistant'" in rewritten


def test_multiple_generated_roles_are_all_marked():
    rewritten = rewrite_template(CHATML, ["assistant", "call"])
    assert "message['role'] == 'assistant' or message['role'] == 'call'" in rewritten


def test_an_already_marked_template_is_left_alone():
    assert has_generation_spans("{% generation %}x{% endgeneration %}")


def test_a_template_with_no_message_loop_is_an_error():
    with pytest.raises(GenerationSpanError, match="no message loop"):
        rewrite_template("{{ bos_token }}hello", ["assistant"])


def test_a_loop_that_is_not_over_messages_is_not_matched():
    """A tools loop is a for-loop too; only one that reads a role is a chat
    loop."""
    with pytest.raises(GenerationSpanError, match="no message loop"):
        rewrite_template(
            "{% for tool in tools %}{{ tool.name }}{% endfor %}", ["assistant"]
        )


def test_nested_loops_do_not_confuse_the_matcher():
    template = (
        "{% for message in messages %}"
        "{{ message['role'] }}"
        "{% for part in message['content'] %}{{ part }}{% endfor %}"
        "{% endfor %}"
    )
    rewritten = rewrite_template(template, ["assistant"])
    # The wrap closes around the WHOLE body, inner loop included.
    assert rewritten.count("{% endfor %}") == 3
    assert "{% endgeneration %}{% else %}" in rewritten


@pytest.mark.network
def test_smollm2_instruct_reports_only_its_assistant_turns():
    """The proof-of-concept checkpoint, end to end."""
    from transformers import AutoTokenizer

    from praxis.tokenizers.generation_spans import add_generation_spans

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    template = add_generation_spans(tokenizer)

    encoded = tokenizer.apply_chat_template(
        MESSAGES,
        chat_template=template,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    kept = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert kept.startswith("<|im_start|>assistant")
    assert "answer" in kept
    assert "question" not in kept
    # The turn's own terminator is inside the span, so halting is a trained
    # target rather than something the generator has to guess.
    assert "<|im_end|>" in kept
