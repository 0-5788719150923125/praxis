"""Standing instructions and decode knobs: who wins, and where a prompt lands.

The rules under test are both "last writer wins" rules, and both exist because
the alternative fails silently: an injected duplicate developer message that
contradicts the caller's, and a typo'd generation kwarg that transformers
accepts and drops.
"""

import pytest

from praxis.inference import apply_standing_prompts, parse_generation_kwargs
from praxis.tokenizers.chat_templates import (
    DEFAULT_FORMAT,
    HF_NATIVE_FORMAT,
    PROSE_FORMAT,
)

USER = [{"role": "user", "content": "hi"}]


def _roles(messages):
    return [m["role"] for m in messages]


def _content(messages, role):
    return next(m["content"] for m in messages if m["role"] == role)


# ---------------------------------------------------------------------------
# standing prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [DEFAULT_FORMAT, PROSE_FORMAT], ids=lambda f: f.name)
def test_both_roles_are_prepended_when_the_format_has_them(fmt):
    out = apply_standing_prompts(USER, "SYS", "DEV", fmt)
    assert _roles(out) == ["system", "developer", "user"]


def test_a_format_without_a_developer_role_folds_it_into_system():
    """`hf_native` renders a published checkpoint's own template. Emitting a
    `developer` turn there would put a role name in the prompt that the model
    has never been trained on."""
    out = apply_standing_prompts(USER, "SYS", "DEV", HF_NATIVE_FORMAT)
    assert _roles(out) == ["system", "user"]
    assert _content(out, "system") == "SYS\n\nDEV"


def test_folding_still_happens_with_no_system_prompt():
    out = apply_standing_prompts(USER, None, "DEV", HF_NATIVE_FORMAT)
    assert _content(out, "system") == "DEV"


@pytest.mark.parametrize("role", ["system", "developer"])
def test_the_callers_own_message_wins(role):
    """What makes the web app's editable prompt an override, not a duplicate."""
    out = apply_standing_prompts(
        [{"role": role, "content": "MINE"}] + USER, "SYS", "DEV", DEFAULT_FORMAT
    )
    assert _roles(out).count(role) == 1
    assert _content(out, role) == "MINE"


def test_nothing_is_added_when_the_run_sets_nothing():
    assert apply_standing_prompts(USER, None, None, DEFAULT_FORMAT) == USER


def test_blank_prompts_are_not_messages():
    assert apply_standing_prompts(USER, "   ", "\n", DEFAULT_FORMAT) == USER


def test_the_caller_list_is_not_mutated():
    messages = list(USER)
    apply_standing_prompts(messages, "SYS", "DEV", DEFAULT_FORMAT)
    assert messages == USER


def test_an_empty_conversation_stays_empty():
    """Nothing to answer means nothing to prompt; a lone preamble would be a
    turn with no question."""
    assert apply_standing_prompts([], "SYS", "DEV", DEFAULT_FORMAT) == []


# ---------------------------------------------------------------------------
# generation kwargs
# ---------------------------------------------------------------------------


def test_both_spellings_reach_the_same_place():
    """A mapping is what an experiment YAML can express; a list of `key=value`
    is what argparse can. They must not mean different things."""
    from_yaml = parse_generation_kwargs({"temperature": 0.7, "max_new_tokens": 96})
    from_cli = parse_generation_kwargs(["temperature=0.7", "max_new_tokens=96"])
    assert from_yaml == from_cli == {"temperature": 0.7, "max_new_tokens": 96}


def test_values_arrive_typed():
    parsed = parse_generation_kwargs(["do_sample=false", "top_k=40", "min_p=0.05"])
    assert parsed == {"do_sample": False, "top_k": 40, "min_p": 0.05}


def test_an_unknown_key_is_an_error():
    """transformers accepts unknown kwargs on generate and drops them, so a
    typo would otherwise decode at the default forever."""
    with pytest.raises(ValueError, match="temperture"):
        parse_generation_kwargs(["temperture=0.2"])


def test_praxis_side_keys_are_accepted():
    """Not GenerationConfig fields, but the generator understands them."""
    parsed = parse_generation_kwargs(
        ["use_cache=false", "timeout=30", "truncate_to=512", "skip_special_tokens=true"]
    )
    assert len(parsed) == 4


def test_empty_is_empty():
    assert parse_generation_kwargs(None) == {}
    assert parse_generation_kwargs([]) == {}


def test_the_shipped_experiment_validates():
    """experiments/smol.yml is the reference config for this feature."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "smol.yml"
    config = yaml.safe_load(path.read_text())
    assert parse_generation_kwargs(config["generation_kwargs"])
    assert config["system_prompt"] and config["developer_prompt"]


def test_length_penalty_under_sampling_is_called_out(capsys):
    """It is the obvious-looking fix for a runaway reply and it does nothing:
    transformers only applies it when ranking finished beams. Measured on
    SmolLM2 - identical median and max length with and without it, and
    transformers itself logs the flag as ignored."""
    parse_generation_kwargs(["length_penalty=2.0"])
    assert "nothing under sampling" in capsys.readouterr().out


def test_length_penalty_is_quiet_when_beams_are_on(capsys):
    """Then it is a real knob, and saying otherwise would be the noise."""
    parse_generation_kwargs(["length_penalty=2.0", "num_beams=4"])
    assert capsys.readouterr().out == ""


def test_the_decay_penalty_survives_the_web_forms_round_trip():
    """The Settings box renders each kwarg as one `key=value` line, so a tuple
    value has to come back as a sequence rather than a string."""
    parsed = parse_generation_kwargs(["exponential_decay_length_penalty=[64, 1.03]"])
    assert list(parsed["exponential_decay_length_penalty"]) == [64, 1.03]
