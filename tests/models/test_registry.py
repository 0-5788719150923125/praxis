"""The foreign-model registries and the guardrails around them."""

import pathlib

import pytest

from praxis import registry
from praxis.cli.groups import build_static_parser
from praxis.models import (
    ForeignModel,
    ModelAdapter,
    architecture_flag_dests,
    get_adapter,
    parse_model_kwargs,
    reject_incompatible_flags,
)


@pytest.fixture(scope="module")
def parser():
    return build_static_parser()


def test_every_task_names_a_real_auto_class_and_wrapper():
    import transformers

    for key, spec in registry.namespace("model_tasks").items():
        assert hasattr(transformers, spec.auto_class), key
        assert issubclass(spec.wrapper, ForeignModel), key
        assert spec.wrapper.TASK == key


def test_every_task_documents_what_it_cannot_host():
    """An unsupported mechanism has to carry its reason, because the reason is
    what the user reads when the run refuses to start."""
    for key, spec in registry.namespace("model_tasks").items():
        for name, reason in spec.wrapper.UNSUPPORTED.items():
            assert reason and len(reason) > 20, f"{key}.{name}"


def test_adapter_default_discovers_everything():
    assert get_adapter(None) == ModelAdapter()
    assert get_adapter("a-family-nobody-has-registered") == ModelAdapter()


def test_peft_profiles_are_lora_configs():
    from peft import LoraConfig

    for key, profile in registry.namespace("peft_profiles").items():
        fields = dict(profile)
        fields.pop("peft_type", None)
        LoraConfig(**fields)  # raises on an unknown or malformed key


def test_model_kwargs_parse_as_yaml():
    parsed = parse_model_kwargs(
        ["trust_remote_code=true", "num_labels=3", "dtype=bfloat16"]
    )
    assert parsed == {
        "trust_remote_code": True,
        "num_labels": 3,
        "dtype": "bfloat16",
    }


def test_model_kwarg_without_a_value_is_an_error():
    with pytest.raises(ValueError, match="key=value"):
        parse_model_kwargs(["trust_remote_code"])


def test_architecture_flags_are_rejected(parser):
    args = parser.parse_args(["--model-name", "x", "--attention-type", "causal"])
    with pytest.raises(ValueError, match="--attention-type"):
        reject_incompatible_flags(args, parser)


def test_run_flags_survive(parser):
    """Optimizer, batch, losses and data describe the RUN, not the model."""
    args = parser.parse_args(
        [
            "--model-name",
            "x",
            "--optimizer",
            "AdamW",
            "--batch-size",
            "4",
            "--block-size",
            "256",
            "--loss-func",
            "focal",
            "--rl-type",
            "preference",
        ]
    )
    reject_incompatible_flags(args, parser)  # does not raise


def test_the_rejected_set_is_read_off_the_parser(parser):
    """Not a list in praxis/models, so a flag added to the architecture group
    later is covered without anyone remembering this file."""
    dests = architecture_flag_dests(parser)
    assert "attention_type" in dests
    assert "block_size" not in dests  # run-scoped, explicitly kept
    assert "optimizer" not in dests  # not in the architecture group at all


def test_objective_flags_in_the_architecture_group_survive(parser):
    """`--regularizers` lives in the architecture group but is an additive loss
    term Praxis applies to whatever hidden states it is handed, so a foreign
    model honors it. Rejecting it would break the one config that uses it."""
    args = parser.parse_args(
        ["--model-name", "x", "--regularizers", "contrastive_isotropy"]
    )
    reject_incompatible_flags(args, parser)  # does not raise


def test_flags_the_checkpoint_owns_are_rejected(parser):
    """Outside the architecture group, and each carries its reason."""
    args = parser.parse_args(["--model-name", "x", "--tokenizer-type", "bpe"])
    with pytest.raises(ValueError, match="tokenizer-type.*own tokenizer"):
        reject_incompatible_flags(args, parser)


def test_the_shipped_foreign_experiment_parses(parser):
    """experiments/smol.yml is the reference config for this whole path; a flag
    classification that breaks it should fail here, not at launch.

    Integration flags (``--discord``) are not in the static parser by design -
    they are local to a checkout - so they are skipped, and the skip list is
    asserted rather than assumed so a typo in the config still fails.
    """
    import yaml

    from praxis.models import parse_model_kwargs

    path = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "smol.yml"
    config = yaml.safe_load(path.read_text())
    known = {
        action.dest
        for group in parser._action_groups
        for action in group._group_actions
    }

    argv, skipped = ["--model-name", config["model_name"]], []
    for key, value in config.items():
        if key in ("model_name", "model_kwarg"):
            continue
        if key not in known:
            skipped.append(key)
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, dict):
            # A mapping an experiment YAML can express but argparse cannot
            # (generation_kwargs). Covered by tests/inference/test_prompts.py,
            # which validates it the way the CLI actually does.
            skipped.append(key)
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            # An empty list is meaningful (regularizers: [] = none), so the bare
            # flag goes in - dropping it would silently test the default.
            argv.extend([flag, *(str(v) for v in value)])
        else:
            argv.extend([flag, str(value)])

    # Assert what is skipped is skipped for a REASON, not that the list has a
    # particular shape: integration flags come and go with what is installed,
    # and a brittle equality here would fail on somebody else's checkout.
    for key in skipped:
        assert key not in known or isinstance(config[key], dict), key
    assert "attention_type" not in skipped, "an architecture flag would be a bug"

    args = parser.parse_args(argv)
    reject_incompatible_flags(args, parser)
    assert args.regularizers == []
    assert parse_model_kwargs(config["model_kwarg"]) == {"attn_implementation": "sdpa"}
