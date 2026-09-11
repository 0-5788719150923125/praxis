"""Tests for praxis/migrations.py: pre-rename configs and checkpoints still load."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.cli.loaders.experiments import load_rendered_config
from praxis.modeling import PraxisForCausalLM
from praxis.migrations import rename_legacy_config, rename_legacy_key


def _config(classifier_type):
    return PraxisConfig(
        vocab_size=64,
        hidden_size=16,
        embed_size=16,
        num_heads=2,
        depth=2,
        max_length=32,
        decoder_type="sequential",
        encoder_type=None,
        classifier_type=classifier_type,
    )


@pytest.mark.parametrize(
    "old, new",
    [
        ("model.head.lm_head.weight", "model.classifier.scorer.weight"),
        ("head.heads.1.lm_head.centers", "classifier.stages.1.scorer.centers"),
        (
            "model.head.branches.0.heads.0.field.amplitudes",
            "model.classifier.branches.0.stages.0.field.amplitudes",
        ),
        ("backward_head.lm_head.weight", "backward_classifier.scorer.weight"),
        (
            "encoder.energy_head.final_layer.weight",
            "encoder.generator.final_layer.weight",
        ),
        ("mtp.patch_head.weight", "mtp.patch_projection.weight"),
        # Attention's own `heads` is not inside a classifier: untouched.
        ("decoder.locals.0.attn.heads.weight", "decoder.locals.0.attn.heads.weight"),
        ("model.classifier.scorer.weight", "model.classifier.scorer.weight"),
    ],
)
def test_legacy_keys_rename(old, new):
    assert rename_legacy_key(old) == new


def test_legacy_config_key_reaches_the_new_field():
    """An old config.json's head_type must select the classifier, not land in
    kwargs where nothing reads it."""
    assert PraxisConfig(head_type="crystal").classifier_type == "crystal"
    assert rename_legacy_config({"head_type": "tied"}) == {"classifier_type": "tied"}
    # The new name wins when both are present.
    both = {"head_type": "tied", "classifier_type": "crystal"}
    assert rename_legacy_config(both) == {"classifier_type": "crystal"}


def test_experiment_yaml_with_the_old_key(tmp_path):
    (tmp_path / "base.yml").write_text("head_type: prismatic8\n")
    (tmp_path / "child.yml").write_text("extends: base\nhidden_size: 32\n")
    config = load_rendered_config(tmp_path / "child.yml")
    assert config == {"classifier_type": "prismatic8", "hidden_size": 32}


@pytest.mark.parametrize(
    "classifier_type", ["forward", "crystal_harmonic", "prismatic8"]
)
def test_pre_rename_checkpoint_loads(classifier_type):
    """A state dict saved under the old paths loads strictly into the new model
    and reproduces its logits."""
    torch.manual_seed(0)
    source = PraxisForCausalLM(_config(classifier_type)).eval()
    new_to_old = {
        "classifier": "head",
        "backward_classifier": "backward_head",
        "scorer": "lm_head",
        "stages": "heads",
    }
    legacy = {
        ".".join(new_to_old.get(s, s) for s in key.split(".")): value
        for key, value in source.state_dict().items()
    }
    assert any(k.startswith("head.") for k in legacy)

    torch.manual_seed(1)
    target = PraxisForCausalLM(_config(classifier_type)).eval()
    target.load_state_dict(legacy, strict=True)

    ids = torch.arange(12).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(source(input_ids=ids).logits, target(input_ids=ids).logits)
