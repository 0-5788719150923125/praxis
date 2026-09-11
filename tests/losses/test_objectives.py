"""Every loss a run carries is registered in one container.

The model used to declare its objectives in three different ways: a
``criterion`` module, a ``reg`` list, and bare ``F.cross_entropy`` calls
inside the heads and the MTP stack that appeared in neither. These tests pin
the collapsed arrangement - one container, one entry per term, each owned
exactly once.
"""

import copy

import pytest
import torch

from praxis.losses import Objectives
from praxis.losses.contrastive_isotropy import ContrastiveIsotropyLoss
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.halo import HALOLoss


def _terms():
    o = Objectives()
    o.register("main", HALOLoss(vocab_size=32))
    o.register("contrastive", ContrastiveIsotropyLoss())
    o.register("mtp", CrossEntropyLoss())
    return o


def test_terms_register_in_order_under_the_tag_they_report():
    o = _terms()
    assert [name for name, _ in o.terms()] == ["main", "contrastive", "mtp"]
    text = repr(o)
    assert "(main): HALOLoss()" in text
    assert "(mtp): CrossEntropyLoss()" in text


def test_main_is_none_when_an_encoder_owns_the_loss():
    o = Objectives()
    o.register("contrastive", ContrastiveIsotropyLoss())
    assert o.main is None
    assert o.regularizers()


def test_a_name_cannot_be_claimed_twice_or_shadow_the_api():
    o = _terms()
    with pytest.raises(KeyError):
        o.register("main", CrossEntropyLoss())
    with pytest.raises(KeyError):
        o.register("register", CrossEntropyLoss())
    with pytest.raises(TypeError):
        o.register("callable", lambda *a: 0)


def test_require_fails_loudly_rather_than_falling_back():
    """A producer whose term never got registered must not quietly compute its
    own - that is the arrangement this container exists to end."""
    o = _terms()
    assert o.require("mtp") is o.get("mtp")
    with pytest.raises(KeyError):
        o.require("arm_ce")


def test_regularizers_are_the_terms_that_shape_the_representation():
    o = _terms()
    assert [type(r).__name__ for r in o.regularizers()] == ["ContrastiveIsotropyLoss"]
    o.reset()  # the unconditional per-forward drop; must reach only those


def test_metrics_and_descriptions_come_from_every_term():
    """The dashboard used to read the criterion and the regularizers through
    two separate extractors; one container answers for both."""
    o = _terms()
    keys = set()
    for descs in o.metric_descriptions():
        keys.update(descs)
    assert "contrastive_loss" in keys
    assert "repr_anisotropy" in keys


# ── the model wiring ───────────────────────────────────────────────────────


def _model(**overrides):
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    cfg = dict(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=1,
        depth=2,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        head_type="prismatic5",
        residual_type="smear",
        byte_level=True,
        loss_func="halo",
        mtp_type="per_depth",
        mtp_depth=2,
    )
    cfg.update(overrides)
    torch.manual_seed(0)
    return PraxisForCausalLM(PraxisConfig(**cfg))


def test_the_model_declares_every_term_in_one_place():
    m = _model()
    names = [name for name, _ in m.criterion.terms()]
    assert names[0] == "main"
    # The terms other modules compute are declared here too, not inline.
    assert "mtp" in names and "arm_ce" in names
    assert "contrastive" in names


def test_a_producer_reads_its_term_back_rather_than_owning_it():
    """Held off the producer's module tree, so nothing is printed - or
    checkpointed - twice."""
    m = _model()
    assert m.mtp.criterion is m.criterion.mtp
    assert sum(1 for mod in m.modules() if mod is m.criterion.mtp) == 1
    assert "CrossEntropyLoss" not in repr(m.mtp)
    # And a copy of the model keeps the two sides pointing at each other.
    clone = copy.deepcopy(m)
    assert clone.mtp.criterion is clone.criterion.mtp


def test_a_pre_container_checkpoint_still_resumes():
    """``criterion.*`` for the main term, ``reg.<index>.*`` for the
    regularizers. Only terms with parameters of their own ever wrote a key."""
    m = _model(regularizers=["contrastive_isotropy"])
    legacy = {}
    for key, value in m.state_dict().items():
        if key.startswith("criterion.main."):
            legacy["criterion." + key[len("criterion.main.") :]] = value
        else:
            legacy[key] = value
    assert "criterion.gamma" in legacy
    fresh = _model(regularizers=["contrastive_isotropy"])
    with torch.no_grad():
        fresh.criterion.main.gamma.fill_(0.0)
    missing, unexpected = fresh.load_state_dict(legacy, strict=False)
    assert not [k for k in missing if k.startswith("criterion")]
    assert not [k for k in unexpected if k.startswith(("criterion", "reg."))]
    assert torch.equal(fresh.criterion.main.gamma, m.criterion.main.gamma)
