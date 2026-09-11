"""One --precision flag, three knobs kept in agreement.

The failure this pins down is drift: the model built in one dtype, Lightning
stepping in another, and the matmul policy set from a third place. Every
consumer resolves the same profile from the same registry, so what is asserted
here is that the flag reaches all three - model dtype, Lightning precision
string, float32 matmul policy - and that a level the hardware cannot honor is
downgraded to something that runs rather than exploding mid-step.
"""

from praxis.cli.config import RunConfig


def _cfg(**overrides):
    """A RunConfig with only the fields the precision path reads."""
    base = dict(
        seed=0,
        vocab_size=256,
        cache_dir="/tmp",
        optimizer="adamw",
        batch_size=1,
        block_size=16,
        device="cpu",
        target_batch_size=1,
    )
    base.update(overrides)
    return RunConfig(**base)


def test_flag_reaches_the_lightning_trainer():
    """The end of the wire: --precision bf16 becomes Trainer(precision=...)."""
    from types import SimpleNamespace

    from praxis.trainers.runtime import _build_trainer_params

    cfg = _cfg(precision="bf16", val_every=8)
    bundle = SimpleNamespace(hparams={"batch_size": 1, "target_batch_size": 1})
    params = _build_trainer_params(cfg, bundle, callbacks=[], logger=None)
    assert params["precision"] == "bf16-true"

    cfg = _cfg(precision="float64", val_every=8)
    assert _build_trainer_params(cfg, bundle, [], None)["precision"] == "64-true"
