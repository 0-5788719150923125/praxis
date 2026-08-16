"""One --precision flag, three knobs kept in agreement.

The failure this pins down is drift: the model built in one dtype, Lightning
stepping in another, and the matmul policy set from a third place. Every
consumer resolves the same profile from the same registry, so what is asserted
here is that the flag reaches all three - model dtype, Lightning precision
string, float32 matmul policy - and that a level the hardware cannot honor is
downgraded to something that runs rather than exploding mid-step.
"""

import torch

from praxis.cli.config import RunConfig
from praxis.trainers.precision import (
    DEFAULT_PRECISION,
    PRECISION_REGISTRY,
    canonical_precision,
    cast_module,
    init_context,
    resolve_precision,
)


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


def test_default_matches_the_status_quo():
    """The default is what Praxis did before the flag existed: fp32 weights,
    fp32 gradients, TF32 matmul kernels."""
    profile = resolve_precision(None, "cpu", verbose=False)
    assert profile.name == DEFAULT_PRECISION == "float32"
    assert profile.lightning == "32-true"
    assert profile.matmul == "high"
    assert profile.torch_dtype is None  # nothing is cast; fp32 is torch's own
    assert _cfg().precision_profile.name == "float32"


def test_aliases_and_case_fold_to_registry_keys():
    for alias, expected in [
        ("bf16", "bfloat16"),
        ("BF16", "bfloat16"),
        ("fp16", "float16"),
        ("half", "float16"),
        ("32", "float32"),
        ("fp-64", "float64"),
        ("float_64", "float64"),
    ]:
        assert canonical_precision(alias) == expected


def test_unknown_precision_is_rejected_with_the_options():
    try:
        canonical_precision("float8")
    except ValueError as e:
        assert "float8" in str(e) and "bfloat16" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown precision")


def test_only_float64_forbids_tf32():
    """ "medium" is defined as permitting a bf16 internal datatype, so no
    profile uses it: the fp32-carrying levels take TF32 via "high", and the
    level whose entire premise is arithmetic width takes none of it."""
    for name, profile in PRECISION_REGISTRY.items():
        assert profile.matmul == ("highest" if name == "float64" else "high")


def test_every_profile_is_internally_coherent():
    """A profile that names a param dtype must run the trainer in the matching
    precision - a bf16 model stepped by a 32-true trainer is the exact drift
    this registry exists to prevent."""
    expected = {"float64": "64-true", "bfloat16": "bf16-true", "float16": "16-true"}
    for name, profile in PRECISION_REGISTRY.items():
        assert profile.name == name
        if profile.param_dtype is not None:
            assert profile.lightning == expected[profile.param_dtype]
        else:
            # No cast means fp32 master weights: either plain fp32 or a mixed
            # plugin that keeps them.
            assert profile.lightning in ("32-true", "16-mixed")


def test_fp16_falls_back_to_bf16_on_cpu():
    """CPU has no fp16 matmul kernels, so the run would die on the first
    linear; bf16 is the nearest level with coverage."""
    assert resolve_precision("float16", "cpu", verbose=False).name == "bfloat16"
    # ... and the request is honored where the kernels exist.
    assert resolve_precision("float16", "cuda", verbose=False).name in (
        "float16",
        "bfloat16",
    )


def test_init_context_scopes_the_default_dtype():
    """Modules build in the profile's dtype; everything created afterwards -
    loss accumulators, metrics, dataset tensors - stays fp32."""
    before = torch.get_default_dtype()
    with init_context(PRECISION_REGISTRY["bfloat16"]):
        assert torch.get_default_dtype() is torch.bfloat16
        assert torch.zeros(2).dtype is torch.bfloat16
    assert torch.get_default_dtype() is before
    assert torch.zeros(2).dtype is before


def test_init_context_restores_on_failure():
    before = torch.get_default_dtype()
    try:
        with init_context(PRECISION_REGISTRY["float64"]):
            raise RuntimeError("model blew up during construction")
    except RuntimeError:
        pass
    assert torch.get_default_dtype() is before


def test_cast_module_moves_params_and_buffers():
    """Buffers registered from explicit fp32 tensors are exactly what the
    default-dtype context misses, so the cast has to reach them too."""
    model = torch.nn.Linear(4, 4)
    model.register_buffer("scale", torch.ones(4, dtype=torch.float32))

    cast_module(model, PRECISION_REGISTRY["bfloat16"])
    assert model.weight.dtype is torch.bfloat16
    assert model.scale.dtype is torch.bfloat16

    # fp32/mixed profiles leave the model alone.
    fp32 = torch.nn.Linear(4, 4)
    cast_module(fp32, PRECISION_REGISTRY["float32"])
    assert fp32.weight.dtype is torch.float32


def test_cast_preserves_parameter_identity():
    """The optimizer is built from these Parameter objects; an in-place data
    swap keeps its references valid, a fresh Parameter would silently orphan
    them."""
    model = torch.nn.Linear(4, 4)
    params = list(model.parameters())
    cast_module(model, PRECISION_REGISTRY["bfloat16"])
    assert [id(p) for p in model.parameters()] == [id(p) for p in params]


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
