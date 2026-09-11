"""Every router, head, block and encoder in the registries is causal, in training
and in inference.

One token changes in one row; no logit at an earlier position of that row, and
none in any other row, may move. CALM's logits reconstruct each K-token chunk
from that chunk's own latent, so for it "earlier" means an earlier chunk. Each
forward runs on a fresh copy of the model with the same RNG, because
training-mode forwards mutate state (EMA buffers, dual variables) and reusing
one model would show movement that is not a leak. Every parameter is first
moved off its initialization, because zero-initialized deviations (LoRA B,
SMEAR banks) make routing invisible at init and would hide a leak there.
"""

import copy
import functools

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from praxis import PraxisConfig, registry
from praxis.modeling import PraxisForCausalLM

BASE = dict(
    vocab_size=1024,
    hidden_size=32,
    embed_size=32,
    num_heads=4,
    num_layers=1,
    depth=3,
    num_experts=4,
    tokenizer_type="byte_level",
    decoder_type="sequential",
)

# What a component needs to build, or to reach the code path under test. The
# default attention has no working shape at this width under a merge router.
OVERRIDES = {
    "router": {"attention_type": "causal"},
    ("router", "arc_mixture"): {"num_layers": 2, "depth": 4},  # a layer under 1.0
    ("head", "tied"): {"tie_weights": True},
    ("block", "mru"): {"hidden_size": 64, "embed_size": 64},  # square head size
    "encoder": {"hidden_size": 64, "embed_size": 64, "num_heads": 2},
}

# Expert-choice top-k decides a token's route from the tokens after it. The
# Mixture-of-Depths paper trains with it anyway and routes causally only at
# inference (arXiv:2404.02258, Sec. 3.5); praxis/routers/mixture_of_depths.py
# follows the paper, so training here is non-causal by design.
MOD_TRAINING = pytest.mark.xfail(
    strict=True,
    reason="Mixture-of-Depths trains on non-causal expert-choice top-k, as the paper does",
)

SEQ, EDITS = 16, (5, 9, 13)

# Encoders pool 8-byte patches or 4-16 token chunks: a longer row of byte ids
# puts each edit in its own patch, with whole patches before and after it.
BYTE_SEQ, BYTE_EDITS = 48, (12, 27, 41)

# Float noise, not a leak: a batch-dependent kernel shape (inference MoD pads
# every row to the batch's widest selection) moves logits by ~1e-7, where a
# leak moves them by 1e-3 or more.
TOLERANCE = 1e-5

CASES = (
    [("router", key) for key in sorted(registry.namespace("routers"))]
    + [("head", key) for key in sorted(registry.namespace("heads"))]
    + [("block", key) for key in sorted(registry.namespace("blocks"))]
    + [("encoder", key) for key in sorted(registry.namespace("encoders"))]
    # Unlisted names that carry a profile of their own, not a listed one's.
    + [("encoder", key) for key in sorted(registry.namespace("encoders").unlisted())]
)


def _is_mod(kind: str, key: str) -> bool:
    return kind == "router" and (
        key.startswith("mixture_of_depths") or key == "arc_mixture"
    )


@functools.lru_cache(maxsize=None)
def _model(kind: str, key: str) -> PraxisForCausalLM:
    cfg = dict(BASE)
    cfg.update(OVERRIDES.get(kind, {}))
    cfg.update(OVERRIDES.get((kind, key), {}))
    cfg[f"{kind}_type"] = key
    torch.manual_seed(0)
    model = PraxisForCausalLM(PraxisConfig(**cfg))
    with torch.no_grad():
        if any(isinstance(p, UninitializedParameter) for p in model.parameters()):
            ids = torch.zeros(1, BYTE_SEQ, dtype=torch.long)
            model(input_ids=ids, labels=ids[:, 1:].contiguous())  # size lazy params
        for p in model.parameters():
            if p.is_floating_point():
                p.add_(0.05 * torch.randn_like(p))
    return model


def _movement(kind: str, key: str, train: bool):
    """Worst logit change before an edited position, and in the other row, over
    several edits - a single edit can miss a data-dependent leak such as a
    top-k whose membership it happens not to flip - and the largest change the
    edits made where they may. Not every edit must reach the logits: training
    dropout can swallow one (CALM's codec drops input tokens)."""
    pristine = _model(kind, key)
    byte = kind == "encoder"
    seq, edits = (BYTE_SEQ, BYTE_EDITS) if byte else (SEQ, EDITS)
    low, high = (0, 256) if byte else (4, 900)
    encoder = getattr(pristine, "encoder", None)
    chunk = encoder.K if getattr(encoder, "handles_loss", False) else 1
    torch.manual_seed(1)
    ids = torch.randint(low, high, (2, seq))

    def logits(inputs):
        model = copy.deepcopy(pristine)
        model.train(train)
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(input_ids=inputs, labels=inputs[:, 1:].contiguous())
        return out.logits.float()

    base = logits(ids)
    before = other_row = reached = 0.0
    for position in edits:
        edited = ids.clone()
        edited[0, position] = (edited[0, position] + 17) % (high - low) + low
        delta = (logits(edited) - base).abs().amax(dim=-1)
        start = position // chunk * chunk
        if start:
            before = max(before, delta[0, :start].max().item())
        other_row = max(other_row, delta[1].max().item())
        reached = max(reached, delta[0, start:].max().item())
    return before, other_row, reached


def _ids(case):
    return f"{case[0]}={case[1]}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_inference_is_causal(case):
    before, other_row, reached = _movement(*case, train=False)
    assert reached > 0, "no edit moved anything, so the check cannot see a leak"
    assert (
        before <= TOLERANCE
    ), f"a logit before the edited position moved by {before:.3e}"
    assert other_row <= TOLERANCE, f"a logit in another row moved by {other_row:.3e}"


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=MOD_TRAINING) if _is_mod(*case) else case
        for case in CASES
    ],
    ids=_ids,
)
def test_training_is_causal(case):
    before, other_row, reached = _movement(*case, train=True)
    assert reached > 0, "no edit moved anything, so the check cannot see a leak"
    assert (
        before <= TOLERANCE
    ), f"a logit before the edited position moved by {before:.3e}"
    assert other_row <= TOLERANCE, f"a logit in another row moved by {other_row:.3e}"
