"""Tests for praxis/modeling.py: PraxisModel and PraxisForCausalLM.

Construction and forward (with and without an encoder), the labelled loss,
generation plumbing (input preparation, the speculative dispatch, what a custom
decoding method receives), RL policy and surgical-classifier wiring, and the
registry-wide causality sweep.
"""

import copy
import functools

import pytest
import torch
from torch.nn.parameter import UninitializedParameter
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StopStringCriteria

from praxis import PraxisConfig, registry
from praxis.attention.cache import PraxisCache
from praxis.inference import speculative
from praxis.modeling import PraxisForCausalLM, PraxisModel, build_rl_policies
from praxis.policies.preference import PreferencePolicy
from praxis.tasks import TaskType
from praxis.tokenizers.byte_level import ByteLevelTokenizer
from praxis.tokenizers.chat_templates import chat_format_of

# ------------------------------------------------------------------------------
# forward and loss
# ------------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return PraxisConfig(
        vocab_size=1000,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        max_length=128,
        decoder_type="sequential",
        encoder_type=None,
    )


@pytest.fixture
def input_ids():
    return torch.randint(0, 1000, (2, 16))


@pytest.fixture
def attention_mask(input_ids):
    return torch.ones_like(input_ids)


def test_praxis_model_forward(small_config, input_ids, attention_mask):
    model = PraxisModel(small_config)
    assert model.encoder is False
    assert model.embeds is not None and model.decoder is not None

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs.last_hidden_state.shape == (
        *input_ids.shape,
        small_config.hidden_size,
    )
    assert outputs.h_encoder is None
    assert outputs.patch_lengths is None


def test_praxis_causal_lm_with_labels(small_config, input_ids, attention_mask):
    """Labels arrive pre-shifted (``input_ids[:, 1:]``): training gives a finite
    scalar loss whose gradient reaches the classifier, and eval logits cover every
    position."""
    model = PraxisForCausalLM(small_config)
    assert small_config.causal is True
    assert model.encoder is False
    assert model.criterion is not None and model.strategy is not None

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert logits.shape == (*input_ids.shape, small_config.vocab_size)

    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids[:, 1:].contiguous(),
    )
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.classifier.parameters()
    )


def test_an_unknown_strategy_falls_back_to_naive(small_config):
    small_config.strategy = "no_such_strategy"
    model = PraxisForCausalLM(small_config)
    assert type(model.strategy) is registry.lookup("strategies", "naive")


@pytest.fixture
def encoder_config():
    return PraxisConfig(
        vocab_size=256,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        max_length=128,
        decoder_type="sequential",
        encoder_type="byte_latent",
    )


@pytest.fixture
def byte_encoder_input_ids():
    """Two copies of one short ASCII text, right-padded to 16 bytes."""
    tokenizer = ByteLevelTokenizer()
    tokens = tokenizer.encode("Hello, world! 123", add_special_tokens=True)
    padded = tokens + [tokenizer.pad_token_id] * (16 - len(tokens))
    return torch.tensor([padded] * 2, dtype=torch.long)


def test_praxis_model_with_encoder_forward(encoder_config, byte_encoder_input_ids):
    model = PraxisModel(encoder_config).eval()
    assert hasattr(model.encoder, "encode")
    assert model.decoder is not None

    with torch.no_grad():
        outputs = model(
            input_ids=byte_encoder_input_ids,
            attention_mask=torch.ones_like(byte_encoder_input_ids),
        )

    assert outputs.last_hidden_state.shape[0] == byte_encoder_input_ids.shape[0]
    assert outputs.last_hidden_state.shape[-1] == encoder_config.hidden_size


def test_praxis_causal_lm_with_encoder_forward(encoder_config, byte_encoder_input_ids):
    """The classifier owns its scorer in every mode, so it exists with an
    encoder too; the encoder produces features, the classifier classifies them."""
    model = PraxisForCausalLM(encoder_config).eval()
    assert model.classifier.scorer is not None
    assert model.criterion is not None and model.strategy is not None

    with torch.no_grad():
        outputs = model(
            input_ids=byte_encoder_input_ids,
            attention_mask=torch.ones_like(byte_encoder_input_ids),
        )

    assert outputs.logits.shape[0] == byte_encoder_input_ids.shape[0]
    assert outputs.logits.shape[-1] >= encoder_config.vocab_size


# ------------------------------------------------------------------------------
# generation
# ------------------------------------------------------------------------------


def test_prepare_inputs_for_generation(small_config, input_ids, attention_mask):
    model = PraxisForCausalLM(small_config)

    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    )
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "past_key_values" not in inputs

    # An empty cache means prefill: the full prompt passes through.
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        current_state="dummy_state",
        use_cache=True,
    )
    assert inputs["input_ids"].shape == input_ids.shape
    assert isinstance(inputs["past_key_values"], PraxisCache)
    assert inputs["current_state"] == "dummy_state"

    # With cached content, only the new suffix is fed.
    cache = PraxisCache()
    past_len = input_ids.shape[1] - 1
    cache.update(
        torch.zeros(input_ids.shape[0], 1, past_len, 4),
        torch.zeros(input_ids.shape[0], 1, past_len, 4),
        0,
    )
    inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    assert inputs["input_ids"].shape == (input_ids.shape[0], 1)
    assert inputs["past_key_values"] is cache


def test_generate_dispatches_to_speculative_by_default():
    """A DEFAULT GenerationConfig must reach the speculative path.

    On transformers>=5 a default GenerationConfig leaves ``num_beams`` as None,
    so a ``getattr(..., "num_beams", 1) == 1`` check is False and every real
    generation silently took the plain HF loop - no drafting, and
    ``mtp_accept_run``/``mtp_draft_width`` never reached the dashboard.
    """
    assert (
        getattr(GenerationConfig(), "num_beams", 1) != 1
    ), "sanity: this test is only meaningful while an unset num_beams is not 1"

    # Byte-latent + prismatic4 classifier + dual memory + VEAR MTP: the drafting stack.
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        classifier_type="prismatic4",
        memory_type="mal_energy_dual",
        mtp_type="vear",
        mtp_depth=4,
    )
    torch.manual_seed(0)
    model = PraxisForCausalLM(config).eval()

    ids = torch.randint(4, 260, (1, 12))
    with torch.no_grad():
        out = model.generate(
            ids, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False)
        )

    assert out.shape[1] > ids.shape[1]
    assert model.mtp._accept_seen > 0, "speculative decoding did not run"
    # ...which is what puts the two realized-throughput metrics on the wire.
    metrics = model.mtp.training_metrics()
    assert "mtp_accept_run" in metrics
    assert 1 <= metrics["mtp_draft_width"] <= config.mtp_depth


def test_speculative_decode_defers_to_the_standard_loop_on_a_batch():
    """Speculative decoding verifies ONE growing prefix - its batch axis
    carries the n truncated prefixes, not n sequences. Handed a real batch it
    died on a ``.item()`` over a per-row tensor, which broke BrierLMCallback
    (two continuations per prompt, for a batch of prompts). Any MTP model
    generating with B > 1 hits it."""
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        max_length=512,
        decoder_type="sequential",
        classifier_type="forward",
        encoder_type="abstractinator_v1",
        tokenizer_type="byte_level",
        codebook_size=256,
        mtp_depth=3,
        mtp_type="per_depth",
    )
    torch.manual_seed(0)
    model = PraxisForCausalLM(config).eval()
    gc = GenerationConfig(max_new_tokens=16, temperature=1.0, do_sample=True)
    for b in (1, 2, 5):
        out = model.generate(torch.randint(0, 256, (b, 32)), generation_config=gc)
        assert out.shape[0] == b, (b, out.shape)


# What transformers hands a CALLABLE decoding method (MTP speculative decoding,
# CALM's patch vote). `_extract_generation_mode_kwargs` (transformers 5.2.0)
# pops `tokenizer` out of kwargs, then - for a callable custom_generate -
# rebuilds the dict from kwargs, where `tokenizer` no longer is; `streamer` is
# lost separately, by being in `_sample`'s signature. Losing the tokenizer made
# `_get_stopping_criteria` RAISE before the method was called, so every request
# under a stop-strings chat format failed with "we could not locate a tokenizer".


@pytest.fixture(scope="module")
def spec_model():
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=1024,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        max_length=512,
        mtp_type="vear",
        mtp_depth=4,
    )
    model = PraxisForCausalLM(config).eval()
    assert model._resolve_decoding_method(torch.zeros(1, 4, dtype=torch.long), None)
    return model


class _Recorder:
    def __init__(self):
        self.puts = []
        self.ended = False

    def put(self, value):
        self.puts.append(value)

    def end(self):
        self.ended = True


def _prose_prompt(tokenizer):
    return torch.tensor([tokenizer.encode("user\n\nhi\n\nassistant\n\n")])


def test_the_method_receives_the_tokenizer_and_the_prepared_stop_criteria(
    spec_model, prose_tokenizer, monkeypatch
):
    """...and it arrives as a real StopStringCriteria, not as a `stop_strings`
    the method would have to re-derive."""
    seen = {}

    def capture(model, input_ids, **kwargs):
        seen.update(kwargs)
        return input_ids

    monkeypatch.setattr(speculative, "speculative_decoding", capture)
    spec_model.generate(
        _prose_prompt(prose_tokenizer),
        generation_config=GenerationConfig(
            max_new_tokens=4,
            do_sample=False,
            stop_strings=list(chat_format_of(prose_tokenizer).stop_strings()),
        ),
        tokenizer=prose_tokenizer,
    )

    assert seen["tokenizer"] is prose_tokenizer
    assert any(
        isinstance(c, StopStringCriteria) for c in seen["stopping_criteria"]
    ), "the format's boundaries never became a criterion"


def test_the_streamer_reaches_a_custom_decoding_method(spec_model, prose_tokenizer):
    """Under its own name - the methods stay signature-compatible with
    `_sample` rather than taking it under a private alias."""
    rec = _Recorder()
    spec_model.generate(
        _prose_prompt(prose_tokenizer),
        generation_config=GenerationConfig(max_new_tokens=5, do_sample=False),
        tokenizer=prose_tokenizer,
        streamer=rec,
    )
    # One put for the prompt (transformers) plus one per committed token (ours).
    assert len(rec.puts) > 1
    assert rec.ended is True


def test_the_standard_path_is_untouched(spec_model, prose_tokenizer):
    """The override only restores kwargs for a CALLABLE method; when none is
    resolved, transformers' own behaviour has to be exactly what it was."""
    mode_kwargs = spec_model._extract_generation_mode_kwargs(
        None, {"tokenizer": prose_tokenizer}, None, None, None
    )
    assert mode_kwargs.get("tokenizer") is prose_tokenizer
    assert "streamer" not in mode_kwargs


# ------------------------------------------------------------------------------
# RL policy wiring
# ------------------------------------------------------------------------------
# The forward-path preference policy (praxis/policies/preference.py) as the model
# builds and weights it.

CHOSEN = int(TaskType.PREF_CHOSEN)
REJECTED = int(TaskType.PREF_REJECTED)


def _policy_config(**kwargs):
    return PraxisConfig(
        vocab_size=64,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        decoder_type="sequential",
        **kwargs,
    )


def test_rejected_tokens_excluded_from_main_ce():
    """_build_loss_weights zeroes PREF_REJECTED positions regardless of the
    weighter profile - the card's no-SFT contract for the rejected side."""
    model = PraxisForCausalLM(_policy_config())
    labels = torch.randint(0, 64, (1, 8))
    task = torch.full((1, 8), CHOSEN, dtype=torch.long)
    task[0, 4:] = REJECTED
    weights = model._build_loss_weights(
        labels=labels, task_type_ids=task, assistant_mask=None
    )
    assert (weights[0, 4:] == 0).all()
    assert (weights[0, :4] > 0).all()


def test_build_rl_policies_recall_family():
    cfg = _policy_config(rl_type=["engagement", "joke", "preference"])
    policy, policy_type, recall = build_rl_policies(cfg)
    assert policy is None and policy_type is None
    assert set(recall) == {"engagement", "joke", "preference"}
    assert isinstance(recall["preference"], PreferencePolicy)


def test_byte_latent_forward_with_preference():
    """The full -d-shaped stack trains a step with the preference loss landing
    in the container and finite gradients."""
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_v0",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        classifier_type="prismatic4",
        residual_type="smear",
        rl_type=["preference"],
    )
    model = PraxisForCausalLM(cfg).train()
    # Long enough that each side clears MIN_SIDE_TOKENS after byte-level
    # patching and repadding.
    ids = torch.randint(4, 260, (2, 128))
    task = torch.full((2, 128), CHOSEN, dtype=torch.long)
    task[1] = REJECTED
    mask = torch.ones(2, 128, dtype=torch.uint8)
    out = model(
        input_ids=ids,
        labels=ids[..., 1:].contiguous(),
        task_type_ids=task,
        assistant_mask=mask,
    )
    assert torch.isfinite(out.loss)
    metrics = model.policies["preference"].get_metrics()
    assert "preference_margin" in metrics
    out.loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


# ------------------------------------------------------------------------------
# surgical classifiers
# ------------------------------------------------------------------------------
# prismatic9 trains each arm on its own objective and hands the trunk one
# PCGrad-combined gradient. The model has to wire around that: HALO's geometric
# term moves from the criterion to the classifier, and MTP must still reach the
# classifier's input through the detached arms.


def _halo_config(classifier_type, **overrides):
    return PraxisConfig(
        vocab_size=1000,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=2,
        max_length=128,
        decoder_type="sequential",
        encoder_type=None,
        classifier_type=classifier_type,
        loss_func="halo",
        **overrides,
    )


def test_full_model_survives_the_lazy_init_pass():
    """End-to-end reproduction of the -o startup crash: train() + no_grad."""
    torch.manual_seed(0)
    m = PraxisForCausalLM(_halo_config("prismatic9"))
    m.train()
    ids = torch.ones((2, 16), dtype=torch.long)
    with torch.no_grad():
        out = m(input_ids=ids, labels=ids[..., 1:].contiguous())
    assert out.loss is not None
    # The real training step still works afterwards.
    out = m(input_ids=ids, labels=ids[..., 1:].contiguous())
    out.loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_validation_loss_stays_comparable_to_prismatic8():
    """prismatic9 flips `composite_geometry` off so HALO's geometric term is
    not double-counted (the classifier owns it as a Jacobian row). That suppression
    must be TRAINING-ONLY: at eval nothing replaces the term, so zeroing it
    there just deletes a component of val_loss."""

    def val_loss(classifier_type):
        torch.manual_seed(0)
        m = PraxisForCausalLM(_halo_config(classifier_type)).eval()
        ids = torch.arange(16).remainder(900).unsqueeze(0).repeat(2, 1)
        with torch.no_grad():
            out = m(input_ids=ids, labels=ids[:, 1:].contiguous())
        return float(out.loss), m

    eight, m8 = val_loss("prismatic8")
    nine, m9 = val_loss("prismatic9")
    # The criterion is configured differently...
    assert m8.criterion.main.composite_geometry is True
    assert m9.criterion.main.composite_geometry is False
    # ...but at EVAL both must score the same composite objective.
    assert nine == pytest.approx(
        eight, rel=1e-4
    ), f"val loss diverged: prismatic8 {eight}, prismatic9 {nine}"


def test_mtp_still_trains_under_a_surgical_classifier():
    """Detaching every arm in the blend severs the path from the classifier's
    OUTPUT back to its INPUT - and MTP classifies its draft states with that
    same classifier, so its loss reached nothing, not even MTP's own bank."""

    def run(classifier_type):
        torch.manual_seed(0)
        m = PraxisForCausalLM(
            _halo_config(classifier_type, mtp_depth=3, mtp_type="per_depth")
        ).train()
        ids = torch.randint(0, 1000, (2, 16))
        m(input_ids=ids, labels=ids[:, 1:].contiguous()).loss.backward()
        live = [
            n
            for n, p in m.mtp.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        return m, live, sum(1 for _ in m.mtp.named_parameters())

    _, live8, total = run("prismatic8")
    _, live9, _ = run("prismatic9")
    assert len(live8) == total, "baseline broke; the comparison is meaningless"
    assert len(live9) == total, f"MTP starved under prismatic9: {len(live9)}/{total}"

    # The fallback path, for a classifier with no undetached() at all.
    m, live, total = run("forward")
    assert not hasattr(m.classifier, "undetached")
    assert len(live) == total


# ------------------------------------------------------------------------------
# causality
# ------------------------------------------------------------------------------
# Every router, classifier, block and encoder in the registries is causal, in training
# and in inference.
#
# One token changes in one row; no logit at an earlier position of that row, and
# none in any other row, may move. CALM's logits reconstruct each K-token chunk
# from that chunk's own latent, so for it "earlier" means an earlier chunk. Each
# forward runs on a fresh copy of the model with the same RNG, because
# training-mode forwards mutate state (EMA buffers, dual variables) and reusing
# one model would show movement that is not a leak. Every parameter is first
# moved off its initialization, because zero-initialized deviations (LoRA B,
# SMEAR banks) make routing invisible at init and would hide a leak there.

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
    ("classifier", "tied"): {"tie_weights": True},
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
    + [("classifier", key) for key in sorted(registry.namespace("classifiers"))]
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


@pytest.fixture(scope="module", autouse=True)
def _release_causality_models():
    """The sweep builds each model once for both tests; drop them afterwards."""
    yield
    _model.cache_clear()


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


def _assert_causal(before, other_row, reached):
    assert reached > 0, "no edit moved anything, so the check cannot see a leak"
    assert (
        before <= TOLERANCE
    ), f"a logit before the edited position moved by {before:.3e}"
    assert other_row <= TOLERANCE, f"a logit in another row moved by {other_row:.3e}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_inference_is_causal(case):
    _assert_causal(*_movement(*case, train=False))


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=MOD_TRAINING) if _is_mod(*case) else case
        for case in CASES
    ],
    ids=_ids,
)
def test_training_is_causal(case):
    _assert_causal(*_movement(*case, train=True))
