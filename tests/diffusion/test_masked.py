"""The masked-diffusion objective: corruption, the ELBO term, and the three
properties that follow from using it.

The properties are the point. A diffusion run is only sound because the token
being scored was replaced with an absorbing symbol before the model saw the
sequence; drop any one of them and the loss measures something other than what
it reports. So each is asserted here, and the causal arm is asserted to be
UNCHANGED in the same file - a check that only fires one way is the check that
lets a leak through (see next/temporal_mesh_audit.md).
"""

import math

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.diffusion.masked import MaskedDiffusion
from praxis.modeling import PraxisForCausalLM

MASK_ID = 256
VOCAB = 257


def _config(**over):
    base = dict(
        hidden_size=64,
        embed_size=64,
        num_heads=1,
        num_queries=1,
        head_size=32,
        depth=2,
        num_layers=1,
        block_size=64,
        dropout=0.0,
        encoder_type=None,
        attention_type="causal",
        classifier_type="forward",
        regularizers=[],
        mtp_type=None,
        rl_type=[],
    )
    base.update(over)
    return PraxisConfig(**base)


def _diffusion_config(**over):
    over.setdefault("diffusion_type", "masked")
    over.setdefault("mask_token_id", MASK_ID)
    over.setdefault("vocab_size", VOCAB)
    return _config(**over)


@pytest.fixture
def diffusion():
    return MaskedDiffusion(mask_token_id=MASK_ID, vocab_size=VOCAB)


# ---------------------------------------------------------------------------
# corruption
# ---------------------------------------------------------------------------


def test_corrupt_writes_the_mask_id_and_only_there(diffusion):
    ids = torch.randint(0, 256, (8, 64))
    noisy, masked, _ = diffusion.corrupt(ids)

    assert (noisy[masked] == MASK_ID).all()
    assert (noisy[~masked] == ids[~masked]).all(), "an uncorrupted position changed"
    assert (noisy == MASK_ID).sum() == masked.sum()


def test_corrupt_rate_tracks_the_sampled_level(diffusion):
    """The realized mask fraction is the sampled level, not a fixed ratio."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (256, 128))
    _, masked, t = diffusion.corrupt(ids)

    realized = masked.float().mean(dim=-1)
    # Binomial(128, t): 4 standard deviations is ~0.18 at the worst t.
    assert torch.allclose(realized, t.squeeze(-1), atol=0.2)
    assert float(t.min()) >= diffusion.eps
    assert float(t.max()) <= 1.0


def test_corrupt_never_leaves_a_row_empty(diffusion):
    """A row with nothing masked contributes no gradient but still divides by
    t, so it is pure variance. The fallback has to fire even at the smallest
    level the sampler can produce."""
    tiny = MaskedDiffusion(mask_token_id=MASK_ID, eps=1e-9, vocab_size=VOCAB)
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (64, 8))
    _, masked, _ = tiny.corrupt(ids)
    assert masked.any(dim=-1).all()


def test_corrupt_leaves_padding_alone(diffusion):
    ids = torch.randint(0, 256, (4, 32))
    attn = torch.ones_like(ids)
    attn[:, 20:] = 0
    _, masked, _ = diffusion.corrupt(ids, attention_mask=attn)
    assert not masked[:, 20:].any(), "padding was corrupted"


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def test_loss_scores_only_corrupted_positions(diffusion):
    """A confident WRONG answer at an untouched position must cost nothing:
    those positions are visible in the input, so scoring them would be
    rewarding a copy."""
    B, L = 2, 16
    targets = torch.randint(0, 256, (B, L))
    masked = torch.zeros(B, L, dtype=torch.bool)
    masked[:, :4] = True
    t = torch.full((B, 1), 0.25)

    logits = torch.zeros(B, L, VOCAB)
    logits.scatter_(-1, targets.unsqueeze(-1), 20.0)  # perfect everywhere
    perfect = diffusion.compute_loss(logits, targets, masked, t)

    logits[:, 8:] = 0.0
    logits[:, 8:].scatter_(-1, ((targets[:, 8:] + 1) % 256).unsqueeze(-1), 20.0)
    spoiled = diffusion.compute_loss(logits, targets, masked, t)

    assert torch.allclose(perfect, spoiled), "an unmasked position entered the loss"
    assert float(perfect) < 1e-4


def test_loss_is_the_weighted_elbo(diffusion):
    """``sum(CE over masked) / (t * L)`` per row, averaged. Checked against the
    definition rather than against a previous value."""
    B, L = 3, 12
    torch.manual_seed(0)
    logits = torch.randn(B, L, VOCAB)
    targets = torch.randint(0, 256, (B, L))
    masked = torch.rand(B, L) < 0.5
    masked[:, 0] = True
    t = torch.rand(B, 1) * 0.9 + 0.05

    got = diffusion.compute_loss(logits, targets, masked, t)

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none"
    ).view(B, L)
    expected = ((ce * masked).sum(-1) / (t.squeeze(-1) * L)).mean()
    assert torch.allclose(got, expected, atol=1e-5)


def test_uniform_logits_score_at_chance(diffusion):
    B, L = 4, 32
    logits = torch.zeros(B, L, VOCAB)
    targets = torch.randint(0, 256, (B, L))
    masked = torch.ones(B, L, dtype=torch.bool)
    t = torch.ones(B, 1)

    loss = diffusion.compute_loss(logits, targets, masked, t)
    assert abs(float(loss) - math.log(VOCAB)) < 1e-4


def test_metrics_report_the_unigram_gap(diffusion):
    """The degenerate solution - predict the marginal, learn no structure -
    looks like a healthy falling loss. The gap is what distinguishes them."""
    B, L = 8, 32
    targets = torch.randint(0, 256, (B, L))
    masked = torch.ones(B, L, dtype=torch.bool)
    t = torch.ones(B, 1)

    perfect = torch.zeros(B, L, VOCAB)
    perfect.scatter_(-1, targets.unsqueeze(-1), 20.0)
    diffusion.compute_loss(perfect, targets, masked, t)
    assert diffusion.training_metrics()["diffusion_unigram_gap"] > 0

    flat = MaskedDiffusion(mask_token_id=MASK_ID, vocab_size=VOCAB)
    flat.compute_loss(torch.zeros(B, L, VOCAB), targets, masked, t)
    # Uniform logits cannot beat the marginal.
    assert flat.training_metrics()["diffusion_unigram_gap"] <= 1e-3


def test_metric_keys_are_all_declared():
    """Every key training_metrics() can emit has a chart declaration, or it
    reaches the dashboard as an undocumented series and gets dropped."""
    d = MaskedDiffusion(mask_token_id=MASK_ID, vocab_size=VOCAB)
    B, L = 16, 32
    targets = torch.randint(0, 256, (B, L))
    torch.manual_seed(0)
    _, masked, t = d.corrupt(targets)
    d.compute_loss(torch.randn(B, L, VOCAB), targets, masked, t)

    declared = set(MaskedDiffusion.metric_descriptions)
    emitted = set(d.training_metrics())
    assert emitted <= declared, f"undeclared: {sorted(emitted - declared)}"


# ---------------------------------------------------------------------------
# the three properties
# ---------------------------------------------------------------------------


def test_diffusion_model_is_bidirectional():
    """Editing a position must move the logits BEFORE it. That is the whole
    reason this objective needs its own corruption: the model is allowed to
    see the future because the answer was removed from the input."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_diffusion_config()).eval()
    assert model.config.causal is False

    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    edited = ids.clone()
    edited[0, 12] = (edited[0, 12] + 5) % 256
    with torch.no_grad():
        after = model(input_ids=edited).logits

    moved = (after - base).abs().amax(-1)[0]
    assert float(moved[:12].max()) > 1e-6, "information did not flow backward"


def test_causal_model_is_still_causal():
    """The regression guard. Flipping a model to bidirectional must not make
    the ordinary next-token arm bidirectional too."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_config()).eval()
    assert model.config.causal is True

    ids = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    edited = ids.clone()
    edited[0, 12] = (edited[0, 12] + 5) % 256
    with torch.no_grad():
        after = model(input_ids=edited).logits

    moved = (after - base).abs().amax(-1)[0]
    assert float(moved[:12].max()) <= 1e-6, "a logit before the edit moved"
    assert float(moved[12:].max()) > 0, "the edit moved nothing at all"


def test_labels_are_unshifted():
    model = PraxisForCausalLM(_diffusion_config())
    assert model.is_diffusion
    assert model.outputs_are_aligned is True


def test_mtp_is_refused():
    """MTP classifies k-step-ahead drafts with the shared classifier, which is
    a left-to-right device. Silently ignoring it would train the classifier
    against a factorisation the model does not have."""
    with pytest.raises(ValueError, match="left-to-right"):
        PraxisForCausalLM(_diffusion_config(mtp_type="per_depth"))


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_registry_entry_becomes_the_main_objective():
    """The diffusion term IS the criterion - it owns which positions are
    scored, so a separate per-token loss_func must not also be registered."""
    model = PraxisForCausalLM(_diffusion_config(loss_func="cross_entropy"))
    assert isinstance(model.criterion.main, MaskedDiffusion)
    names = [name for name, _ in model.criterion.terms()]
    assert names.count("main") == 1


@pytest.mark.parametrize("key", sorted(registry.namespace("diffusion")))
def test_every_registry_entry_builds_and_trains(key):
    cls = registry.lookup("diffusion", key)
    module = cls(mask_token_id=MASK_ID, vocab_size=VOCAB)
    ids = torch.randint(0, 256, (4, 32))
    noisy, masked, t = module.corrupt(ids)
    loss = module.compute_loss(torch.randn(4, 32, VOCAB), ids, masked, t)
    assert torch.isfinite(loss)
    assert (noisy[masked] == MASK_ID).all()


def test_forward_produces_a_finite_gradient():
    torch.manual_seed(0)
    model = PraxisForCausalLM(_diffusion_config()).train()
    ids = torch.randint(0, 256, (2, 32))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "nothing received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_forward_does_not_corrupt_without_labels():
    """A label-free forward is inference, where the caller has already placed
    the masks. Corrupting there would overwrite the prompt."""
    torch.manual_seed(0)
    model = PraxisForCausalLM(_diffusion_config()).eval()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=ids).logits
    assert torch.allclose(a, b), "an inference forward was nondeterministic"


# ---------------------------------------------------------------------------
# the mask symbol has to exist for every tokenizer, not just byte ones
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _derive(tokenizer, **arg_kw):
    """Run the CLI's config derivation against a stand-in tokenizer."""
    from praxis.cli.processors.config import ConfigBuilder

    args = _FakeArgs(
        diffusion_type="masked", vocab_size=None, tokenizer_type="x", **arg_kw
    )
    return ConfigBuilder.create_praxis_config(args, tokenizer)


def test_mask_id_is_derived_for_a_subword_tokenizer():
    """It used to live inside the byte-alphabet branch, which a subword
    tokenizer never enters - so `mask_token_id` stayed None and the corruption
    masked with None. The absorbing symbol is not a byte-level concept."""

    class _Subword:
        vocab_size = 24004
        pad_token_id = 0

    config = _derive(_Subword())
    assert config.mask_token_id == 24004, "no mask symbol for a subword vocab"
    assert config.vocab_size == 24005, "the alphabet did not widen for it"


def test_mask_id_sits_past_a_byte_alphabet():
    class _Byte:
        vocab_size = 1024
        byte_alphabet_size = 256
        byte_offset = 0
        pad_token_id = 0

    config = _derive(_Byte())
    assert config.mask_token_id == 256
    assert config.byte_vocab_size == 257, "the byte alphabet did not widen"


def test_no_mask_id_without_diffusion():
    class _Byte:
        vocab_size = 1024
        byte_alphabet_size = 256
        byte_offset = 0
        pad_token_id = 0

    from praxis.cli.processors.config import ConfigBuilder

    config = ConfigBuilder.create_praxis_config(
        _FakeArgs(diffusion_type=None, vocab_size=None, tokenizer_type="x"), _Byte()
    )
    assert config.mask_token_id is None
    assert config.byte_vocab_size == 256, "the alphabet widened with no diffusion"


def test_mask_id_clears_tokens_added_past_the_vocab():
    """``vocab_size`` is not always the highest id a tokenizer can emit - HF
    registers added tokens past it, and ``len(tokenizer)`` counts them. Placing
    the mask at ``vocab_size`` would then sit on top of a real token, and the
    only symptom would be the model being asked to reconstruct the very symbol
    that means "reconstruct me"."""

    class _WithAddedTokens:
        vocab_size = 4100
        pad_token_id = 0

        def __len__(self):
            return 4104  # four tokens registered past the reported vocab

    config = _derive(_WithAddedTokens())
    assert config.mask_token_id == 4104, "the mask landed on an added token"
    assert config.vocab_size == 4105


def test_mask_id_is_one_past_the_last_real_id():
    """The arithmetic the run repr shows: id N, width N+1."""

    class _Plain:
        vocab_size = 4100
        pad_token_id = 0

        def __len__(self):
            return 4100

    config = _derive(_Plain())
    assert config.mask_token_id == 4100
    assert config.vocab_size == config.mask_token_id + 1
