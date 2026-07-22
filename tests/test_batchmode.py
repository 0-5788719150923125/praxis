"""Mode-level losses (praxis/losses/batchmode.py): mode_cross_entropy
(mode-as-target) and mode_baseline_cross_entropy (the deviation-above-mode
fallback)."""

import torch

from praxis.losses import LOSS_REGISTRY
from praxis.losses.batchmode import (
    FLOOR,
    ModeBaselineCrossEntropyLoss,
    ModeCrossEntropyLoss,
)


def _bimodal_batch(vocab=32, n_easy=90, n_hard=10):
    """Logits with a 90/10 bimodal loss split: easy tokens near-zero loss,
    hard tokens confidently wrong (high loss)."""
    torch.manual_seed(0)
    n = n_easy + n_hard
    labels = torch.randint(0, vocab, (1, n))
    logits = torch.zeros(1, n, vocab)
    logits.scatter_(-1, labels.unsqueeze(-1), 10.0)  # easy: correct and confident
    hard = torch.arange(n_easy, n)
    wrong = (labels[0, hard] + 1) % vocab
    logits[0, hard] = 0.0
    logits[0, hard, wrong] = 10.0  # hard: confidently wrong
    logits.requires_grad_(True)
    return logits, labels


def test_target_and_baseline_bracket_the_mean():
    """Mode-as-target sits below the plain mean (tail down-weighted); the
    baseline dual sits above it (consensus down-weighted)."""
    logits, labels = _bimodal_batch()
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    )
    mean_ce = float(per_token.detach().mean())

    target = float(ModeCrossEntropyLoss()(logits=logits, labels=labels).detach())
    baseline = float(
        ModeBaselineCrossEntropyLoss()(logits=logits, labels=labels).detach()
    )
    assert target < mean_ce < baseline, (target, mean_ce, baseline)


def test_gradients_flow_and_degenerate_batch_matches_mean():
    for cls in (ModeCrossEntropyLoss, ModeBaselineCrossEntropyLoss):
        logits, labels = _bimodal_batch()
        loss = cls()(logits=logits, labels=labels)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

        # Uniform logits: every token has the same loss, so any re-weighting
        # reduces to the plain mean.
        flat = torch.zeros(1, 20, 32, requires_grad=True)
        lab = torch.randint(0, 32, (1, 20))
        got = cls()(logits=flat, labels=lab)
        per_token = torch.nn.functional.cross_entropy(
            flat.reshape(-1, 32), lab.reshape(-1), reduction="none"
        )
        assert torch.allclose(got, per_token.mean(), atol=1e-5)


def test_masked_and_zero_weight_positions_stay_excluded():
    logits, labels = _bimodal_batch()

    # All positions masked: no nan, backward still runs (denom<=0 path).
    all_masked = torch.full_like(labels, -100)
    loss = ModeCrossEntropyLoss()(logits=logits, labels=all_masked)
    assert torch.isfinite(loss)

    # Hard-zeroed loss_weights (the preference policy's rejected positions)
    # stay zeroed under the mode weighting: flipping those labels is invisible.
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[0, 90:] = 0.0
    a = ModeCrossEntropyLoss()(logits=logits, labels=labels, loss_weights=weights)
    flipped = labels.clone()
    flipped[0, 90:] = (flipped[0, 90:] + 5) % 32
    b = ModeCrossEntropyLoss()(logits=logits, labels=flipped, loss_weights=weights)
    assert torch.allclose(a, b, atol=1e-5)


def test_floor_keeps_tail_gradient_alive():
    """The hard tokens' label logits must still receive gradient under
    mode-as-target - the floor is the whole hypothesis."""
    logits, labels = _bimodal_batch()
    ModeCrossEntropyLoss()(logits=logits, labels=labels).backward()
    hard_grads = logits.grad[0, 90:].gather(-1, labels[0, 90:].unsqueeze(-1))
    assert (hard_grads.abs() > 0).all()


def test_byte_latent_forward_with_mode_criterion():
    """The -e-shaped byte-latent stack trains a step through the mode
    criterion with finite loss and gradients, and the dynamics extractor
    picks up the criterion's cards."""
    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM

    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=1024,
        hidden_size=32,
        embed_size=96,
        num_heads=4,
        num_layers=2,
        depth=4,
        encoder_type="abstractinator_harmonic_serpent",
        tokenizer_type="byte_level",
        decoder_type="sequential",
        activation="serpent",
        byte_level=True,
        head_type="prismatic4",
        residual_type="smear",
        loss_func="mode_cross_entropy",
    )
    model = PraxisForCausalLM(cfg).train()
    assert isinstance(model.criterion, ModeCrossEntropyLoss)
    ids = torch.randint(4, 260, (2, 24))
    out = model(input_ids=ids, labels=ids[..., 1:].contiguous())
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in model.decoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
    assert "batchmode_mode" in model.criterion.training_metrics()


def test_metrics_and_registry():
    assert LOSS_REGISTRY["mode_cross_entropy"] is ModeCrossEntropyLoss
    assert LOSS_REGISTRY["mode_baseline_cross_entropy"] is ModeBaselineCrossEntropyLoss

    crit = ModeCrossEntropyLoss()
    assert crit.training_metrics() == {}  # nothing before a forward
    logits, labels = _bimodal_batch()
    crit(logits=logits, labels=labels)
    m = crit.training_metrics()
    assert m["batchmode_mean_gap"] > 0  # bimodal batch has real tail mass
    assert FLOOR <= m["batchmode_weight_mean"] <= 1.0
    for key in m:
        assert key in ModeCrossEntropyLoss.metric_descriptions
