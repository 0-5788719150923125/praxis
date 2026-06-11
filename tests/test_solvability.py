import math
import torch

from praxis.losses.solvability import SolvabilityProbe


def make_batch(B=4, T=16, D=32, V=11):
    hidden = torch.randn(B, T, D)
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T - 1))
    labels[0, :3] = -100  # masked region must not poison the per-sample mean
    return hidden, logits, labels


def test_probe_loss_and_metrics():
    probe = SolvabilityProbe(hidden_size=32)
    hidden, logits, labels = make_batch()
    loss = probe(hidden, logits, labels)
    assert loss.requires_grad
    assert torch.isfinite(loss)

    metrics = probe.training_metrics()
    for key in (
        "solvability_confidence",
        "solvability_solve_rate",
        "solvability_brier",
    ):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_probe_is_observational():
    # Gradients must reach only the probe, never the trunk states.
    probe = SolvabilityProbe(hidden_size=32)
    hidden, logits, labels = make_batch()
    hidden.requires_grad_(True)
    loss = probe(hidden, logits, labels)
    loss.backward()
    assert hidden.grad is None or torch.all(hidden.grad == 0)
    assert any(p.grad is not None for p in probe.parameters())


def test_ema_baseline_tracks():
    probe = SolvabilityProbe(hidden_size=32)
    for _ in range(3):
        hidden, logits, labels = make_batch()
        probe(hidden, logits, labels)
    assert probe.ema_steps.item() == 3
    assert probe.loss_ema.item() > 0


def test_aligned_logits_accepted():
    # Encoder-aligned case: logits already match label length (no shift).
    probe = SolvabilityProbe(hidden_size=32)
    hidden = torch.randn(2, 8, 32)
    logits = torch.randn(2, 8, 11)
    labels = torch.randint(0, 11, (2, 8))
    loss = probe(hidden, logits, labels)
    assert torch.isfinite(loss)


def test_forward_scalar_fallback():
    """Encoder/cut-CE runs grade the batch's realized loss instead of
    per-sample CE; same metric keys, so the dashboard cards light up."""
    torch.manual_seed(0)
    probe = SolvabilityProbe(hidden_size=32)
    hidden = torch.randn(4, 16, 32)

    for step, loss_val in enumerate([2.0, 1.7, 1.4]):
        out = probe.forward_scalar(hidden, torch.tensor(loss_val))
        assert torch.isfinite(out)

    metrics = probe.training_metrics()
    assert set(metrics) == {
        "solvability_confidence",
        "solvability_solve_rate",
        "solvability_brier",
    }
    assert all(math.isfinite(v) for v in metrics.values())
    # A falling loss beats the EMA baseline: the batch reads as solved.
    assert metrics["solvability_solve_rate"] == 1.0
