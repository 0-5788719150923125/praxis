"""praxis/heads/harmonic_latent.py: the harmonic latent head, FlowHead's sibling
in the ``latent_heads`` registry."""

import torch

from praxis import registry


def test_harmonic_latent_head():
    """Same flow_loss/forward/sample surface as FlowHead, but the flow lives in
    a compact harmonic coefficient space and synthesized latents lie exactly in
    the harmonic subspace."""
    assert "harmonic" in registry.namespace("latent_heads")
    head = registry.lookup("latent_heads", "harmonic")(
        cond_dim=32, noise_dim=0, latent_dim=16, hidden_dim=32, num_blocks=2
    )
    assert head.noise_dim == 16  # caller builds a latent-width start state
    assert head.coeff_dim <= 16  # compressed (DC + low freqs)

    cond = torch.randn(2, 4, 32)
    target = torch.randn(2, 4, 16)
    loss = head.flow_loss(target, cond).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.net.parameters()
    )
    with torch.no_grad():
        best = head.forward(cond, torch.zeros(2, 4, 16))
        samples = head.sample(cond[:, :1], num_samples=3)
    assert best.shape == (2, 4, 16)
    assert samples.shape == (3, 2, 1, 16)
    # synthesized latents are pure harmonic superpositions (idempotent project)
    assert torch.allclose(best, head.synthesize(head.project(best)), atol=1e-5)
