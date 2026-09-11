"""praxis/generators/harmonic_latent.py: the harmonic latent generator, FlowGenerator's sibling
in the ``generators`` registry."""

import torch

from praxis import registry


def test_harmonic_latent_generator():
    """Same flow_loss/forward/sample surface as FlowGenerator, but the flow lives in
    a compact harmonic coefficient space and synthesized latents lie exactly in
    the harmonic subspace."""
    assert "harmonic" in registry.namespace("generators")
    generator = registry.lookup("generators", "harmonic")(
        cond_dim=32, noise_dim=0, latent_dim=16, hidden_dim=32, num_blocks=2
    )
    assert generator.noise_dim == 16  # caller builds a latent-width start state
    assert generator.coeff_dim <= 16  # compressed (DC + low freqs)

    cond = torch.randn(2, 4, 32)
    target = torch.randn(2, 4, 16)
    loss = generator.flow_loss(target, cond).mean()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in generator.net.parameters()
    )
    with torch.no_grad():
        best = generator.forward(cond, torch.zeros(2, 4, 16))
        samples = generator.sample(cond[:, :1], num_samples=3)
    assert best.shape == (2, 4, 16)
    assert samples.shape == (3, 2, 1, 16)
    # synthesized latents are pure harmonic superpositions (idempotent project)
    assert torch.allclose(
        best, generator.synthesize(generator.project(best)), atol=1e-5
    )
