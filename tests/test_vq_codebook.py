"""The residual quantizer's codebook starts usable and stays in use: the EMA
state starts equal to the codebook, so an update moves only the codes that
received vectors, and each stage is seeded from the first training batch, so no
code starts far from the data where the few nearest would absorb every vector.
"""

import torch

from praxis.encoders.quantization.vector_quantizer import (
    MultiStageResidualVQ,
    VectorQuantizer,
)

K, D = 64, 16


def _clustered(n=256, clusters=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(clusters, D, generator=g) * 2

    def draw():
        z = centers[torch.randint(0, clusters, (n,))] + 0.3 * torch.randn(n, D)
        return (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).view(1, n, D)

    return draw


def test_ema_state_starts_as_the_codebook():
    vq = VectorQuantizer(K=K, D=D, decay=0.999)
    ratio = vq.ema_weight_sum / vq.ema_cluster_size.unsqueeze(1)
    torch.testing.assert_close(ratio, vq.codebook.detach())


def test_one_update_leaves_unused_codes_where_they_were():
    torch.manual_seed(0)
    vq = VectorQuantizer(K=K, D=D, decay=0.999, reset_codes=False).train()
    before = vq.codebook.detach().clone()
    with torch.no_grad():
        _, _, idx, _ = vq(torch.randn(1, 8, D))
    unused = torch.ones(K, dtype=torch.bool)
    unused[idx.flatten()] = False
    assert unused.any()
    # Laplace smoothing nudges them by ~1e-5, nothing more.
    torch.testing.assert_close(vq.codebook[unused], before[unused], rtol=1e-3, atol=0)


def test_codebooks_are_seeded_from_the_first_training_batch_only():
    torch.manual_seed(0)
    rvq = MultiStageResidualVQ(K=K, D=D, depth=2).train()
    reference = MultiStageResidualVQ(K=K, D=D, depth=2).train()
    reference.load_state_dict(rvq.state_dict())
    reference.seeded.fill_(True)
    first = _clustered()()

    out = rvq(first)[0]
    # Seeding runs after quantization: this batch saw the unseeded codebook.
    torch.testing.assert_close(out, reference(first)[0])
    assert bool(rvq.seeded)
    # Every stage-0 code is (within the jitter) a vector of that batch.
    flat = first.reshape(-1, D)
    gap = torch.cdist(rvq.stages[0].codebook.detach(), flat).min(dim=1).values
    assert gap.max() < 0.05

    seeded = rvq.stages[0].codebook.detach().clone()
    with torch.no_grad():
        rvq.eval()(first)
    torch.testing.assert_close(rvq.stages[0].codebook.detach(), seeded)


def test_a_checkpoint_from_before_seeding_is_not_reseeded():
    rvq = MultiStageResidualVQ(K=K, D=D, depth=2)
    state = {k: v for k, v in rvq.state_dict().items() if k != "seeded"}
    fresh = MultiStageResidualVQ(K=K, D=D, depth=2)
    fresh.load_state_dict(state)
    assert bool(fresh.seeded)


def test_the_codebook_stays_in_use_on_clustered_latents():
    torch.manual_seed(0)
    draw = _clustered()
    rvq = MultiStageResidualVQ(K=K, D=D, depth=2, decay=0.999).train()
    with torch.no_grad():
        for _ in range(100):
            rvq(draw())
        z = draw()
        z_hat, _, _, _ = rvq.eval()(z)
        used = rvq.stages[0].nearest(z.reshape(-1, D)).unique().numel()
    assert used >= 16  # at least one code per cluster
    error = ((z - z_hat).pow(2).sum(-1) / z.pow(2).sum(-1)).mean()
    assert error < 0.2
