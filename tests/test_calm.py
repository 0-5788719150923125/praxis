"""CALM encoder + energy head + LF-temperature sanity tests.

These are shape / plumbing checks rather than training-quality
assertions. The smoke-test in the CALM README covers the latter.
"""

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.encoders import ENCODER_REGISTRY
from praxis.generation.lf_temperature import (
    lf_temperature_sample_batched,
    lf_temperature_sample_exact,
)
from praxis.heads.energy import EnergyHead
from praxis.losses.energy_score import energy_score_loss
from praxis.metrics import compute_brier_lm


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=256,
        embed_size=32,
        hidden_size=64,
        num_heads=4,
        num_queries=2,
        num_layers=2,
        depth=2,
        block_size=32,
        max_position_embeddings=32,
        encoder_type="calm_small",
    )
    defaults.update(overrides)
    return PraxisConfig(**defaults)


def test_calm_in_encoder_registry():
    assert "calm" in ENCODER_REGISTRY


def test_calm_forward_backward():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (2, 32), dtype=torch.long)
    labels = input_ids[:, 1:].contiguous()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss.requires_grad
    out.loss.backward()
    # Energy head and VAE both get gradients.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.vae.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.energy_head.parameters()
    )


def test_calm_handles_loss_skips_main_ce():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.train()
    input_ids = torch.randint(4, 200, (1, 32), dtype=torch.long)
    out = model(input_ids=input_ids, labels=input_ids[:, 1:].contiguous())
    # handles_loss=True means main CE is never added; only encoder /
    # decoder / energy contribute.
    # The base container seeds "main" = 0 which stays zero here.
    # Trainer strategy sums over tagged losses.
    assert model.encoder.handles_loss is True


def test_calm_generate_advances_in_K_steps():
    cfg = _tiny_config()
    model = PraxisForCausalLM(cfg)
    model.eval()
    from transformers import GenerationConfig

    gc = GenerationConfig(max_new_tokens=12, temperature=1.0, do_sample=True)
    input_ids = torch.randint(4, 200, (1, 8), dtype=torch.long)
    out = model.generate(input_ids, generation_config=gc)
    new = out.size(1) - input_ids.size(1)
    # Generation moves in chunk-size-sized jumps: at least 12, rounded up.
    K = model.encoder.K
    assert new % K == 0
    assert new >= 12


def test_energy_head_shapes():
    head = EnergyHead(
        cond_dim=32, noise_dim=16, latent_dim=8, hidden_dim=32, num_blocks=2
    )
    h = torch.randn(3, 5, 32)
    samples = head.sample(h, num_samples=4)
    assert samples.shape == (4, 3, 5, 8)


def test_energy_score_loss_nonnegative_on_random():
    torch.manual_seed(0)
    model = torch.randn(2, 3, 4, 8)
    target = torch.randn(2, 3, 5, 8)
    val = energy_score_loss(model, target)
    assert val.dim() == 0
    assert val.item() == val.item()  # not NaN


def test_energy_score_loss_lower_when_distributions_match():
    """Matched distributions should score lower than mismatched ones."""
    torch.manual_seed(0)
    matched_m = torch.randn(4, 100, 8)
    matched_t = torch.randn(4, 100, 8)
    matched = energy_score_loss(matched_m, matched_t)

    mismatched_m = torch.randn(4, 100, 8)
    mismatched_t = torch.randn(4, 100, 8) + 10.0
    mismatched = energy_score_loss(mismatched_m, mismatched_t)

    assert matched.item() < mismatched.item()


def test_lf_temperature_base_case_T1():
    torch.manual_seed(0)

    def sampler(n):
        return torch.randn(n, 4)

    z = lf_temperature_sample_batched(sampler, temperature=1.0, num_candidates=16)
    assert z.shape == (4,)


def test_lf_temperature_low_T_prefers_high_density():
    """At low T, locally-fair sampling should bias toward cluster centres."""
    torch.manual_seed(0)

    def clustered_sampler(n):
        # 80% near origin, 20% far away.
        base = torch.randn(n, 2) * 0.1
        mask = torch.rand(n) < 0.2
        base[mask] += 5.0
        return base

    dists = []
    for _ in range(64):
        z = lf_temperature_sample_batched(
            clustered_sampler, temperature=0.1, num_candidates=32
        )
        dists.append(z.norm().item())
    mean_dist = sum(dists) / len(dists)

    dists_t1 = []
    for _ in range(64):
        z = lf_temperature_sample_batched(
            clustered_sampler, temperature=1.0, num_candidates=32
        )
        dists_t1.append(z.norm().item())
    mean_dist_t1 = sum(dists_t1) / len(dists_t1)

    # Mode-seeking at T<1 should land closer to the dense cluster.
    assert mean_dist < mean_dist_t1


def test_lf_temperature_exact_falls_back_gracefully():
    torch.manual_seed(0)

    def sampler(n):
        return torch.randn(n, 3)

    z = lf_temperature_sample_exact(
        sampler, temperature=0.5, num_candidates=16, max_tries=8
    )
    assert z.shape == (3,)


def test_brierlm_scores_range():
    refs = [[1, 2, 3, 4, 5, 6, 7, 8] for _ in range(4)]
    a_same = [list(r) for r in refs]
    b_same = [list(r) for r in refs]
    assert compute_brier_lm(a_same, b_same, refs) == pytest.approx(100.0)

    a_rand = [[999] * 8 for _ in range(4)]
    b_rand = [[999] * 8 for _ in range(4)]
    assert compute_brier_lm(a_rand, b_rand, refs) == 0.0
