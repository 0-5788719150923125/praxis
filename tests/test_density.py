import torch

from praxis.metrics.density import BUCKETS, DensityProbe


def _trajectory(probe, states):
    """Drive the probe over a list of [B, T, D] states and read the result."""
    probe.begin(states[0])
    for state in states[1:]:
        probe.observe(state)
    probe.finalize()
    return probe.get_metrics()


def _ramp(length, head, tip):
    """Per-position scale rising linearly from ``head`` to ``tip``."""
    return torch.linspace(head, tip, length).view(1, length, 1)


def test_flat_trajectory_reads_flat():
    """The falsifying case has to actually read as falsified: a trajectory that
    moves every position equally must not manufacture a positional gradient."""
    torch.manual_seed(0)
    base = torch.randn(4, 64, 32)
    states = [base + 0.1 * torch.randn(4, 64, 32) for _ in range(5)]

    metrics = _trajectory(DensityProbe(), states)
    assert abs(metrics["density_norm_slope"]) < 0.1
    profile = [metrics[f"density_norm_b{b}"] for b in range(BUCKETS)]
    assert max(profile) - min(profile) < 0.2


def test_tip_heavy_trajectory_reads_positive():
    """Density at the rim: positions late in the window move more per step."""
    torch.manual_seed(0)
    base = torch.randn(4, 64, 32)
    scale = _ramp(64, 0.02, 0.6)
    states = [base]
    for _ in range(4):
        states.append(states[-1] + scale * torch.randn(4, 64, 32))

    metrics = _trajectory(DensityProbe(), states)
    assert metrics["density_norm_slope"] > 0.5
    assert metrics["density_hop_slope"] > 0.0
    # b7 is the tip, b0 the head.
    assert metrics["density_norm_b7"] > metrics["density_norm_b0"]


def test_steepening_across_depth():
    """The second half of the falsifier: the tilt must GROW with each pass."""
    torch.manual_seed(0)
    base = torch.randn(4, 64, 32)
    states = [base]
    for step in range(1, 5):
        # Each pass tilts harder than the last.
        scale = _ramp(64, 0.02, 0.1 * step**2)
        states.append(states[-1] + scale * torch.randn(4, 64, 32))

    metrics = _trajectory(DensityProbe(), states)
    assert metrics["density_norm_steepening"] > 0.0


def test_occupancy_catches_what_norm_misses():
    """The paper's stated reason for a second coordinate: 'a single bit's worth
    of change can move the hidden state from one basin to another while
    remaining silent in norm'.

    Constructed directly - every position gets an identically sized
    perturbation, so the NORM profile is flat by design, but tip positions sit
    on the partition boundaries (state orthogonal to every hyperplane) while
    head positions sit far from them. Only occupancy should see it.
    """
    torch.manual_seed(0)
    batch, length, width = 4, 64, 64
    probe = DensityProbe()

    reference = torch.randn(batch, length, width)
    projection = probe._get_projection(reference)  # [width, BITS]
    basis, _ = torch.linalg.qr(projection)  # orthonormal span of the hyperplanes

    # Head: inside the span, so projections are large and a nudge cannot flip
    # a sign. Tip: orthogonal to the span, so every projection sits at ~0 and
    # the same nudge crosses boundaries.
    coeffs = torch.randn(batch, length, basis.shape[1])
    in_span = coeffs @ basis.t()
    off_span = torch.randn(batch, length, width)
    off_span = off_span - (off_span @ basis) @ basis.t()

    weight = torch.linspace(0.0, 1.0, length).view(1, length, 1)
    previous = (1 - weight) * in_span + weight * off_span
    # Equal magnitude at every position, so the probe's per-position
    # standardization scales the nudge identically head to tip. Without this
    # the norm profile tilts for a reason that has nothing to do with the
    # partition, and the test would be measuring its own construction.
    previous = previous / previous.norm(dim=-1, keepdim=True)

    # One identically sized perturbation for every position.
    nudge = torch.randn(batch, length, width)
    nudge = 0.05 * nudge / nudge.norm(dim=-1, keepdim=True)
    current = previous + nudge

    metrics = _trajectory(probe, [previous, current])

    # Norm is flat by construction; occupancy is not.
    assert abs(metrics["density_norm_slope"]) < 0.25
    assert metrics["density_hop_slope"] > 0.5
    assert metrics["density_hop_b7"] > 3 * metrics["density_hop_b0"]


def test_degenerate_inputs_emit_nothing():
    """No depth loop, a 2-D state, or a compressor changing sequence length
    mid-loop must all be silent rather than fatal."""
    probe = DensityProbe()
    probe.begin(torch.randn(2, 16, 8))
    probe.finalize()
    assert probe.get_metrics() == {}  # single boundary -> no transition

    probe = DensityProbe()
    probe.begin(torch.randn(2, 8))  # not [B, T, D]
    probe.observe(torch.randn(2, 8))
    probe.finalize()
    assert probe.get_metrics() == {}

    probe = DensityProbe()
    probe.begin(torch.randn(2, 16, 8))
    probe.observe(torch.randn(2, 4, 8))  # sequence length changed
    probe.finalize()
    assert probe.get_metrics() == {}


def test_probe_holds_no_state_dict():
    """It must never be able to change a checkpoint."""
    import torch.nn as nn

    from praxis.metrics.density import DensityProbe as Probe

    module = nn.Module()
    module.density = Probe()
    assert module.state_dict() == {}
