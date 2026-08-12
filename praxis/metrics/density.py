"""Information-density profile: the instrument for the paper's rim conjecture.

The claim (body.tex, "Information density at the rim") is that a harmonic latent
pushes density - the share of the representation's distinguishing content
carried at each position - to the extremes of the window: a slow hum at the
head, an input-conditional chirp at the tip. The paper states its own falsifier
precisely, and this module measures exactly that and nothing more:

    "hidden-state deviation must grow with recurrent depth under a monotone
    positional gradient - early positions settling toward a fixed point while
    the tip stays live [...] the claim is falsified the moment the
    per-position, per-step deviation profile - read in norm and in symbol
    occupancy alike, since a silent bit flip can change the geometry without
    moving the norm - is measured flat in position, or fails to steepen across
    depth."

So there are two predictions and two coordinate systems, and all four readings
are emitted:

    density_norm_slope        deviation rises head -> tip          (> 0)
    density_norm_steepening   that rise steepens with depth        (> 0)
    density_hop_slope         same, in symbol-occupancy            (> 0)
    density_hop_steepening    same                                 (> 0)

A slope at zero in BOTH coordinates, or slopes that fail to steepen, kills the
reading. The paper is explicit that "flat in norm AND flat in occupancy leaves
it nothing to hide behind" - which is why a norm-only probe would not have been
enough, and why this one is not.

WHAT "DEVIATION" IS. Between successive loop boundaries r-1 and r, per position:
the distance the hidden state moved. Both states are standardized over the
feature axis first (shift/scale invariant), the same trick
``praxis/halting/kl.py`` uses so the measure does not drift as residual norms
grow during training. Each per-step profile is then divided by its own mean, so
the slope is a dimensionless "fractional rise per unit of normalized position" -
comparable across architectures, widths and runs, which is the point of putting
this in a general module rather than in one experiment.

WHAT "OCCUPANCY" IS, AND HOW IT DIFFERS FROM THE PAPER'S LETTER. The paper asks
which cluster of the *learned codebook* a state expresses. That is only defined
for models carrying a codebook, so as a general instrument this uses a fixed
random hyperplane signature (SimHash) instead: which cell of a random partition
the state falls in, and the rate at which positions cross a boundary between
depth steps. It shares the property that motivated the second coordinate - a
change of cell registers at full weight however small the norm movement - but it
reads a *geometric* partition, not a learned-semantic one. A codebook-backed
version, for models that have one, is the stricter test and is not this.

WHERE IT RUNS. ``BaseDecoder`` owns one probe and the depth loop drives it, so
every decoder with a recurrence gets the profile for free and architectures
without one simply emit nothing. Cost is one [D, BITS] matmul and a handful of
reductions per loop boundary - negligible beside the block itself, and the same
always-on shape as the existing ``depth_prints`` accounting.
"""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

# Position buckets in the profile. Fixed, so the metric family has a stable
# width regardless of sequence length or curriculum stage.
BUCKETS: int = 8

# Hyperplanes in the occupancy signature. 16 bits is plenty to make a boundary
# crossing detectable while staying free next to the block's own compute.
BITS: int = 16

# Seeded so the partition is identical across runs and architectures - the
# occupancy numbers would otherwise not be comparable between them.
PROJECTION_SEED: int = 20260812

_EPS: float = 1e-6


def _standardize(hidden_states: Tensor) -> Tensor:
    """Shift/scale invariant over the feature axis, matching KL halting."""
    mean = hidden_states.mean(dim=-1, keepdim=True)
    std = hidden_states.std(dim=-1, keepdim=True).clamp_min(_EPS)
    return (hidden_states - mean) / std


def _bucketize(profile: Tensor) -> Tensor:
    """Reduce a per-position [T] profile to BUCKETS means over position."""
    length = profile.shape[0]
    if length <= BUCKETS:
        return profile
    edges = torch.linspace(0, length, BUCKETS + 1, device=profile.device).long()
    return torch.stack(
        [profile[edges[b] : edges[b + 1]].mean() for b in range(BUCKETS)]
    )


def _slope(values: Tensor) -> float:
    """Least-squares slope of ``values`` against position normalized to [0, 1].

    Returns the rise across the whole window, so it reads directly as "the tip
    carries N times more than the mean relative to the head".
    """
    n = values.numel()
    if n < 2:
        return 0.0
    x = torch.linspace(0.0, 1.0, n, device=values.device, dtype=values.dtype)
    x_centered = x - x.mean()
    denom = x_centered.square().sum()
    if float(denom) < _EPS:
        return 0.0
    return float((x_centered * (values - values.mean())).sum() / denom)


class DensityProbe:
    """Accumulates the per-position, per-depth-step deviation profile.

    Deliberately a plain object, not an ``nn.Module``: it holds no parameters
    and no persistent buffers, so it never touches ``state_dict`` and cannot
    change a checkpoint. The projection is cached per (device, dtype, width) on
    first use.
    """

    def __init__(self) -> None:
        self._projection: Optional[Tensor] = None
        self._previous: Optional[Tensor] = None
        self._norm_profiles: List[Tensor] = []
        self._hop_profiles: List[Tensor] = []
        self._metrics: Dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def begin(self, hidden_states: Tensor) -> None:
        """Reset per-forward state and stash the pre-loop hidden state."""
        self._previous = None
        self._norm_profiles = []
        self._hop_profiles = []
        if self._usable(hidden_states):
            self._previous = _standardize(hidden_states.detach().float())

    def observe(self, hidden_states: Tensor) -> None:
        """Record one loop boundary's deviation against the previous one."""
        if not self._usable(hidden_states):
            return
        current = _standardize(hidden_states.detach().float())
        previous, self._previous = self._previous, current
        # A compressor can change sequence length mid-loop; positions no longer
        # correspond, so that boundary contributes nothing rather than garbage.
        if previous is None or previous.shape != current.shape:
            return

        delta = current - previous
        # Norm coordinate: per-position distance moved, averaged over batch.
        norm = delta.square().mean(dim=-1).sqrt().mean(dim=0)  # [T]
        self._norm_profiles.append(_bucketize(norm))

        # Occupancy coordinate: fraction of hyperplanes the position crossed.
        projection = self._get_projection(current)
        before = (previous @ projection) > 0
        after = (current @ projection) > 0
        hop = (before ^ after).float().mean(dim=-1).mean(dim=0)  # [T]
        self._hop_profiles.append(_bucketize(hop))

    def finalize(self) -> None:
        """Turn the recorded profiles into the four readings plus the charts."""
        metrics: Dict[str, float] = {}
        for name, profiles in (
            ("norm", self._norm_profiles),
            ("hop", self._hop_profiles),
        ):
            if not profiles:
                continue
            # Normalize each step's profile by its own mean, so the slope is
            # dimensionless and the steepening is not just "everything moved
            # more this pass".
            relative = [p / p.mean().clamp_min(_EPS) for p in profiles]
            slopes = [_slope(p) for p in relative]

            metrics[f"density_{name}_slope"] = sum(slopes) / len(slopes)
            if len(slopes) >= 2:
                # Does the positional gradient STEEPEN as the loop deepens?
                by_step = torch.tensor(slopes, dtype=torch.float32)
                metrics[f"density_{name}_steepening"] = _slope(by_step)

            profile = torch.stack(relative).mean(dim=0)
            for b, value in enumerate(profile.tolist()):
                metrics[f"density_{name}_b{b}"] = value

        if metrics:
            self._metrics = metrics

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    # -- internals ---------------------------------------------------------

    def _usable(self, hidden_states: Tensor) -> bool:
        return (
            isinstance(hidden_states, Tensor)
            and hidden_states.dim() == 3
            and hidden_states.shape[1] >= 2
        )

    def _get_projection(self, reference: Tensor) -> Tensor:
        width = reference.shape[-1]
        cached = self._projection
        if (
            cached is not None
            and cached.shape[0] == width
            and cached.device == reference.device
            and cached.dtype == reference.dtype
        ):
            return cached
        generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
        projection = torch.randn(width, BITS, generator=generator).to(
            device=reference.device, dtype=reference.dtype
        )
        self._projection = projection
        return projection
