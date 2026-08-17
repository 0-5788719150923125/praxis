"""Whole-sequence readout by position: the instrument for the paper's
information-density conjecture (body.tex, "Information density at the rim",
Figure fig:density).

WHAT THE CONJECTURE SAYS. Figure fig:density piles the sequence's characters
into the head cells and thins them toward the tip: the whole sequence, compressed
as far as one vector's capacity allows, is already present in the early hidden
states, and later positions add sparse detail rather than new structure. The
received picture (tokens in boxes, an arrow to the next) says the opposite: a
vector at position t knows its prefix and nothing else, so what a single vector
carries about the whole sequence grows monotonically toward the tip.

WHAT THIS MEASURES. Take the sequence as the decoder receives it - the pre-loop
hidden state, ``[T, D]`` per example - and ask, of the hidden state at ONE
position, how much of that whole sequence can be read back out of it linearly.
"The whole sequence" is taken at three resolutions, as bands of its discrete
Fourier transform along position: ``bag`` (mode 0: the mean over the window,
the bag of content), ``coarse`` (modes 1-3: window-scale structure) and ``mid``
(modes 4-7). The bands are window-relative because the decoder's window is
whatever it is - patches for a byte-latent encoder, a curriculum tier's worth
of tokens - and the conjecture speaks about the window's rim, not about
absolute token counts. The
readout is a ridge regression fit per position bucket and per depth step, and
the reading is its held-out R^2 in each band, ABOVE CHANCE: the same readout is
fit against a shuffled target (another sequence's), and the reported value is
(R^2 - R^2_null) / (1 - R^2_null) - the fraction of the way from chance to
perfect - so the finite-sample overfitting bias of a D-dimensional readout
cancels, zero means "nothing readable" and one means "fully read", at any
sample count. This is the framing the probing
literature calls a linear readout (Future Lens reads tokens k ahead from one
state; vec2text shows one vector can hold a short sequence exactly), applied
across the window instead of at one offset.

Predictions, stated so a flat or opposite reading is unmistakable:

    readout_cell_{pos}_{band}  the profile: R^2 at position bucket ``pos``
                               (head, q1..q6, tip) for band index ``band``
                               (0 bag, 1 coarse, 2 mid), at the last executed
                               depth step. Drawn as ONE heatmap - x = position,
                               y = band, shade = R^2 - which is fig:density
                               measured: is the head strip lit?
    readout_{band}_rim_gap     R^2(tip) - R^2(head). The received picture puts
                               this strongly positive - the tip has seen
                               everything, the head only its own token; for
                               the bag it is a linear rise. The conjecture
                               says the bag and coarse gaps close: the head
                               anticipates the window-scale content.
    readout_{band}_depth_gain  mean over positions of R^2(last step) -
                               R^2(entry). Whether the depth loop BUILDS
                               whole-sequence content into the states, over
                               what the raw embeddings already carry.

Causality makes the null hypothesis sharp. A causal state cannot contain later
tokens, so the only way the head can carry the whole is by anticipating it, and
anticipation is only possible for the slow structure - which is exactly the
band the conjecture claims for the head. The entry-step reading (the raw
embeddings, before any depth) is the built-in null: nothing there can carry the
whole beyond its own token, so ``depth_gain`` is what the loop added.

HOW THE READOUT IS FIT, AND WHY IT IS HONEST. One batch has too few sequences
to fit and hold out a D-dimensional readout, so the probe keeps exponentially
decayed running moments (mean, second moment, cross moment) per (bucket, step)
and solves the ridge readout from those. Every sampled forward is scored FIRST
against the readout fit from earlier batches, THEN folded into the moments -
prequential evaluation, so R^2 is always on unseen sequences. It warms up over
the first few dozen sampled forwards and tracks the model with the decay. One
batch's R^2 is a noisy draw, so the emitted readings are an EMA over sampled
forwards (``SCORE_EMA``) - a card shows a trend, not one batch. The
shuffled-null readout shares the left-hand side (same x), so both come out of
one batched solve; the null is what makes a 256-feature readout on a few
thousand sequences readable at all - without it every unpredictable target
dimension costs ~D/N of R^2 and swamps the signal.

WHERE IT RUNS. ``BaseDecoder`` owns one probe and the depth loop drives it
(``begin`` pre-loop, ``observe`` after each layer application, ``finalize``
after). It is a plain object, not an ``nn.Module``: no parameters, no
persistent buffers, nothing in ``state_dict``. The running moments live on the
object and are rebuilt from scratch after a restart.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

# Position buckets in the profile. Fixed, so the metric family has a stable
# width regardless of sequence length or curriculum stage.
BUCKETS: int = 8

# The whole-sequence target is the pre-loop state projected to this many
# features (fixed seeded Gaussian projection) and Fourier-transformed along
# position. Small on purpose: the target's width multiplies the readout cost
# and the running moments' memory.
TARGET_WIDTH: int = 8

# Resolution bands over the DFT along position, as inclusive mode ranges.
# Mode 0 is the mean over the window (the bag of content); mode k completes
# k cycles across the window, so the bands are window-relative scales, which
# is what the conjecture speaks about. The bag stands alone because it has the
# cleanest received-picture signature: a prefix predicts it in proportion to
# the prefix fraction, so it rises linearly head to tip, whereas modes 1-3 a
# prefix predicts at intermediate positions and NOT at the tip (the full
# window's sum is orthogonal to them), which would blur the two together.
BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("bag", 0, 0),
    ("coarse", 1, 3),
    ("mid", 4, 7),
)
# rfft of a T-long sequence has T//2 + 1 modes; the mid band needs mode 7. A
# byte-latent decoder's window is PATCHES (tens of them at a short curriculum
# tier), so this floor is deliberately low - a probe that only fires on long
# windows never fires on the runs it is for.
MIN_LENGTH: int = 16

# Position bucket labels, head to tip; the heatmap card reads them as its x
# axis. They sort alphabetically in this order, which the renderer relies on.
POSITION_LABELS: Tuple[str, ...] = ("head", "q1", "q2", "q3", "q4", "q5", "q6", "tip")

# Seeded so the target projection is identical across runs and architectures.
PROJECTION_SEED: int = 20260812

# Only every N-th forward is scored and folded into the moments. Keeps the
# probe at a few ms per training step amortized.
SAMPLE_EVERY: int = 4
# EMA of the emitted readings over sampled forwards; a single batch's R^2
# swings by tenths, the trend is what a card is for.
SCORE_EMA: float = 0.9
# Exponential decay of the running moments per sampled forward. 0.98 is an
# effective window of ~50 sampled forwards - a few thousand sequences.
DECAY: float = 0.98
# Ridge strength relative to the mean feature variance. Fixed: this is an
# instrument, not a model, and it must read the same for every run.
RIDGE: float = 1e-2
# Sampled forwards folded in before the readout is scored at all.
WARMUP: int = 4
# Positions read per bucket, evenly strided. The readout's sample size is
# bounded by SEQUENCES (every position in a sequence shares its target), so
# more positions per bucket cost matmul time without adding evidence.
POSITIONS_PER_BUCKET: int = 16

_EPS: float = 1e-6


class _Moments:
    """Running moments for one depth step's readouts, all buckets batched.

    ``x`` is per bucket ``[BUCKETS, N, D]``; ``y`` is the target stacked with
    its shuffled null along the last axis, ``[BUCKETS, N, 2K]``, so one solve
    with a shared left-hand side yields both readouts.
    """

    __slots__ = ("count", "mx", "my", "mxx", "mxy", "myy")

    def __init__(self) -> None:
        self.count = 0
        self.mx: Optional[Tensor] = None
        self.my: Optional[Tensor] = None
        self.mxx: Optional[Tensor] = None
        self.mxy: Optional[Tensor] = None
        self.myy: Optional[Tensor] = None

    def update(self, x: Tensor, y: Tensor) -> None:
        n = x.shape[1]
        mx = x.mean(dim=1)  # [G, D]
        my = y.mean(dim=1)  # [G, 2K]
        mxx = x.transpose(1, 2) @ x / n  # [G, D, D]
        mxy = x.transpose(1, 2) @ y / n  # [G, D, 2K]
        myy = y.square().mean(dim=1)  # [G, 2K]
        if self.mx is None:
            self.mx, self.my, self.mxx, self.mxy, self.myy = mx, my, mxx, mxy, myy
        else:
            self.mx.lerp_(mx, 1.0 - DECAY)
            self.my.lerp_(my, 1.0 - DECAY)
            self.mxx.lerp_(mxx, 1.0 - DECAY)
            self.mxy.lerp_(mxy, 1.0 - DECAY)
            self.myy.lerp_(myy, 1.0 - DECAY)
        self.count += 1

    def readout(self) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """(W [G, D, 2K], mean_x [G, D], mean_y [G, 2K], var_y [G, 2K])."""
        if self.mx is None or self.count < WARMUP:
            return None
        cov_xx = self.mxx - self.mx.unsqueeze(2) * self.mx.unsqueeze(1)
        cov_xy = self.mxy - self.mx.unsqueeze(2) * self.my.unsqueeze(1)
        var_y = (self.myy - self.my.square()).clamp_min(_EPS)
        ridge = RIDGE * cov_xx.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(_EPS)
        eye = torch.eye(cov_xx.shape[-1], device=cov_xx.device, dtype=cov_xx.dtype)
        lhs = cov_xx + ridge.view(-1, 1, 1) * eye
        try:
            weight = torch.linalg.solve(lhs, cov_xy)
        except RuntimeError:
            return None
        return weight, self.mx, self.my, var_y


class DensityProbe:
    """Whole-sequence readout R^2 (above a shuffled-target null) by position
    bucket, per depth step.

    Deliberately a plain object, not an ``nn.Module``: it holds no parameters
    and no persistent buffers, so it never touches ``state_dict`` and cannot
    change a checkpoint.
    """

    def __init__(self) -> None:
        self._projection: Optional[Tensor] = None
        self._forwards = 0
        self._active = False
        self._target: Optional[Tensor] = None  # [B, K]
        self._band_slices: List[Tuple[str, slice]] = []
        self._step = 0
        self._moments: Dict[int, _Moments] = {}
        # Per executed step this forward: {band: [BUCKETS] R^2-above-null}
        self._scores: Dict[int, Dict[str, Tensor]] = {}
        # EMA'd readings across sampled forwards: exit profile and depth gain
        # per band, on the CPU (a handful of floats; no per-call device sync).
        self._smoothed: Dict[str, Tensor] = {}
        self._metrics: Dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def begin(self, hidden_states: Tensor) -> None:
        """Reset per-forward state; build this forward's whole-sequence target
        from the pre-loop state and score the entry state as step 0."""
        self._active = False
        self._target = None
        self._scores = {}
        self._step = 0
        self._forwards += 1
        if not self._usable(hidden_states):
            return
        if self._forwards % SAMPLE_EVERY != 0:
            return
        with torch.no_grad():
            self._target = self._sequence_target(hidden_states.detach().float())
        self._active = True
        self._score_and_fold(hidden_states, step=0)

    def observe(self, hidden_states: Tensor) -> None:
        """Score one loop boundary's hidden state against the whole sequence."""
        if not self._active:
            return
        self._step += 1
        self._score_and_fold(hidden_states, step=self._step)

    def finalize(self) -> None:
        """Fold this forward's per-step scores into the running readings."""
        if not self._active or not self._scores:
            return
        last_step = max(self._scores)
        last = self._scores[last_step]
        entry = self._scores.get(0) if last_step > 0 else None
        for band, _ in self._band_slices:
            profile = last.get(band)
            if profile is None:
                continue
            self._ema(f"exit_{band}", profile)
            if entry is not None and band in entry:
                self._ema(f"gain_{band}", (profile - entry[band]).mean())
        metrics: Dict[str, float] = {}
        for index, (band, _) in enumerate(self._band_slices):
            profile = self._smoothed.get(f"exit_{band}")
            if profile is None:
                continue
            values = profile.tolist()
            for label, value in zip(POSITION_LABELS, values):
                metrics[f"readout_cell_{label}_{index}"] = value
            metrics[f"readout_{band}_rim_gap"] = values[-1] - values[0]
            gain = self._smoothed.get(f"gain_{band}")
            if gain is not None:
                metrics[f"readout_{band}_depth_gain"] = float(gain)
        if metrics:
            self._metrics = metrics

    def _ema(self, key: str, value: Tensor) -> None:
        value = value.detach().float().cpu()
        previous = self._smoothed.get(key)
        self._smoothed[key] = (
            value
            if previous is None
            else SCORE_EMA * previous + (1.0 - SCORE_EMA) * value
        )

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    # -- internals ---------------------------------------------------------

    def _usable(self, hidden_states: Tensor) -> bool:
        return (
            isinstance(hidden_states, Tensor)
            and hidden_states.dim() == 3
            and hidden_states.shape[0] >= 2
            and hidden_states.shape[1] >= MIN_LENGTH
        )

    def _sequence_target(self, states: Tensor) -> Tensor:
        """Whole-sequence target ``[B, K]``: the state projected to
        TARGET_WIDTH features, DFT'd along position, cut into BANDS."""
        projection = self._get_projection(states)
        projected = states @ projection  # [B, T, r]
        spectrum = torch.fft.rfft(projected, dim=1) / states.shape[1]  # [B, M, r]
        pieces: List[Tensor] = []
        slices: List[Tuple[str, slice]] = []
        offset = 0
        for name, lo, hi in BANDS:
            modes = spectrum[:, lo : hi + 1, :]  # [B, m, r]
            parts = [modes.real.flatten(1)]
            # Mode 0 is real by construction; its imaginary part is identically
            # zero and would be a dead target dimension.
            imag = modes[:, 1:, :] if lo == 0 else modes
            if imag.shape[1] > 0:
                parts.append(imag.imag.flatten(1))
            piece = torch.cat(parts, dim=1)
            pieces.append(piece)
            slices.append((name, slice(offset, offset + piece.shape[1])))
            offset += piece.shape[1]
        self._band_slices = slices
        return torch.cat(pieces, dim=1)

    def _score_and_fold(self, hidden_states: Tensor, step: int) -> None:
        if not self._usable(hidden_states):
            return
        target = self._target
        if target is None or hidden_states.shape[0] != target.shape[0]:
            return
        with torch.no_grad():
            states = hidden_states.detach().float()
            batch, length, width = states.shape
            per = length // BUCKETS
            k = target.shape[-1]
            # [BUCKETS, B*n, D]: bucket-major, positions past 8*per dropped,
            # up to POSITIONS_PER_BUCKET evenly strided positions per bucket.
            n = min(per, POSITIONS_PER_BUCKET)
            offsets = torch.arange(n, device=states.device) * (per // n)
            x = (
                states[:, : BUCKETS * per, :]
                .reshape(batch, BUCKETS, per, width)[:, :, offsets, :]
                .permute(1, 0, 2, 3)
                .reshape(BUCKETS, batch * n, width)
            )
            # Target beside its shuffled null: sequence i paired with sequence
            # i-1's target. The null readout carries the same overfitting bias
            # as the real one, and the reported R^2 is their difference.
            paired = torch.cat([target, target.roll(1, dims=0)], dim=1)  # [B, 2K]
            y = (
                paired.unsqueeze(1)
                .expand(batch, n, 2 * k)
                .reshape(1, batch * n, 2 * k)
                .expand(BUCKETS, -1, -1)
            )
            moments = self._moments.get(step)
            if moments is None:
                moments = _Moments()
                self._moments[step] = moments
            readout = moments.readout()
            if readout is not None:
                weight, mean_x, mean_y, var_y = readout
                predicted = mean_y.unsqueeze(1) + (x - mean_x.unsqueeze(1)) @ weight
                # Per-dimension standardized residual energy, so every target
                # dimension counts equally within its band.
                sse = (predicted - y).square().mean(dim=1) / var_y  # [G, 2K]
                sst = (y - mean_y.unsqueeze(1)).square().mean(dim=1) / var_y
                scores: Dict[str, Tensor] = {}
                for name, sl in self._band_slices:
                    null = slice(sl.start + k, sl.stop + k)
                    real = 1.0 - sse[:, sl].sum(dim=1) / sst[:, sl].sum(
                        dim=1
                    ).clamp_min(_EPS)
                    chance = 1.0 - sse[:, null].sum(dim=1) / sst[:, null].sum(
                        dim=1
                    ).clamp_min(_EPS)
                    # Fraction of the way from chance to perfect: 0 = nothing
                    # readable beyond the overfitting bias, 1 = fully read.
                    scores[name] = (real - chance) / (1.0 - chance).clamp_min(_EPS)
                self._scores[step] = scores
            moments.update(x, y)

    def _get_projection(self, reference: Tensor) -> Tensor:
        width = reference.shape[-1]
        cached = self._projection
        if (
            cached is not None
            and cached.shape[0] == width
            and cached.device == reference.device
        ):
            return cached
        generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
        projection = torch.randn(width, TARGET_WIDTH, generator=generator).to(
            device=reference.device, dtype=torch.float32
        )
        self._projection = projection
        return projection
