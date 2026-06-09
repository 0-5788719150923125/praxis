"""Learned curriculum over sequence-length multipliers.

The data pipeline trades batch size for sequence length at constant token
count (see ``SEQUENCE_MULTIPLIER_TIERS`` in manager.py): a multiplier ``m``
yields ``block_size*m`` length, ``batch_size//m**2`` rows. Today the tier is
rolled from fixed per-tier chances. This replaces that roll, when enabled,
with an adaptive distribution the *training dynamics* control: a bandit over
the multipliers driven by **learning progress** - sample more of the length
the model is currently improving fastest on.

Why learning progress (loss decrease rate) and not raw loss: mean per-token
loss is lower for longer sequences purely because more positions have more
context, which would bias the controller toward long regardless of value.
Progress is the *change* over time within a multiplier, so the absolute
level cancels - it measures "which length is teaching the model the most
right now" (Graves et al., automated curriculum learning).

Cross-process note: like the loss/tasker sampling feedback, the controller's
distribution is class-level state. The trainer updates it (``observe``); the
data sampler reads it (``sample``). This propagates within one process
(num_workers=0, the CUDA/spawn default); a forked worker sees the snapshot it
was forked with - the same limitation the loss/tasker modes already have.
"""

import math
from typing import Dict, List, Optional


class SequenceCurriculum:
    """Learning-progress bandit over sequence-length multipliers."""

    # Fixed, model-agnostic constants (no per-experiment tuning).
    ema_alpha = 0.1  # smoothing for per-arm loss + progress EMAs
    temperature = 1.0  # softmax temp over z-scored progress
    explore = 0.1  # uniform floor so every arm keeps being sampled (and re-estimated)

    enabled: bool = False
    block_size: int = 0
    arms: List[int] = []  # eligible multipliers, e.g. [1, 2, 4, 8]
    _loss_ema: Dict[int, float] = {}
    _progress: Dict[int, float] = {}
    shared_probs: Optional[Dict[int, float]] = None  # consumed by sample()

    @classmethod
    def enable(cls, block_size: int, tiers) -> None:
        """Arm the controller. ``arms`` = {1} plus every tier multiplier."""
        cls.enabled = True
        cls.block_size = int(block_size)
        cls.arms = sorted({1} | {int(m) for m, _ in tiers})
        cls._loss_ema = {}
        cls._progress = {}
        cls.shared_probs = None  # stays None until the first observation
        print(
            f"[SeqCurriculum] adaptive sequence-length curriculum on; "
            f"arms={cls.arms} (block_size={cls.block_size})"
        )

    @classmethod
    def reset(cls) -> None:
        cls.enabled = False
        cls.block_size = 0
        cls.arms = []
        cls._loss_ema = {}
        cls._progress = {}
        cls.shared_probs = None

    @classmethod
    def observe(cls, seq_len: int, loss: float) -> None:
        """Feed a batch's sequence length and loss back to the controller."""
        if not cls.enabled or cls.block_size <= 0:
            return
        m = max(1, int(seq_len) // cls.block_size)
        if m not in cls.arms:
            return  # unknown length (e.g. fixed-length validation) - ignore
        prev = cls._loss_ema.get(m)
        if prev is None:
            cls._loss_ema[m] = float(loss)  # seed; no progress yet
        else:
            progress = prev - float(loss)  # positive = loss dropped on this arm
            cls._loss_ema[m] = cls.ema_alpha * loss + (1 - cls.ema_alpha) * prev
            cls._progress[m] = (
                cls.ema_alpha * progress
                + (1 - cls.ema_alpha) * cls._progress.get(m, 0.0)
            )
            cls._recompute()

    @classmethod
    def _recompute(cls) -> None:
        """Softmax over z-scored per-arm progress, with a uniform floor.

        Z-scoring makes the distribution respond to *relative* progress
        regardless of the absolute loss-delta scale, so no temperature tuning
        is needed per model/experiment."""
        prog = [cls._progress.get(m, 0.0) for m in cls.arms]
        n = len(prog)
        mean = sum(prog) / n
        std = (sum((p - mean) ** 2 for p in prog) / n) ** 0.5
        z = [(p - mean) / (std + 1e-8) for p in prog]
        mx = max(z)
        exps = [math.exp((zi - mx) / cls.temperature) for zi in z]
        s = sum(exps)
        probs = [e / s for e in exps]
        u = 1.0 / n
        probs = [(1 - cls.explore) * p + cls.explore * u for p in probs]
        cls.shared_probs = dict(zip(cls.arms, probs))

    @classmethod
    def sample(cls, batch_size: int, tiers, rng) -> Optional[int]:
        """Pick a multiplier from the learned distribution, restricted to tiers
        eligible for this batch size. Returns None when not armed yet so the
        caller falls back to the fixed-chance roll."""
        if not cls.enabled or cls.shared_probs is None:
            return None
        eligible = [m for m in cls.arms if batch_size >= m * m]
        if not eligible:
            return 1
        weights = [cls.shared_probs.get(m, 0.0) for m in eligible]
        total = sum(weights)
        if total <= 0:
            return 1
        r = rng.random() * total
        acc = 0.0
        for m, w in zip(eligible, weights):
            acc += w
            if r <= acc:
                return m
        return eligible[-1]

    @classmethod
    def metrics(cls) -> Dict[str, float]:
        """Per-arm sampling probability + learning progress, for logging."""
        if not cls.enabled or cls.shared_probs is None:
            return {}
        out: Dict[str, float] = {}
        for m in cls.arms:
            out[f"seq_prob_x{m}"] = float(cls.shared_probs.get(m, 0.0))
            out[f"seq_progress_x{m}"] = float(cls._progress.get(m, 0.0))
        return out
