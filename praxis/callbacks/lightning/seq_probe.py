"""Lightning wiring for the probe-attribution sequence curriculum.

Owns the measurement half of :mod:`praxis.data.seq_probe`: it builds a fixed
held-out probe, re-scores it every ``window`` optimizer steps, and hands the
probe's improvement plus the window's arm mixture to the regression.

The probe is drawn ONCE from the validation data manager, one batch per
sequence-length multiplier, and frozen. Two properties matter:

* **Held out.** Scoring training batches would credit memorization of those
  batches to whichever arm produced them, which is precisely the confound the
  regression exists to remove.
* **Fixed.** The probe's loss must change only because the model changed. A
  freshly sampled probe each window would add batch-to-batch variance to the
  one quantity the fit is trying to explain.

The objective is the MEAN probe loss across lengths: "be good at every length
the curriculum can produce". An arm therefore earns credit for transferring to
other lengths, not just for improving its own - the distinction that makes this
track the objective where self-prediction gain chases the least-converged arm.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning.pytorch.callbacks import Callback

from praxis.data.seq_probe import SequenceProbe


class SequenceProbeCallback(Callback):
    """Fits the sequence-length distribution from a held-out probe."""

    # Fixed, model-agnostic constants (no per-experiment tuning).
    #
    # The window RAMPS: 16 optimizer steps, doubling to `window`. Early in
    # training the loss moves fast, so a short window already carries signal,
    # and the first cards then land ~16 steps in instead of ~128. Unequal
    # windows are not a problem for the fit - it regresses on actual visit
    # counts, and a varying window total actively helps identifiability by
    # breaking the constant-sum collinearity between the arms.
    #
    # Note the unit: OPTIMIZER steps. The dashboard's counter is
    # ``total_batch_idx`` (raw microbatches), which is larger by the
    # accumulation factor - so a card due at optimizer step 112 shows up around
    # batch 224 at accumulation 2.
    warmup_window = 16  # first window length
    window = 128  # steady-state optimizer steps between probe scorings
    probe_rows = 4  # rows per probe batch; one probe forward per arm per window

    def __init__(self, block_size: int, sequence_multiplier_tiers=()) -> None:
        super().__init__()
        self.block_size = max(1, int(block_size))
        self.tiers = tuple(sequence_multiplier_tiers)
        self._probe: List[Tuple[int, torch.Tensor]] = []
        self._visits: Dict[int, int] = {}
        self._last_loss: Optional[float] = None
        self._steps = 0
        self._in_window = 0
        self._window_len = min(self.warmup_window, self.window)
        self._aligned = False
        self._failed = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def on_train_start(self, trainer, pl_module) -> None:
        SequenceProbe.enable(self.block_size, self.tiers)
        self._aligned = bool(getattr(pl_module, "outputs_are_aligned", False))
        self._visits = {}
        self._last_loss = None
        if not self._probe:
            self._build_probe(trainer, pl_module)

    def _build_probe(self, trainer, pl_module) -> None:
        """Draw one held-out batch per arm from the validation data manager.

        A probe failure must never take training down with it, so any problem
        here disarms the controller and the pipeline falls back to the fixed
        per-tier roll.
        """
        manager = self._val_manager(trainer)
        if manager is None:
            self._disarm(
                "no validation data manager reachable from the trainer "
                "(no validation datasets configured?)"
            )
            return

        device = getattr(pl_module, "device", None)
        built: List[Tuple[int, torch.Tensor]] = []
        for m in SequenceProbe.arms:
            try:
                # get_batch applies the m**2 division itself, so ask in nominal
                # units to land exactly probe_rows rows.
                result = manager.get_batch(self.probe_rows * m * m, m)
                rows = result.get("batch")
                if not rows:
                    continue
                ids = torch.stack(rows)
                if device is not None:
                    ids = ids.to(device)
                built.append((m, ids))
            except Exception as exc:  # a probe is never worth a crash
                print(f"[SeqProbe] probe build failed for x{m}: {exc}")

        if not built:
            self._disarm("every probe batch came back empty")
            return

        self._probe = built
        shapes = ", ".join(f"x{m}:{tuple(t.shape)}" for m, t in built)
        print(
            f"[SeqProbe] probe built from held-out data ({shapes}); "
            f"seq_value/seq_tstat cards from optimizer step "
            f"~{self.first_report_step()}, seq_prob mix from "
            f"~{self.first_mix_step()} (OPTIMIZER steps - the dashboard's "
            f"batch counter is larger by the accumulation factor)"
        )

    @classmethod
    def window_lengths(cls, count: int) -> List[int]:
        """The first ``count`` window lengths under the ramp."""
        lengths, current = [], min(cls.warmup_window, cls.window)
        for _ in range(count):
            lengths.append(current)
            current = min(cls.window, current * 2)
        return lengths

    @classmethod
    def first_report_step(cls) -> int:
        """Optimizer step at which seq_value/seq_tstat first appear.

        Two windows, not one: the first scoring only anchors the probe's loss
        level, so the earliest window with a delta to regress is the second.
        """
        return sum(cls.window_lengths(2))

    @classmethod
    def first_mix_step(cls) -> int:
        """Optimizer step at which the seq_prob mix first appears.

        One extra window over ``min_windows``: the first scoring only anchors
        the probe's loss level, it produces no delta to regress.
        """
        return sum(cls.window_lengths(SequenceProbe.min_windows + 1))

    def _disarm(self, reason: str) -> None:
        """Stand down loudly. A silently inert controller looks exactly like a
        missing dashboard card, which costs more to diagnose than it saves."""
        print(
            f"[SeqProbe] DISARMED: {reason}. The sequence-length mix falls back "
            "to the fixed per-tier chances, and the seq_* dashboard cards will "
            "not appear. Set seq_curriculum=fixed to make this explicit."
        )
        SequenceProbe.reset()
        self._failed = True

    @staticmethod
    def _val_manager(trainer):
        """The validation dataset's InterleaveDataManager, if there is one."""
        dm = getattr(trainer, "datamodule", None)
        val = getattr(dm, "val_datasets", None) if dm is not None else None
        if not val:
            return None
        return getattr(val, "data_manager", None)

    # ── measurement ───────────────────────────────────────────────────────

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        """Count this microbatch against the arm that produced it.

        The multiplier is read back from the batch's own length rather than from
        the schedule, so the count reflects what the model actually consumed.
        """
        if self._failed or not SequenceProbe.enabled:
            return
        ids = batch.get("input_ids") if isinstance(batch, dict) else batch
        if not torch.is_tensor(ids) or ids.dim() < 2:
            return
        m = max(1, int(ids.size(1)) // self.block_size)
        self._visits[m] = self._visits.get(m, 0) + 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if self._failed or not SequenceProbe.enabled:
            return
        self._steps += 1
        self._in_window += 1
        if self._in_window < self._window_len:
            return
        self._in_window = 0
        # Ramp toward the steady-state window once this one closes.
        self._window_len = min(self.window, self._window_len * 2)
        loss = self._score_probe(pl_module)
        if loss is None:
            return
        if self._last_loss is not None:
            # Positive delta = the model improved over the window.
            SequenceProbe.observe_window(self._visits, self._last_loss - loss)
        self._last_loss = loss
        self._visits = {}
        self._stash(pl_module)

    @torch.no_grad()
    def _score_probe(self, pl_module) -> Optional[float]:
        """Mean held-out loss across the probe's lengths.

        Runs in eval mode with grads off; the model's training flag is restored
        afterwards so the probe cannot perturb dropout or any train-time path.
        """
        model = getattr(pl_module, "model", pl_module)
        was_training = model.training
        model.eval()
        try:
            losses = []
            for _, ids in self._probe:
                labels = (
                    ids.contiguous() if self._aligned else ids[..., 1:].contiguous()
                )
                out = model(input_ids=ids, labels=labels)
                loss = getattr(out, "loss", None)
                if loss is not None and torch.isfinite(loss):
                    losses.append(float(loss))
            return sum(losses) / len(losses) if losses else None
        except Exception as exc:
            self._disarm(f"probe scoring raised {type(exc).__name__}: {exc}")
            return None
        finally:
            model.train(was_training)

    def _stash(self, pl_module) -> None:
        """Publish onto the core model; DynamicsLogger drains it on its cadence."""
        model = getattr(pl_module, "model", pl_module)
        core = getattr(model, "_orig_mod", model)
        core._seq_probe_metrics = SequenceProbe.metrics()

    # ── resume ────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "steps": self._steps,
            "last_loss": self._last_loss,
            "xtx": [list(row) for row in SequenceProbe._xtx],
            "xty": list(SequenceProbe._xty),
            "resid_var": SequenceProbe._resid_var,
            "windows": SequenceProbe._windows,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._steps = int(state.get("steps", 0))
        xtx, xty = state.get("xtx"), state.get("xty")
        n = len(SequenceProbe.arms)
        if xtx and xty and len(xtx) == n and len(xty) == n:
            SequenceProbe._xtx = [list(row) for row in xtx]
            SequenceProbe._xty = list(xty)
            SequenceProbe._resid_var = float(state.get("resid_var", 0.0))
            SequenceProbe._windows = int(state.get("windows", 0))
            SequenceProbe._solve()
            if SequenceProbe._windows >= SequenceProbe.min_windows:
                SequenceProbe._recompute()
        # The probe tensor itself is not checkpointed - a fresh draw from
        # held-out data is equivalent. But its loss LEVEL differs from the old
        # probe's, so the anchor must be dropped: the first window after a
        # resume would otherwise regress a level shift onto that window's arms.
        self._last_loss = None
