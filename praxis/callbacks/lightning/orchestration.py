"""ExpertPoolCallback: drive the remote-expert pool during training.

Brings the orchestration layer online for a run: starts the Node sidecar (so the
pool has backend experts before any browser connects), then each step syncs
membership (picking up browser-joined experts), runs a non-blocking local update
across the pool, and routes a vote through it. This keeps the shared pool status
fresh so the ``remote_layers`` count and the expert-pool capacity line grow on
both the CLI dashboard and the web Terminal tab as peers join.

OBSERVER MODE (current branch). The pool runs in passive observer mode: experts
train locally on real batches but their outputs do NOT feed back into the main
model's activations or loss. This is the safe, fully-observable swarm - the
browser shows agents as OBSERVE (blue) while they're fed, decaying to IDLE when
not.

STUB - contribute-back seam (next step, intentionally not implemented):
    The integration point that would make this an actual model layer is a
    decoder ``RemoteLayer`` (praxis/layers/remote.py) calling ``pool.infer()`` on
    the *real* hidden states and mixing the vote back into the residual stream -
    at which point contributing agents become TRAINING (not just OBSERVE). Until
    that seam is drawn, the pool is observer-only and the model is untouched.
"""

from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn

from praxis.orchestration import ExpertPool, LocalExpert, SidecarManager
from praxis.orchestration import status as pool_status

# Browser agents train on short id sequences over the tiny swarm vocab; the
# backend publishes real batches downsampled to this shape (matches swarm.js SEQ).
SEQ = 8


def _make_local_expert(uid: str, dim: int, vocab: int) -> LocalExpert:
    """A dependency-free in-process expert (one tiny block + projection)."""
    block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
    return LocalExpert(uid, block, hidden_size=dim, vocab_size=vocab, lr=1e-2)


class ExpertPoolCallback(Callback):
    def __init__(
        self,
        mixing: str = "vote",
        sidecar: bool = False,
        port: int = 7777,
        init_experts: int = 4,
        dim: int = 14,
        vocab: int = 16,
        sync_every: int = 25,
        drive_every: int = 5,
        metrics_every: int = 50,
    ) -> None:
        super().__init__()
        self.pool = ExpertPool([], mixing=mixing)
        self.dim = dim
        self.vocab = vocab
        self.init_experts = max(0, init_experts)
        self.sync_every = max(1, sync_every)
        self.drive_every = max(1, drive_every)
        self.metrics_every = max(1, metrics_every)
        # The Node sidecar is optional extra peers; the pool's baseline experts
        # are in-process so the count is real even without Node.
        self.manager: Optional[SidecarManager] = (
            SidecarManager(
                self.pool, port=port, init_experts=init_experts, dim=dim, vocab=vocab
            )
            if sidecar
            else None
        )
        # Fixed random embedding: real token ids -> the experts' tiny dim. Frozen
        # and shared, so every expert sees the same compressed view of the REAL
        # training data (they still diverge via their own weights/init). This is
        # how an observer learns from real batches without coupling to the main
        # model's hidden_size.
        self._embed: Optional[nn.Embedding] = None
        self._started = False

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.local_rank != 0 or self._started:
            return
        self._started = True
        # Size the input embedding to the model's real vocab so it can ingest
        # actual token ids; keep it frozen (it's a fixed projection, not learned).
        model_vocab = int(
            getattr(getattr(pl_module, "config", None), "vocab_size", 0) or 0
        )
        self._embed_vocab = max(self.vocab, model_vocab, 256)
        self._embed = nn.Embedding(self._embed_vocab, self.dim)
        self._embed.weight.requires_grad_(False)
        # Seed the pool with in-process experts (real, dependency-free), then
        # register it so web routes can grow it on a browser join.
        for i in range(self.init_experts):
            self.pool.add(_make_local_expert(f"expert-{i}", self.dim, self.vocab))
        # Tell the join route what size to make browser-joined experts.
        self.pool._join_dim = self.dim
        self.pool._join_vocab = self.vocab
        pool_status.register_pool(self.pool)
        if self.manager is not None:
            self.manager.start()
        self.pool.capacity()  # publish initial status

    def _embed_batch(self, batch):
        """Turn a real training batch (token ids) into (activations, targets) for
        the tiny experts: embed ids -> [B, T-1, dim] inputs, next ids as targets.
        Returns None if the batch isn't usable."""
        ids = batch["input_ids"] if isinstance(batch, dict) else batch
        if ids is None or ids.dim() != 2 or ids.shape[1] < 2:
            return None
        ids = ids.detach().to("cpu").long().clamp_(0, self._embed_vocab - 1)
        inp, tgt = ids[:, :-1], ids[:, 1:]
        # Targets must fall in the experts' own (smaller) vocab.
        tgt = tgt.clamp(max=self.vocab - 1)
        with torch.no_grad():
            acts = self._embed(inp)
        return acts, tgt

    def _publish_browser_batch(self, batch) -> None:
        """Publish one real batch as token-id rows for browser agents to train on.

        Browser agents carry their own embedding over the swarm's tiny vocab, so
        they want *ids* (not activations), folded into [0, vocab) and capped to a
        short sequence. Bounded depth 1 (newest wins) - the queue can never grow.
        """
        ids = batch["input_ids"] if isinstance(batch, dict) else batch
        if ids is None or getattr(ids, "dim", lambda: 0)() != 2 or ids.shape[1] < 2:
            return
        # One row, capped length, folded into the tiny vocab via modulo (a stable
        # surjection - real token structure survives, just wrapped).
        row = ids[0, : SEQ + 1].detach().to("cpu").long()
        row = (row % self.vocab).tolist()
        if len(row) < 2:
            return
        pool_status.publish_batch(row[:-1], row[1:])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.local_rank != 0:
            return
        # Pick up sidecar/browser-joined experts periodically.
        if self.manager is not None and batch_idx % self.sync_every == 0:
            self.manager.sync()
        # Drive the pool on the REAL training batch: embed the actual token ids
        # into the experts' tiny dim and run a detached local update + vote. Each
        # expert folds its own loss/accuracy EMAs along the way. (Observer mode:
        # gradients never leave an expert; the model is untouched.)
        if (
            batch_idx % self.drive_every == 0
            and self.pool.alive()
            and self._embed is not None
        ):
            embedded = self._embed_batch(batch)
            if embedded is not None:
                acts, tgt = embedded
                try:
                    self.pool.train_step(acts, tgt, timeout=2.0)
                    self.pool.infer(acts)
                except Exception:
                    pass
        # Publish the real batch (as ids) for browser agents - every drive step,
        # newest-wins, so remote peers train on the actual data the model sees.
        if batch_idx % self.drive_every == 0:
            try:
                self._publish_browser_batch(batch)
            except Exception:
                pass
        # Sample the pool cheaply at logging intervals (read each expert's
        # already-computed EMAs - no recompute), publish for the live dashboards,
        # and log scalars so they flow to the metrics DB -> /api/metrics -> the
        # Research-tab cards (swarm_loss / _std / _acc / _experts).
        if batch_idx % self.metrics_every == 0:
            m = self.pool.sample_metrics(k=16)
            if m:
                pool_status.publish_metrics(m)
                scalars = {"swarm_experts": float(len(self.pool.alive()))}
                if m.get("loss_mean") is not None:
                    scalars["swarm_loss"] = float(m["loss_mean"])
                if m.get("loss_std") is not None:
                    scalars["swarm_loss_std"] = float(m["loss_std"])
                if m.get("acc_mean") is not None:
                    scalars["swarm_acc"] = float(m["acc_mean"])
                try:
                    pl_module.log_dict(
                        scalars, on_step=True, on_epoch=False, logger=True
                    )
                except Exception:
                    pass
        self.pool.capacity()

    def on_fit_end(self, trainer, pl_module) -> None:
        self._teardown()

    def teardown(self, trainer, pl_module, stage) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self.manager is not None:
            self.manager.stop()
            self.manager = None
        self.pool.shutdown()
        pool_status.clear()
