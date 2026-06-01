"""Remote-expert interface for the distributed pooling layer.

A *remote expert* is one tiny model owned by a peer (a browser tab, another
process, a node across the network). The pool talks to all of them through one
transport-agnostic interface so the same pooling/mixing logic runs whether the
expert is in-process, a Ray actor, a Hivemind backend, or a browser peer over
GUN. This mirrors the JS ``SwarmAgent`` (praxis/web/src/js/swarm.js): each expert
owns its weights + a local projection and does detached, layer-wise updates - so
no gradient crosses the network and the forward pass can be non-blocking.

The two ends:

* :class:`RemoteExpert` - the abstract contract every expert type implements.
* :class:`LocalExpert`  - an in-process reference expert (a Mono-Forward block),
  used as the default and for single-host swarm simulation.

Concrete transports (Ray, Hivemind, GUN/browser) subclass :class:`RemoteExpert`
and forward these calls over their wire; the pool never sees the difference.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RemoteExpert(ABC):
    """One peer's tiny model, addressed through a uniform interface.

    Implementations may run the work locally or ship it over a transport. All
    methods take/return CPU tensors so a transport can serialize them; gradients
    never cross the boundary (Mono-Forward: each expert trains itself).
    """

    #: short type tag, set by subclasses (mirrors the JS ``kind`` field).
    kind: str = "remote"

    # EMA halflife for the self-tracked metrics, in local steps. Each expert
    # folds its own loss/accuracy into a running average as it trains, so the
    # pool can be sampled cheaply (read the EMA, never recompute).
    _METRIC_EMA = 0.95

    def __init__(self, uid: str) -> None:
        self.uid = uid
        self.passes = 0  # forward passes served (liveness + load)
        self.steps = 0  # local update steps performed
        self.last_loss: Optional[float] = None
        self.last_beat: Optional[float] = None
        self._alive = True
        # Self-tracked running metrics (folded in during train_step; sampled, not
        # polled). None until the expert has taken a step.
        self.ema_loss: Optional[float] = None
        self.ema_acc: Optional[float] = None

    def _track(self, loss: float, acc: Optional[float] = None) -> None:
        """Fold one step's loss (and optional next-token accuracy) into the EMAs.
        Cheap (two mul-adds); called by subclasses inside their train_step."""
        a = self._METRIC_EMA
        self.ema_loss = (
            loss if self.ema_loss is None else a * self.ema_loss + (1 - a) * loss
        )
        if acc is not None:
            self.ema_acc = (
                acc if self.ema_acc is None else a * self.ema_acc + (1 - a) * acc
            )

    # -- the wire ------------------------------------------------------------

    @abstractmethod
    def forward(self, activations: Tensor) -> Tensor:
        """Run a forward pass, returning this expert's output activations.

        No autograd graph crosses back: the result is detached. Counts as a
        served pass.
        """

    @abstractmethod
    def train_step(self, activations: Tensor, labels: Tensor) -> float:
        """Do one detached, local layer-wise update; return the local loss."""

    # -- liveness / capacity -------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._alive

    @abstractmethod
    def rank(self) -> int:
        """The expert's representational width (its hidden dim) - the unit a
        buyer rents (see the rank-priced connection in next/world_models.md)."""

    def heartbeat(self) -> None:
        """Cheap liveness ping. Default: stamp the clock (override to probe a
        real transport). Pass a monotonic time in via :meth:`_stamp`."""
        self._stamp()

    def _stamp(self, now: Optional[float] = None) -> None:
        self.last_beat = now if now is not None else time.monotonic()

    def info(self) -> Dict[str, Any]:
        """Plain snapshot for the dashboard - shaped like the JS ``toView``."""
        return {
            "uid": self.uid,
            "kind": self.kind,
            "rank": self.rank(),
            "passes": self.passes,
            "steps": self.steps,
            "last_loss": self.last_loss,
            "ema_loss": self.ema_loss,
            "ema_acc": self.ema_acc,
            "alive": self._alive,
        }


class LocalExpert(RemoteExpert):
    """In-process reference expert: a single Mono-Forward block + projection.

    The Python twin of one JS ``SwarmAgent``'s nanoformer block. Owns its layer,
    a projection ``M_i`` to the label space, and a local optimizer; ``train_step``
    runs a detached local CE update (gradients never leave this expert). Used as
    the default expert and to simulate a swarm single-host.
    """

    kind = "local"

    def __init__(
        self,
        uid: str,
        block: nn.Module,
        hidden_size: int,
        vocab_size: int,
        lr: float = 1e-2,
        device: str = "cpu",
    ) -> None:
        super().__init__(uid)
        self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Independent copy so experts never share weights (a diverse basis).
        self.block = copy.deepcopy(block).to(device)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        nn.init.normal_(self.projection.weight, std=(2.0 / hidden_size) ** 0.5)
        params = list(self.block.parameters()) + list(self.projection.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _run_block(self, h: Tensor) -> Tensor:
        # Accept either a plain nn.Module (returns a tensor) or a Praxis
        # LocalLayer (positional attention_mask/past_kv/state/depth, returns a
        # tuple whose first element is the hidden states). Try the simple call,
        # fall back to the full LocalLayer signature.
        try:
            out = self.block(h)
        except TypeError:
            out = self.block(h, None, None, None, 0)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, activations: Tensor) -> Tensor:
        self.passes += 1
        self._stamp()
        with torch.no_grad():
            h = activations.to(self.device)
            return self._run_block(h).detach().cpu()

    def train_step(self, activations: Tensor, labels: Tensor) -> float:
        h = activations.to(self.device).detach().requires_grad_(True)
        out = self._run_block(h)
        logits = self.projection(out)
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_labels = labels.to(self.device).reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        self.passes += 1
        self.last_loss = float(loss.detach().item())
        # Cheap next-token accuracy on this batch (argmax == label), folded into
        # the running EMAs so the pool can be sampled without recomputation.
        with torch.no_grad():
            acc = float(
                (flat_logits.argmax(dim=-1) == flat_labels).float().mean().item()
            )
        self._track(self.last_loss, acc)
        self._stamp()
        return self.last_loss

    def rank(self) -> int:
        return self.hidden_size
