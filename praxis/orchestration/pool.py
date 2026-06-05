"""ExpertPool: the remote-expert pooling layer.

Holds many tiny experts (local or remote) and exposes them as one layer with two
distinct execution modes, matching the swarm's needs:

* **train (non-blocking)** - dispatch a batch to every live expert without
  waiting on stragglers. Each expert does a *detached, local* Mono-Forward
  update on its own copy, so there is no gradient to synchronize and a slow peer
  never blocks the others. Returns whatever finished; the rest catch up next
  step. With thousands of tiny peers and network latency, this is the only mode
  that scales.
* **infer (stochastic)** - sample a subset of experts, run their forwards, and
  combine with a mixing strategy (mean / vote / sample / standing-wave). The
  sampling is the exploration; the mixer is how peers compose.

The pool also reports **live capacity** (alive experts, total rank, served
passes) for the CLI and web dashboards.

Transport-agnostic: it only calls the :class:`RemoteExpert` interface, so the
same pool drives in-process experts, Ray actors, Hivemind backends, or browser
peers over GUN.
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from praxis.orchestration import status
from praxis.orchestration.base import RemoteExpert
from praxis.orchestration.mixing import build_mixer


class ExpertPool:
    """A pool of remote experts behind one mixing layer.

    Args:
        experts: initial experts (may be empty; peers can join later).
        mixing: a ``MIXING_REGISTRY`` key for the inference combiner.
        sample_size: experts to sample per inference step (None = all alive).
        max_workers: thread pool size for non-blocking dispatch.
    """

    def __init__(
        self,
        experts: Optional[List[RemoteExpert]] = None,
        mixing: str = "mean",
        sample_size: Optional[int] = None,
        max_workers: int = 16,
    ) -> None:
        self.experts: List[RemoteExpert] = list(experts or [])
        self.mixing_name = mixing
        self.mixer = build_mixer(mixing)
        self.sample_size = sample_size
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers)
        # At most one in-flight task per expert (uid -> Future). A straggler is
        # skipped on later steps until it lands, so the executor queue - and the
        # tensors each queued task pins - stay bounded no matter how slow a peer is.
        self._inflight: Dict[str, cf.Future] = {}
        # Cumulative routing telemetry (inference): how heavily the pool is
        # actually leaning on its experts, and how many forwards land.
        self.infer_rounds = 0  # inference forwards driven through the pool
        self.routed = 0  # expert-forwards dispatched (sum over rounds)
        self.routed_ok = 0  # expert-forwards that returned successfully
        self.last_routed = 0  # experts routed to in the most recent round

    # -- membership ----------------------------------------------------------

    def add(self, expert: RemoteExpert) -> None:
        self.experts.append(expert)

    def remove(self, uid: str) -> bool:
        for i, e in enumerate(self.experts):
            if e.uid == uid:
                del self.experts[i]
                self._inflight.pop(uid, None)
                return True
        return False

    def _submit(self, fn, expert, *args) -> Optional[cf.Future]:
        """Submit work for an expert unless it still has a task in flight."""
        prev = self._inflight.get(expert.uid)
        if prev is not None and not prev.done():
            return None
        fut = self._pool.submit(fn, *args)
        self._inflight[expert.uid] = fut
        return fut

    def alive(self) -> List[RemoteExpert]:
        return [e for e in self.experts if e.alive]

    # -- capacity (for the dashboards) --------------------------------------

    def capacity(self) -> Dict[str, Any]:
        """Live snapshot of pool capacity. Cheap; safe to call every step. Also
        publishes to the shared status so the dashboards can read it."""
        alive = self.alive()
        total_rank = sum(e.rank() for e in alive)
        passes = sum(e.passes for e in self.experts)
        steps = sum(e.steps for e in self.experts)
        cap = {
            "experts_total": len(self.experts),
            "experts_alive": len(alive),
            "total_rank": total_rank,
            "passes": passes,
            "steps": steps,
            "mixing": self.mixing_name,
            # Inference routing telemetry.
            "infer_rounds": self.infer_rounds,
            "routed": self.routed,  # cumulative expert-forwards dispatched
            "routed_ok": self.routed_ok,  # cumulative successful forwards
            "last_routed": self.last_routed,  # experts in the most recent round
        }
        status.publish(cap, experts=self.expert_infos())
        return cap

    def expert_infos(self) -> List[Dict[str, Any]]:
        return [e.info() for e in self.experts]

    def sample_metrics(self, k: Optional[int] = 16) -> Dict[str, Any]:
        """Cheaply estimate pool-wide training metrics by sampling ``k`` experts
        and reading their *already-computed* EMAs (no recompute, no forward). Use
        at logging intervals - polling thousands of tiny experts every step would
        cost more than the experts themselves. Returns mean loss/accuracy and
        their spread across the sample, or empty when nothing has trained yet.
        """
        alive = [e for e in self.alive() if e.ema_loss is not None]
        if not alive:
            return {}
        if k and k < len(alive):
            idx = torch.randperm(len(alive))[:k].tolist()
            alive = [alive[i] for i in idx]
        losses = [e.ema_loss for e in alive]
        accs = [e.ema_acc for e in alive if e.ema_acc is not None]
        n = len(losses)
        mean_loss = sum(losses) / n
        var_loss = sum((x - mean_loss) ** 2 for x in losses) / n
        out = {
            "sampled": n,
            "loss_mean": mean_loss,
            "loss_std": var_loss**0.5,
        }
        if accs:
            out["acc_mean"] = sum(accs) / len(accs)
        return out

    # -- train: non-blocking detached updates -------------------------------

    def train_step(
        self,
        activations: Tensor,
        labels: Tensor,
        timeout: Optional[float] = None,
        ids: Optional[Tensor] = None,
        id_targets: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Dispatch a local update to every live expert without blocking on
        stragglers. Returns the losses that completed within ``timeout`` (None =
        wait for all). Each expert updates only itself - nothing to synchronize.

        Experts declaring ``consumes = "ids"`` (e.g. sidecar transformers) get the
        raw token-id payload instead of embedded activations. An expert still
        busy with an earlier dispatch is skipped this round (no backlog).
        """
        alive = self.alive()
        futs = {}
        for e in alive:
            if getattr(e, "consumes", "activations") == "ids":
                if ids is None or id_targets is None:
                    continue
                fut = self._submit(e.train_step, e, ids, id_targets)
            else:
                fut = self._submit(e.train_step, e, activations, labels)
            if fut is not None:
                futs[fut] = e
        losses: Dict[str, float] = {}
        try:
            for fut in cf.as_completed(futs, timeout=timeout):
                e = futs[fut]
                try:
                    losses[e.uid] = fut.result()
                except Exception:
                    e._alive = False  # a real error drops a peer from the pool
        except cf.TimeoutError:
            # Stragglers: take whatever finished in time. They keep running in the
            # background and land on a later step - NOT marked dead, so a large
            # pool doesn't silently stop training its tail. (This is the whole
            # point of non-blocking dispatch.)
            pass
        done = list(losses.values())
        self.capacity()  # refresh the published pool status after the step
        return {
            "losses": losses,
            "completed": len(done),
            "dispatched": len(futs),
            "mean_loss": (sum(done) / len(done)) if done else None,
        }

    # -- infer: stochastic sample + mix -------------------------------------

    def infer(
        self,
        activations: Tensor,
        generator: Optional[torch.Generator] = None,
        timeout: Optional[float] = None,
        ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Sample a subset of experts, run forwards in parallel, and combine via
        the mixing strategy. Returns None if no expert is alive. Mixes whatever
        lands within ``timeout`` (None = wait for all); busy experts are skipped,
        so this can never block behind a backlog.
        """
        chosen = self._sample_experts(generator)
        self.infer_rounds += 1
        futs = {}
        for e in chosen:
            if getattr(e, "consumes", "activations") == "ids":
                if ids is None:
                    continue
                fut = self._submit(e.forward, e, ids)
            else:
                fut = self._submit(e.forward, e, activations)
            if fut is not None:
                futs[fut] = e
        self.last_routed = len(futs)
        self.routed += len(futs)
        if not futs:
            self.capacity()
            return None
        outputs: List[Tensor] = []
        try:
            for fut in cf.as_completed(futs, timeout=timeout):
                e = futs[fut]
                try:
                    outputs.append(fut.result())
                    self.routed_ok += 1
                except Exception:
                    e._alive = False
        except cf.TimeoutError:
            pass  # mix what finished; stragglers stay in flight and get skipped
        if not outputs:
            self.capacity()
            return None
        # Heterogeneous pools (id-voters vs activation experts) return different
        # shapes; mix only the majority shape rather than crash the stack.
        shapes: Dict[Any, int] = {}
        for o in outputs:
            shapes[o.shape] = shapes.get(o.shape, 0) + 1
        majority = max(shapes, key=shapes.get)
        outputs = [o for o in outputs if o.shape == majority]
        mixed = self.mixer(torch.stack(outputs, dim=0))
        self.capacity()  # refresh published status after a routed round
        return mixed

    def _sample_experts(
        self, generator: Optional[torch.Generator]
    ) -> List[RemoteExpert]:
        alive = self.alive()
        k = self.sample_size
        if not k or k >= len(alive):
            return alive
        idx = torch.randperm(len(alive), generator=generator)[:k].tolist()
        return [alive[i] for i in idx]

    def heartbeat(self) -> None:
        """Ping every expert; transports override RemoteExpert.heartbeat to mark
        unreachable peers as not-alive."""
        for e in self.experts:
            try:
                e.heartbeat()
            except Exception:
                e._alive = False

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
