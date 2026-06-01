"""Node.js sidecar: backend-hosted remote experts.

So the pool has peers the moment the app comes online - before any browser
connects - we spawn a single Node process running the *same* transformer math as
the browser agents (praxis/web/src/js/sidecar.js -> nanoformer.js) and proxy its
experts into the pool over localhost HTTP. The frontend joins the very same pool
by POSTing more experts to the sidecar; the pool re-discovers them and the
``remote_layers`` count grows on both dashboards.

Each sidecar expert is a full tiny transformer mapping token ids -> next-token
logits, so it *votes* over the distribution (the CALM expert-vote model the
``vote`` mixer expects). Training is local, layer-wise Mono-Forward in JS.

* :class:`SidecarExpert` - a :class:`RemoteExpert` that forwards calls over HTTP.
* :class:`SidecarManager` - launches the Node process, discovers experts, and
  syncs the pool's membership to the sidecar's.

Graceful by design: if Node is unavailable the manager logs and yields no
experts (the pool just runs whatever local/browser experts it has).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from praxis.orchestration.base import RemoteExpert

_SIDECAR_JS = (
    Path(__file__).resolve().parents[1] / "web" / "src" / "js" / "sidecar.js"
)


def _http(url: str, payload: Optional[dict] = None, method: str = "GET", timeout: float = 10.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class SidecarExpert(RemoteExpert):
    """A remote expert living in the Node sidecar, addressed over HTTP.

    Votes over next-token logits: :meth:`forward` returns a 1-D logit vector for
    the batch's *first* row (the sidecar scores one id-sequence per call). The
    pool's ``vote`` mixer averages these across experts.
    """

    kind = "sidecar"

    def __init__(self, uid: str, base_url: str, rank: int) -> None:
        super().__init__(uid)
        self.base_url = base_url
        self._rank = rank

    def _ids_from(self, activations: Tensor) -> List[int]:
        # The sidecar speaks token ids; for the wire we pass the first row's ids.
        # (activations here are expected to be a [.., T] int tensor of ids.)
        flat = activations.reshape(activations.shape[0], -1)[0]
        return [int(x) for x in flat.tolist()]

    def forward(self, activations: Tensor) -> Tensor:
        self.passes += 1
        self._stamp()
        try:
            out = _http(
                f"{self.base_url}/expert/{self.uid}/forward",
                {"ids": self._ids_from(activations)}, method="POST",
            )
            return torch.tensor(out["logits"], dtype=torch.float32)
        except Exception:
            self._alive = False
            raise

    def train_step(self, activations: Tensor, labels: Tensor) -> float:
        ids = self._ids_from(activations)
        tgt = [int(x) for x in labels.reshape(labels.shape[0], -1)[0].tolist()]
        try:
            out = _http(
                f"{self.base_url}/expert/{self.uid}/train",
                {"ids": ids, "targets": tgt}, method="POST",
            )
        except Exception:
            self._alive = False
            raise
        self.steps += 1
        self.passes += 1
        self.last_loss = float(out["loss"])
        self._stamp()
        return self.last_loss

    def rank(self) -> int:
        return self._rank


class SidecarManager:
    """Owns the Node sidecar process and proxies its experts into a pool."""

    def __init__(self, pool, port: int = 7777, init_experts: int = 4, dim: int = 14, vocab: int = 16):
        self.pool = pool
        self.port = port
        self.init_experts = init_experts
        self.dim = dim
        self.vocab = vocab
        self.base_url = f"http://127.0.0.1:{port}"
        self.proc: Optional[subprocess.Popen] = None

    def available(self) -> bool:
        return shutil.which("node") is not None and _SIDECAR_JS.exists()

    def start(self, wait: float = 8.0) -> bool:
        """Launch the sidecar and wait until it answers. Returns success."""
        if not self.available():
            print("[orchestration] node sidecar unavailable; pool runs without backend experts")
            return False
        self.proc = subprocess.Popen(
            ["node", str(_SIDECAR_JS),
             "--port", str(self.port), "--experts", str(self.init_experts),
             "--dim", str(self.dim), "--vocab", str(self.vocab)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                _http(f"{self.base_url}/capacity", timeout=1.0)
                self.sync()
                print(f"[orchestration] sidecar up on {self.base_url}; synced {len(self.pool.experts)} experts")
                return True
            except Exception:
                time.sleep(0.2)
        print("[orchestration] sidecar did not come up in time")
        return False

    def sync(self) -> None:
        """Reconcile the pool's sidecar experts with the sidecar's actual list -
        new browser-spawned experts get added, removed ones dropped. Cheap; call
        periodically (or each train step) to watch the pool grow."""
        try:
            infos = _http(f"{self.base_url}/experts", timeout=2.0)
        except Exception:
            return
        live = {i["uid"]: i for i in infos}
        have = {e.uid for e in self.pool.experts if getattr(e, "kind", "") == "sidecar"}
        for uid, i in live.items():
            if uid not in have:
                self.pool.add(SidecarExpert(uid, self.base_url, int(i["rank"])))
        for e in list(self.pool.experts):
            if getattr(e, "kind", "") == "sidecar" and e.uid not in live:
                self.pool.remove(e.uid)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            self.proc = None
