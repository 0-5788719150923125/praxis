"""Shared live status for the expert pool.

A tiny process-global the active :class:`~praxis.orchestration.pool.ExpertPool`
publishes its capacity to, and the dashboards read from. This decouples the pool
(which may not even exist in a given run) from the terminal callback that paints
the CLI + web surfaces - the same decoupling ``LiveMetrics`` uses for training
metrics. When no pool is active the status is empty and the dashboards simply
omit the capacity line.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_status: Dict[str, Any] = {}
_experts: list = []
_metrics: Dict[str, Any] = {}  # sampled pool-wide training metrics
_pool = None  # the live ExpertPool, registered by ExpertPoolCallback
_batch: Optional[Dict[str, Any]] = None  # latest real batch for browser agents
_batch_seq = 0  # monotonic id so streamers/clients can skip already-seen batches


def publish_batch(ids: list, targets: list) -> None:
    """Publish the latest real training batch (token-id rows, downsampled to the
    swarm's tiny vocab/seq) for browser agents to train on. Bounded depth 1:
    each call overwrites the previous, so a slow consumer simply trains on the
    freshest batch and never builds a backlog - the drop-the-stale policy."""
    global _batch, _batch_seq
    with _lock:
        _batch_seq += 1
        _batch = {"seq": _batch_seq, "ids": ids, "targets": targets}


def latest_batch() -> Optional[Dict[str, Any]]:
    with _lock:
        return None if _batch is None else dict(_batch)


def publish_metrics(metrics: Dict[str, Any]) -> None:
    """Set the latest sampled pool-wide training metrics (loss/acc mean+spread)."""
    global _metrics
    with _lock:
        _metrics = dict(metrics)


def metrics() -> Dict[str, Any]:
    with _lock:
        return dict(_metrics)


def register_pool(pool) -> None:
    """Register the live pool so web routes can grow it (e.g. a browser join)."""
    global _pool
    with _lock:
        _pool = pool


def get_pool():
    """The live ExpertPool, or None when no pool is active."""
    with _lock:
        return _pool


def publish(capacity: Dict[str, Any], experts: Optional[list] = None) -> None:
    """Set the current pool capacity snapshot (called by the pool each step).

    ``experts`` is the optional per-expert info list (so the web Stage tab can
    list backend-hosted experts alongside browser ones).
    """
    global _status, _experts
    with _lock:
        _status = dict(capacity)
        if experts is not None:
            _experts = list(experts)


def clear() -> None:
    global _status, _experts, _metrics, _pool, _batch
    with _lock:
        _status = {}
        _experts = []
        _metrics = {}
        _pool = None
        _batch = None


def snapshot() -> Dict[str, Any]:
    with _lock:
        return dict(_status)


def experts() -> list:
    """Per-expert info list for the most recent publish."""
    with _lock:
        return list(_experts)


def info_line() -> Optional[str]:
    """A terse pool-capacity string for the info panel, or None when no pool is
    active (so the dashboards omit the row). Compact: ``<alive> exp`` plus, once
    inference has routed, the last round's vote width (``/<N>``)."""
    with _lock:
        if not _status:
            return None
        alive = _status.get("experts_alive", 0)
        line = f"{alive} exp"
        loss = _metrics.get("loss_mean")
        if loss is not None:
            line += f", L{loss:.2f}"  # sampled mean expert loss
        return line
