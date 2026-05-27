"""Summarize a CUDA memory snapshot into a readable, attributed report.

Loads the ``.pickle`` written by :class:`MemoryProfilerCallback`, reconstructs
the allocation timeline from the recorded trace, finds the peak-allocated moment,
and attributes the live bytes at that peak to the praxis call sites that produced
them. Answers "what owns the spike" without the browser visualizer.

    python -m praxis.utils.memory_snapshot build/runs/<id>/memory_snapshot.pickle
"""

import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_GB = 1024**3
_MB = 1024**2


def _salient_frame(frames: Optional[List[dict]]) -> str:
    """Pick the call site to attribute an allocation to: the innermost praxis
    frame, falling back to the innermost frame of any kind."""
    if not frames:
        return "<no stack>"
    for f in frames:
        fn = f.get("filename", "")
        if "praxis" in fn and "site-packages" not in fn:
            return f"{_short(fn)}:{f.get('line','?')} {f.get('name','?')}"
    f = frames[0]
    return f"{_short(f.get('filename',''))}:{f.get('line','?')} {f.get('name','?')}"


def _short(path: str) -> str:
    """Trim a filename to the praxis-relative tail (or the basename)."""
    if "praxis/" in path:
        return "praxis/" + path.split("praxis/", 1)[1]
    return path.rsplit("/", 1)[-1] if path else "?"


def _peak_from_trace(trace: List[dict]) -> Tuple[int, Dict[int, dict]]:
    """Walk alloc/free events to find peak live (allocated) bytes and return
    ``(peak_bytes, {addr: event})`` for the allocations live at that peak."""
    live: Dict[int, dict] = {}
    cur = 0
    peak = 0
    peak_live: Dict[int, dict] = {}
    for e in trace:
        action = e.get("action")
        size = e.get("size", 0)
        addr = e.get("addr")
        if action == "alloc":
            live[addr] = e
            cur += size
            if cur > peak:
                peak = cur
                peak_live = dict(live)
        elif action == "free_completed":
            if addr in live:
                cur -= live.pop(addr).get("size", 0)
    return peak, peak_live


def _attribute(events: Dict[int, dict]) -> List[Tuple[str, int, int]]:
    """Aggregate a set of live allocations by call site -> (site, bytes, count)."""
    by_site_bytes: Dict[str, int] = defaultdict(int)
    by_site_count: Dict[str, int] = defaultdict(int)
    for e in events.values():
        site = _salient_frame(e.get("frames"))
        by_site_bytes[site] += e.get("size", 0)
        by_site_count[site] += 1
    rows = [(s, by_site_bytes[s], by_site_count[s]) for s in by_site_bytes]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _pick_trace(snapshot: Dict[str, Any]) -> List[dict]:
    """Return the device trace with the most events (the active GPU)."""
    traces = snapshot.get("device_traces") or []
    return max(traces, key=len, default=[])


def _reserved_summary(snapshot: Dict[str, Any]) -> Tuple[float, float, int]:
    """Reserved vs live-allocated bytes (and segment count) at dump time. The
    gap is the caching allocator's free-but-held pool - i.e. fragmentation,
    which is what ``nvidia-smi`` / reserved reflect, not live tensors."""
    reserved = allocated = 0
    segs = snapshot.get("segments", [])
    for s in segs:
        reserved += s.get("total_size", 0)
        for b in s.get("blocks", []):
            if b.get("state") == "active_allocated":
                allocated += b.get("size", 0)
    return reserved, allocated, len(segs)


def summarize_snapshot(path: str, top: int = 15) -> str:
    """Build the text report for the snapshot at ``path``."""
    with open(path, "rb") as fh:
        snapshot = pickle.load(fh)

    out: List[str] = [f"Memory snapshot: {path}"]
    trace = _pick_trace(snapshot)

    reserved, alloc_now, n_segs = _reserved_summary(snapshot)
    if n_segs:
        frag = reserved - alloc_now
        out.append(
            f"\nReserved at dump: {reserved/_GB:.2f} GB across {n_segs} segments "
            f"(live {alloc_now/_GB:.2f} GB, free/fragmented {frag/_GB:.2f} GB = "
            f"{frag/reserved*100 if reserved else 0:.0f}%)"
        )
        out.append("  Reserved (not live tensors) is what the device/nvidia-smi shows.")

    if len(trace) >= 100_000:
        out.append(
            f"\n[!] Trace holds {len(trace)} events (at the ring-buffer cap); older "
            "events incl. segment allocs were evicted. Re-run with a larger "
            "--profile-memory window cap for the full reserved timeline."
        )

    if trace:
        peak, peak_live = _peak_from_trace(trace)
        out.append(f"\nPeak allocated (live tensors): {peak/_GB:.2f} GB")
        out.append(f"Live allocations at peak: {len(peak_live)}")
        out.append(f"\nTop {top} call sites at peak (by bytes held):")
        out.append(f"  {'GB':>7}  {'%':>5}  {'allocs':>7}  call site")
        for site, b, n in _attribute(peak_live)[:top]:
            pct = (b / peak * 100) if peak else 0
            out.append(f"  {b/_GB:7.3f}  {pct:5.1f}  {n:7d}  {site}")
    else:
        out.append("\n(no allocation trace; reporting live segments at dump time)")
        live: Dict[int, dict] = {}
        for seg in snapshot.get("segments", []):
            off = seg.get("address", 0)
            for blk in seg.get("blocks", []):
                if blk.get("state") == "active_allocated":
                    live[off] = blk
                off += blk.get("size", 0)
        total = sum(b.get("size", 0) for b in live.values())
        out.append(f"Live allocated: {total/_GB:.2f} GB across {len(live)} blocks")
        out.append(f"\nTop {top} call sites (by bytes held):")
        for site, b, n in _attribute(live)[:top]:
            pct = (b / total * 100) if total else 0
            out.append(f"  {b/_GB:7.3f}  {pct:5.1f}  {n:7d}  {site}")

    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(summarize_snapshot(sys.argv[1]))
