"""Per-module compute-time attribution: where does a training step actually go?

Answers "why is this experiment slow" by attributing measured GPU time to the
``nn.Module`` that caused it, forward *and* backward, then rolling the result up
by module class. The dashboard renders it as a qdirstat-style treemap (area =
share of the step) over a ranked table.

Three facts about this repo and this hardware shaped the design; each was
measured, not assumed.

1. **Wall-clock in a forward hook is meaningless.** CUDA is asynchronous, so
   ``perf_counter`` around a submodule times the kernel *launch*. Measured: hooks
   claimed 10.3 ms of a 106.7 ms step.

2. **``torch.profiler`` reports zero device time on this box.** Kineto's CUPTI
   backend fails to initialise on the RTX 5060 Ti (sm_120) with
   ``CUPTI_ERROR_INVALID_DEVICE``; every ``self_device_time_total`` comes back
   0.000 ms. The *legacy* ``torch.autograd.profiler`` works, because it times
   with CUDA events rather than CUPTI - measured 40.36 ms against a 40.1 ms wall
   step. That is why this module uses the legacy profiler.

3. **Backward cannot be timed with module backward hooks.**
   ``register_full_backward_hook`` spans go *negative* for residual blocks (the
   parent's span comes back shorter than its children's). Instead each backward
   autograd node is linked to the forward op that created it via ``sequence_nr``
   - measured 1894/1910 linked on a real model - and the node's whole subtree is
   credited to that forward op's module, because the kernels are children of the
   node, not the node itself.

Attribution uses ``self_device_time_total``, which already excludes child
events, so per-module numbers are *exclusive* and sum to the total with no
inclusive-minus-children arithmetic (and therefore no negative entries).

Cost is paid on sampled steps only, and the sampled window is ONE microbatch,
not a whole accumulation cycle - every microbatch runs the same graph, so the
extra ones cost the accumulation factor and tell you nothing. Getting that wrong
turned a ~4x profiled step into a ~35x one and burned 9% of a real 123-minute
run in 16-55s stalls. With the window right, a profiled step is ~4x normal
(recording ~2.3x, plus profiler teardown and the attribution walk over ~18k
events), so at one step in 100 the run pays around 3%.

libkineto also writes ``profiler_start``/``profiler_stop`` and its CUPTI
warnings directly to file descriptors 1 and 2, below anything Python can
capture; :func:`_quiet_native_logs` silences that around the enter/exit calls so
it cannot land on top of the terminal dashboard.

Sampling makes the raw numbers jumpy - halting samples a different depth each
pass, the sequence-length curriculum changes shape, and the profiler's own
overhead is not uniform across ops. Every reported value is therefore an EMA
across samples (:data:`COMPUTE_DEFAULTS`), so the card settles instead of
flickering. A component absent from a sample is observed as zero and decays out
rather than vanishing.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from praxis.metrics.ema import compute_ema

# Model-agnostic constants. Not CLI flags, not per-experiment knobs: a registry
# of fixed defaults baked into the profiler (see feedback_registry_over_cli_args).
COMPUTE_DEFAULTS: Dict[str, Any] = {
    # Profile one optimizer step every N steps. A profiled step runs ~2.7x
    # normal, so 100 keeps the amortised cost under 2%.
    "interval": 100,
    # Skip early steps: lazy-module init, autotuning and cache warmup are not
    # representative of steady-state cost.
    "warmup_steps": 50,
    # Smoothing across samples. ~3-sample half-life; with interval=100 that is
    # ~300 steps to half-adapt to a genuine shift.
    "ema_alpha": 0.2,
    # Module paths retained per class in the snapshot; the tail folds into one
    # residual tile so the treemap stays legible.
    "top_k": 18,
    # Classes retained as top-level treemap tiles; the tail folds into "other".
    "max_classes": 14,
}

# Bucket names that are not module classes. Kept distinct so the renderer can
# grey them out and nobody reads them as a slow layer.
OUTSIDE_MODEL = "(outside model)"
"""Device time measured under no module scope: the loss function, optimizer
math, and host-device syncs. Real cost, but not attributable to a layer."""


@contextlib.contextmanager
def _quiet_native_logs():
    """Silence libkineto's C++ writes for the duration of the block.

    Starting and stopping the profiler makes libkineto write lines like
    ``ActivityProfilerController.cpp:415] profiler_start`` and its CUPTI-init
    warnings straight to file descriptors 1 and 2. That is below Python's
    logging and below the terminal dashboard's capture, so the text lands on
    top of the rendered TUI and corrupts the frame - once per profiled step,
    which is exactly the cadence this profiler runs at.

    The redirect is at the fd level because nothing above it can intercept a
    C++ write, and it wraps only the enter/exit calls: the profiled step itself
    runs with normal descriptors, so real output is never swallowed. Restores
    on every path, and no-ops if the descriptors cannot be duplicated.
    """
    try:
        saved_out, saved_err = os.dup(1), os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        # Best-effort: another thread can have swapped sys.stdout for a buffer
        # of its own and closed it (contextlib.redirect_stdout mutates a
        # process-global), and optional telemetry must not be the thing that
        # notices. The fd-level redirect below is what actually matters here;
        # flushing is only so buffered Python output does not land in devnull.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
        finally:
            for fd in (devnull, saved_out, saved_err):
                try:
                    os.close(fd)
                except OSError:
                    pass


class ComputeProfiler:
    """Attribute measured device time to modules, smoothed across samples.

    Install once against an *uncompiled* model (see :meth:`install`), arm it
    around a training step, then read :meth:`metrics` and :meth:`snapshot`.

    Args:
        cfg: Overrides merged over :data:`COMPUTE_DEFAULTS`. Model-agnostic
            constants live in the defaults; this exists for tests, not per-run
            tuning.
    """

    def __init__(self, cfg: Optional[dict] = None) -> None:
        self.cfg = {**COMPUTE_DEFAULTS, **(cfg or {})}
        self._handles: List[Any] = []
        self._scopes: Dict[int, List[Any]] = {}
        self._calls: Dict[str, int] = {}
        self._arming = False
        self._fired = 0
        self.samples = 0
        # Thread that owns the armed window; hooks from any other thread are
        # ignored (see pre_hook). Set in start(), which runs on the trainer.
        self._owner: Optional[int] = None
        # EMA state, keyed by "path|ClassName".
        self._ema_fwd: Dict[str, float] = {}
        self._ema_bwd: Dict[str, float] = {}
        self._ema_calls: Dict[str, float] = {}
        self._ema_total: Optional[float] = None
        self._ema_coverage: Optional[float] = None
        self._device = "cuda"
        # Forward-only mode, set for compiled models. aot_autograd collapses the
        # whole backward into one CompiledFunctionBackward node, so sequence_nr
        # carries no per-module structure and every backward kernel would land
        # in the unattributed bucket. The forward split alone is the honest
        # subset - measured to track eager within 2.5% at top-level scope.
        self.forward_only = False

    # ------------------------------------------------------------------ setup
    def install(
        self, model: Any, device: str = "cuda", max_depth: Optional[int] = None
    ) -> bool:
        """Attach ``record_function`` scopes to named submodules.

        Must run *before the first forward* - under ``torch.compile`` Dynamo
        installs no guard on ``_forward_hooks``, so hooks attached after the
        first traced forward are silently never called.

        Args:
            model: The model to instrument. A ``torch.compile`` wrapper is
                unwrapped to ``_orig_mod`` so scope names stay clean.
            device: ``"cuda"`` or ``"cpu"``; selects which timing column is read.
            max_depth: Hook only modules at most this many dots deep in the
                dotted name (0 = the model's direct children). ``None`` hooks
                everything. Coarse depths exist for compiled models: each hook
                is a graph break, so a handful of them at component boundaries
                costs little, while hooking every leaf both taxes the step and
                misattributes the kernels Inductor fuses across the boundaries.

        Returns:
            True if hooks were installed.
        """
        import torch

        model = getattr(model, "_orig_mod", model)
        self._device = "cuda" if str(device).startswith("cuda") else "cpu"

        def pre_hook(mod, *_args):
            # Only the thread that armed the window may push. The dashboard
            # generates from this same model on a Flask request thread, and a
            # foreign forward would interleave onto the shared scope stack -
            # corrupting the sample and leaking record_function contexts.
            if not self._arming or threading.get_ident() != self._owner:
                return
            try:
                ctx = torch.profiler.record_function(f"@{mod._praxis_scope}")
                ctx.__enter__()
                self._scopes.setdefault(id(mod), []).append(ctx)
                self._calls[mod._praxis_scope] = (
                    self._calls.get(mod._praxis_scope, 0) + 1
                )
                self._fired += 1
            except Exception:
                pass

        def post_hook(mod, *_args):
            # No _arming check: a scope opened while armed must still close if
            # the window ended mid-forward. The thread check still applies, so
            # a foreign thread can never pop what the owner pushed.
            if threading.get_ident() != self._owner:
                return
            stack = self._scopes.get(id(mod))
            if not stack:
                return
            try:
                stack.pop().__exit__(None, None, None)
            except Exception:
                pass

        for name, mod in model.named_modules():
            if not name:
                continue
            if max_depth is not None and name.count(".") > max_depth:
                continue
            # The same object can sit at several ModuleList indices (the decoder
            # reuses one LocalLayer across layer positions), and named_modules
            # dedups by identity - so the FIRST name wins and every invocation
            # accumulates under it. That is intended: one bucket per physical
            # module, with the call count showing how often it runs.
            mod._praxis_scope = f"{name}|{type(mod).__name__}"
            self._handles.append(mod.register_forward_pre_hook(pre_hook))
            self._handles.append(mod.register_forward_hook(post_hook))
        return bool(self._handles)

    def remove(self) -> None:
        """Detach every hook."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

    # -------------------------------------------------------------- profiling
    def start(self) -> Optional[Any]:
        """Open a profiler context and arm the hooks. Returns the context."""
        import torch.autograd.profiler as ap

        self._scopes.clear()
        self._calls.clear()
        self._fired = 0
        self._owner = threading.get_ident()
        prof = ap.profile(
            use_device=self._device if self._device == "cuda" else None,
            record_shapes=False,
        )
        # libkineto announces profiler_start here, straight to fd 1/2.
        with _quiet_native_logs():
            prof.__enter__()
        self._arming = True
        return prof

    def stop(self, prof: Any) -> bool:
        """Close the profiler, attribute its events, and fold into the EMA.

        Returns:
            True if the sample produced attributable time.
        """
        import torch

        self._arming = False
        # Drain any scope left open by an early exit mid-forward, so a torn
        # sample cannot leak record_function contexts into the next one.
        for stack in self._scopes.values():
            while stack:
                try:
                    stack.pop().__exit__(None, None, None)
                except Exception:
                    break
        self._scopes.clear()
        try:
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            # ...and profiler_stop here.
            with _quiet_native_logs():
                prof.__exit__(None, None, None)
        except Exception:
            return False
        if self._fired == 0:
            return False
        try:
            fwd, bwd, total, attributed = self.attribute(prof.function_events)
        except Exception:
            return False
        if total <= 0:
            return False
        self._fold(fwd, bwd, total, attributed)
        return True

    def hooks_fired(self) -> int:
        """Number of scope entries recorded in the last armed window."""
        return self._fired

    # ------------------------------------------------------------ attribution
    def attribute(
        self, events: Any
    ) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
        """Map profiler events onto module scopes.

        Returns:
            ``(forward_ms, backward_ms, total_device_ms, attributed_ms)``, with
            the per-scope dicts keyed by ``"path|ClassName"``.
        """
        events = list(events)
        if not events:
            return {}, {}, 0.0, 0.0

        attr = (
            "self_device_time_total"
            if hasattr(events[0], "self_device_time_total")
            else "self_cuda_time_total"
        )
        if self._device == "cpu":
            attr = "self_cpu_time_total"

        def self_ms(ev: Any) -> float:
            return (getattr(ev, attr, 0) or 0) / 1000.0

        parent_of = {}
        children: Dict[int, List[Any]] = {}
        for ev in events:
            parent = getattr(ev, "cpu_parent", None)
            parent_of[id(ev)] = parent
            if parent is not None:
                children.setdefault(id(parent), []).append(ev)

        # Innermost enclosing "@path|Class" scope, walking up cpu_parent.
        # Iterative with memoisation: the event tree runs to ~17k nodes on a
        # real model and recursion would risk the interpreter's limit.
        scope_of: Dict[int, Optional[str]] = {}

        def innermost(ev: Any) -> Optional[str]:
            chain = []
            cur = ev
            found: Optional[str] = None
            while cur is not None:
                if id(cur) in scope_of:
                    found = scope_of[id(cur)]
                    break
                chain.append(cur)
                if cur.name.startswith("@"):
                    found = cur.name[1:]
                    break
                cur = parent_of.get(id(cur))
            for node in chain:
                scope_of[id(node)] = found
            return found

        def is_backward(ev: Any) -> bool:
            return getattr(ev, "fwd_thread", 0) not in (0, None)

        # sequence_nr links a backward autograd node to the forward op that
        # created it; that forward op sits inside a module scope, which is how
        # backward time reaches a module at all.
        fwd_by_seq: Dict[int, str] = {}
        for ev in events:
            if is_backward(ev):
                continue
            seq = getattr(ev, "sequence_nr", -1)
            if seq is None or seq < 0 or seq in fwd_by_seq:
                continue
            scope = innermost(ev)
            if scope:
                fwd_by_seq[seq] = scope

        # The kernels are children of the autograd node, so credit the whole
        # subtree. Deepest-first, so an inner linked node keeps its own scope
        # rather than inheriting its parent's.
        def depth(ev: Any) -> int:
            d, cur = 0, parent_of.get(id(ev))
            while cur is not None and d < 256:
                d += 1
                cur = parent_of.get(id(cur))
            return d

        painted: Dict[int, str] = {}
        linked = [
            ev
            for ev in events
            if is_backward(ev) and fwd_by_seq.get(getattr(ev, "sequence_nr", -1))
        ]
        for root in sorted(linked, key=depth, reverse=True):
            scope = fwd_by_seq[getattr(root, "sequence_nr", -1)]
            stack = [root]
            while stack:
                node = stack.pop()
                if id(node) in painted:
                    continue
                painted[id(node)] = scope
                stack.extend(children.get(id(node), ()))

        fwd: Dict[str, float] = {}
        bwd: Dict[str, float] = {}
        total = 0.0
        attributed = 0.0
        for ev in events:
            ms = self_ms(ev)
            if ms <= 0:
                continue
            total += ms
            if is_backward(ev) or id(ev) in painted:
                scope = painted.get(id(ev))
                if scope:
                    bwd[scope] = bwd.get(scope, 0.0) + ms
                    attributed += ms
            else:
                scope = innermost(ev)
                if scope:
                    fwd[scope] = fwd.get(scope, 0.0) + ms
                    attributed += ms
        return fwd, bwd, total, attributed

    # -------------------------------------------------------------------- EMA
    def _fold(
        self,
        fwd: Dict[str, float],
        bwd: Dict[str, float],
        total: float,
        attributed: float,
    ) -> None:
        """Fold one sample into the running EMA.

        Scopes missing from this sample are observed as zero so they decay out
        instead of freezing at a stale value - a block skipped by an early halt
        genuinely cost nothing on this pass.
        """
        alpha = float(self.cfg["ema_alpha"])
        keys = set(self._ema_fwd) | set(self._ema_bwd) | set(fwd) | set(bwd)
        for key in keys:
            self._ema_fwd[key] = compute_ema(
                fwd.get(key, 0.0), self._ema_fwd.get(key), alpha
            )
            self._ema_bwd[key] = compute_ema(
                bwd.get(key, 0.0), self._ema_bwd.get(key), alpha
            )
            self._ema_calls[key] = compute_ema(
                float(self._calls.get(key, 0)), self._ema_calls.get(key), alpha
            )
        self._ema_total = compute_ema(total, self._ema_total, alpha)
        self._ema_coverage = compute_ema(
            attributed / total if total > 0 else 0.0, self._ema_coverage, alpha
        )
        self.samples += 1

    # ----------------------------------------------------------------- output
    def _rollup(self) -> Tuple[List[dict], float]:
        """Smoothed per-class groups with their module children, plus the total."""
        rows = []
        for key in set(self._ema_fwd) | set(self._ema_bwd):
            ms = self._ema_fwd.get(key, 0.0)
            if not self.forward_only:
                ms += self._ema_bwd.get(key, 0.0)
            if ms <= 0:
                continue
            path, _, cls = key.partition("|")
            rows.append(
                {
                    "path": path,
                    "cls": cls or "?",
                    "ms": ms,
                    "fwd_ms": self._ema_fwd.get(key, 0.0),
                    "bwd_ms": self._ema_bwd.get(key, 0.0),
                    "calls": self._ema_calls.get(key, 0.0),
                }
            )

        # The unattributed remainder is a real, informative slice when we can
        # see the whole step. In forward-only mode it would be mostly the
        # backward pass we deliberately excluded, so showing it would swamp the
        # treemap with a bucket that means "not measured" rather than "not a
        # layer"; the header carries the coverage number instead.
        outside = 0.0
        if (
            not self.forward_only
            and self._ema_total is not None
            and self._ema_coverage is not None
        ):
            outside = max(0.0, self._ema_total * (1.0 - self._ema_coverage))

        groups: Dict[str, dict] = {}
        for row in rows:
            grp = groups.setdefault(
                row["cls"],
                {"name": row["cls"], "ms": 0.0, "calls": 0.0, "children": []},
            )
            grp["ms"] += row["ms"]
            grp["calls"] += row["calls"]
            grp["children"].append(row)

        out = sorted(groups.values(), key=lambda g: -g["ms"])
        max_classes = int(self.cfg["max_classes"])
        if len(out) > max_classes:
            tail = out[max_classes:]
            out = out[:max_classes]
            out.append(
                {
                    "name": f"other ({len(tail)} classes)",
                    "ms": sum(g["ms"] for g in tail),
                    "calls": sum(g["calls"] for g in tail),
                    "children": [],
                    "residual": True,
                }
            )

        top_k = int(self.cfg["top_k"])
        for grp in out:
            kids = sorted(grp["children"], key=lambda r: -r["ms"])
            if len(kids) > top_k:
                rest = kids[top_k:]
                kids = kids[:top_k]
                kids.append(
                    {
                        "path": f"+{len(rest)} more",
                        "cls": grp["name"],
                        "ms": sum(r["ms"] for r in rest),
                        "fwd_ms": 0.0,
                        "bwd_ms": 0.0,
                        "calls": 0.0,
                        "residual": True,
                    }
                )
            grp["children"] = kids

        if outside > 0:
            out.append(
                {
                    "name": OUTSIDE_MODEL,
                    "ms": outside,
                    "calls": 0.0,
                    "children": [],
                    "outside": True,
                }
            )
            out.sort(key=lambda g: -g["ms"])

        total = sum(g["ms"] for g in out)
        return out, total

    def metrics(self) -> Dict[str, float]:
        """Scalars for dynamics.db. Empty until the first sample lands."""
        if not self.samples:
            return {}
        groups, total = self._rollup()
        if total <= 0:
            return {}
        inside = [g for g in groups if not g.get("outside")]
        out = {
            "compute_coverage": float(self._ema_coverage or 0.0),
            "compute_samples": float(self.samples),
        }
        if inside:
            out["compute_top_share"] = float(inside[0]["ms"] / total)
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Treemap payload for the dashboard. Empty until the first sample."""
        if not self.samples:
            return {}
        groups, total = self._rollup()
        if total <= 0:
            return {}
        payload = {
            "total_ms": total,
            "coverage": float(self._ema_coverage or 0.0),
            "samples": int(self.samples),
            "interval": int(self.cfg["interval"]),
            "ema_alpha": float(self.cfg["ema_alpha"]),
            "mode": "forward" if self.forward_only else "full",
            "groups": [
                {
                    "name": g["name"],
                    "ms": g["ms"],
                    "share": g["ms"] / total,
                    "calls": g["calls"],
                    "outside": bool(g.get("outside")),
                    "residual": bool(g.get("residual")),
                    "children": [
                        {
                            "name": c["path"],
                            "ms": c["ms"],
                            "share": c["ms"] / total,
                            "fwd_ms": c["fwd_ms"],
                            "bwd_ms": c["bwd_ms"],
                            "calls": c["calls"],
                        }
                        for c in g["children"]
                    ],
                }
                for g in groups
            ],
        }
        return {"compute_profile": payload}

    def summary(self, limit: int = 8) -> str:
        """One-line-per-class text table, for the console."""
        if not self.samples:
            return "[ComputeProfiler] no samples yet"
        groups, total = self._rollup()
        if total <= 0:
            return "[ComputeProfiler] no attributable time"
        scope = "forward only, compiled" if self.forward_only else "forward+backward"
        lines = [
            f"[ComputeProfiler] {self.samples} sample(s), {scope}, "
            f"{self._ema_coverage or 0.0:.1%} of device time attributed "
            f"(EMA alpha={self.cfg['ema_alpha']})",
            f"    {'CLASS':<28}{'share':>8}{'ms':>10}{'calls':>8}",
        ]
        for grp in groups[:limit]:
            lines.append(
                f"    {grp['name'][:27]:<28}{grp['ms'] / total:7.1%}"
                f"{grp['ms']:10.2f}{grp['calls']:8.1f}"
            )
        return "\n".join(lines)


COMPUTE_METRIC_DESCRIPTIONS: Dict[str, dict] = {
    "compute_profile": {
        "description": (
            "Where the step's GPU time goes, attributed to the module that caused it. "
            "Tile area is share of the step. Read shares, not milliseconds."
        ),
        "snapshot": {
            "title": "Compute Time by Component",
            "renderer": "compute_treemap",
            "group": "Compute",
            "order": 130,
        },
    },
    "compute_top_share": {
        "description": (
            "Share of attributed step time held by the heaviest module class. Rising = "
            "the run is becoming dominated by one component."
        ),
        "chart": {
            "title": "Dominant Component Share",
            "y_label": "share of step",
            "group": "Compute",
            "order": 131,
        },
    },
    "compute_coverage": {
        "description": (
            "Fraction of measured device time the profiler could attribute to a "
            "module. The rest is loss, optimizer and syncs - work no layer owns."
        ),
        "chart": {
            "title": "Attribution Coverage",
            "y_label": "fraction",
            "group": "Compute",
            "order": 132,
        },
    },
    "compute_samples": {
        "description": (
            "Profiled steps folded into the EMA so far. Early readings move; after a "
            "few samples the card settles."
        ),
    },
}
