"""Real-time metrics WebSocket streaming.

Streams structured training metrics to web clients via the /metrics-live
SocketIO namespace, independent of the terminal dashboard.
"""

import threading
import time

from flask_socketio import Namespace, SocketIO, emit


def setup_metrics_live_namespace(socketio: SocketIO) -> None:
    """Set up the /metrics-live WebSocket namespace."""

    class MetricsLiveNamespace(Namespace):
        def on_connect(self):
            """Send immediate snapshot on connect."""
            try:
                from praxis.interface.state.live_metrics import LiveMetrics

                lm = LiveMetrics()
                emit("metrics_snapshot", lm.snapshot())
            except Exception:
                pass

        def on_disconnect(self):
            pass

    socketio.on_namespace(MetricsLiveNamespace("/metrics-live"))

    # Start background emitter
    _start_emitter(socketio)


def _start_emitter(socketio: SocketIO) -> None:
    """Background thread that emits metrics snapshots at 2 Hz."""

    def emitter_loop():
        from praxis.interface.state.live_metrics import LiveMetrics

        lm = LiveMetrics()
        last_update_count = -1

        while True:
            try:
                snapshot = lm.snapshot()
                # Only emit if the update count changed
                if snapshot["update_count"] != last_update_count:
                    socketio.emit(
                        "metrics_snapshot",
                        snapshot,
                        namespace="/metrics-live",
                    )
                    last_update_count = snapshot["update_count"]
                time.sleep(0.5)
            except Exception:
                time.sleep(1)

    t = threading.Thread(target=emitter_loop, daemon=True)
    t.start()
