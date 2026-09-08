"""The server's live push channel.

ONE socket carries everything the server sends a client unprompted while a run
is going: the training metrics snapshot, typed cache invalidations, and the
generation deltas of a reply being written (:mod:`generation_stream`). They
share a connection because opening a second one buys nothing - it is already
connected, already reconnects itself, and already pushes from a background
thread.

The namespace is deliberately NOT named for any one of those. It was
``/metrics-live`` while metrics were all it carried, which stopped being true
the moment inference started riding it; :data:`NAMESPACE` is the single place
the name lives now, so the next thing to share it does not have to agree with a
literal spelled out in three files.
"""

import threading
import time

from flask_socketio import Namespace, SocketIO, emit

# The one place the namespace is spelled. Imported by every publisher.
NAMESPACE = "/realtime"


def setup_realtime_namespace(socketio: SocketIO) -> None:
    """Set up the live push namespace and start the metrics emitter."""

    class RealtimeNamespace(Namespace):
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

    socketio.on_namespace(RealtimeNamespace(NAMESPACE))

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
                        namespace=NAMESPACE,
                    )
                    # Typed invalidation: tells clients chart/history data may
                    # have changed, so they refresh on events, not timers.
                    socketio.emit(
                        "invalidate",
                        {"topic": "metrics", "version": snapshot["update_count"]},
                        namespace=NAMESPACE,
                    )
                    last_update_count = snapshot["update_count"]
                time.sleep(0.5)
            except Exception:
                time.sleep(1)

    t = threading.Thread(target=emitter_loop, daemon=True)
    t.start()
