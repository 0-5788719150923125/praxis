"""Carrying a reply to the browser while it is still being written.

``POST /messages/`` stays exactly what it was - one request, one final JSON
reply - and this is a SIDE CHANNEL beside it. The client mints a stream id,
sends it with the request, and listens on the socket it already has open; the
POST's answer remains authoritative, so a client that ignores the channel (or
whose socket is down) is unaffected.

Sharing the live push channel (:mod:`praxis.web.websocket.realtime`) rather
than adding an SSE route is deliberate: the socket is already connected,
already reconnects itself, and already carries server-pushed frames from a
background thread (``_start_emitter``). One more event type is far less
machinery than a second transport.

WHICH THREAD THIS RUNS ON. The callbacks below fire from wherever the request
is served, which for a queued generation is the TRAINING thread inside
``on_train_batch_end``. ``socketio.emit`` is safe there in ``threading`` async
mode - it is the same call the metrics emitter makes from its own thread - but
everything here must stay cheap and must never raise, because an exception
would surface inside the training step.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from praxis.web.websocket.realtime import NAMESPACE

_log = logging.getLogger("praxis.web")

# Monotonic, process-wide. The client uses it to drop a delta that arrives after
# the POST has already delivered the final reply: the two travel on different
# connections and are produced by different threads, so their arrival order is
# not guaranteed even though their production order is.
_seq = itertools.count(1)


def _emit(event: str, payload: Dict[str, Any]) -> None:
    from praxis.web.app import socketio

    try:
        socketio.emit(event, payload, namespace=NAMESPACE)
    except Exception:
        # A dropped frame costs the client a chunk of preview text and nothing
        # else - the POST still returns the whole reply. Never worth raising
        # into a training step over.
        _log.debug("Failed to emit %s for a generation stream", event, exc_info=True)


def stream_callbacks(
    stream_id: Optional[str],
) -> Tuple[Optional[Callable[[str], None]], Optional[Callable[[], None]]]:
    """``(on_text, on_reset)`` for a request, or ``(None, None)``.

    ``None`` when the client sent no stream id, which is how a caller opts out:
    ``Generator`` then builds no streamer at all and the decode is exactly what
    it was before.

    ``on_reset`` means "drop what you have" and fires once per tool call - the
    runtime's turn anchor moves past each spliced tool result, so the reply is
    only what the model writes after it (see
    :class:`praxis.generation.streamers.ReplyStreamer`).
    """
    if not stream_id:
        return None, None

    def on_text(delta: str) -> None:
        _emit("gen_delta", {"id": stream_id, "seq": next(_seq), "text": delta})

    def on_reset() -> None:
        _emit("gen_reset", {"id": stream_id, "seq": next(_seq)})

    return on_text, on_reset
