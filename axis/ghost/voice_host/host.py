#!/usr/bin/env python3
"""ghost voice host - a neural synthesis subprocess, spoken to over stdio.

WHY A SUBPROCESS AND NOT A GDEXTENSION
--------------------------------------
Godot has no first-class ONNX support, and wrapping ONNX Runtime's C API natively
means per-platform binaries and a build story ghost does not have (there is no
export_presets.cfg for any target yet). It would also mean a native rebuild every
time we want to try a different model, which is exactly the thing the design is
supposed to make cheap.

So: a Python process. ghost already bootstraps a private virtualenv at
user://ytdlp_venv for its YouTube import, complete with upgrade-and-retry, so the
pattern, the failure handling and the precedent all exist. Adding a model becomes
a file in backends/ and an entry in the registry, with no Godot code touched at
all - that is the swappability test in VOICE_PLAN.md section 3.

If inference latency ever justifies it, the transport can be replaced by a
GDExtension without changing the protocol below.

PROTOCOL
--------
Newline-delimited JSON on stdin and stdout. One request per line, one response
per line, `id` echoed back. Audio never travels as JSON - it is written to a file
whose path is returned - because base64 over a pipe is a needless copy of
megabytes and Godot can load a WAV directly.

  -> {"id": 1, "op": "capabilities"}
  <- {"id": 1, "ok": true, "backends": {...}}

  -> {"id": 2, "op": "voices"}
  <- {"id": 2, "ok": true, "voices": [{"id", "name", "backend", "license", ...}]}

  -> {"id": 3, "op": "synthesize", "text": "...", "voice": "en_US-...",
      "out": "/path/to/take.wav", "params": {"length_scale": 1.0}}
  <- {"id": 3, "ok": true, "wav": "...", "sample_rate": 22050, "duration": 4.2,
      "words": [{"text": "the", "t0": 0.0, "t1": 0.12}], "phones": [...]}

  -> {"id": 4, "op": "shutdown"}

Every response carries "ok". On failure it is false and "error" holds a string
that is safe to show a user. The host never exits on a bad request - a crashed
host is a worse failure than a failed request, because the model load is the slow
part and we want it to stay warm.

STDOUT IS THE PROTOCOL. Anything a backend prints would corrupt it, so stdout is
redirected for the whole process and only the protocol writer holds the real one.
Diagnostics go to stderr, which ghost's console surfaces.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backends import REGISTRY, BackendError  # noqa: E402

PROTOCOL_VERSION = 1


class Host:
    def __init__(self) -> None:
        # Claim the real stdout before anything else can write to it, then point
        # the process's stdout at stderr so a chatty backend cannot corrupt the
        # protocol stream. Model loaders are notoriously chatty.
        self._out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        self._backends: dict = {}

    # -- transport ---------------------------------------------------------

    def send(self, payload: dict) -> None:
        self._out.write(json.dumps(payload) + "\n")
        self._out.flush()

    def run(self) -> int:
        self.send(
            {
                "event": "ready",
                "protocol": PROTOCOL_VERSION,
                "backends": sorted(REGISTRY.keys()),
            }
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                self.send({"ok": False, "error": f"malformed request: {exc}"})
                continue
            rid = req.get("id")
            op = req.get("op", "")
            if op == "shutdown":
                self.send({"id": rid, "ok": True})
                return 0
            try:
                self.send({"id": rid, "ok": True, **self.dispatch(op, req)})
            except BackendError as exc:
                self.send({"id": rid, "ok": False, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.send(
                    {"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )
        return 0

    # -- operations --------------------------------------------------------

    def dispatch(self, op: str, req: dict) -> dict:
        if op == "capabilities":
            return {
                "protocol": PROTOCOL_VERSION,
                "backends": {name: cls.describe() for name, cls in REGISTRY.items()},
            }
        if op == "voices":
            voices = []
            for name in REGISTRY:
                try:
                    voices.extend(self.backend(name).voices())
                except BackendError as exc:
                    # one unavailable backend must not hide the others: a missing
                    # dependency is a reportable state, not a fatal one
                    self.send(
                        {
                            "event": "backend_unavailable",
                            "backend": name,
                            "error": str(exc),
                        }
                    )
            return {"voices": voices}
        if op == "synthesize":
            voice = req.get("voice") or ""
            backend = self.backend(req.get("backend") or self.backend_for(voice))
            result = backend.synthesize(
                text=req.get("text", ""),
                voice=voice,
                out_path=req["out"],
                params=req.get("params") or {},
                phonemes=req.get("phonemes"),
            )
            return result
        if op == "ensure":
            # download/prepare a voice without synthesizing, so a UI can show
            # progress before the user is waiting on audio
            return self.backend(
                req.get("backend") or self.backend_for(req.get("voice", ""))
            ).ensure(req.get("voice", ""))
        raise BackendError(f"unknown op '{op}'")

    # -- backend management ------------------------------------------------

    def backend(self, name: str):
        if name not in REGISTRY:
            raise BackendError(f"unknown backend '{name}'")
        if name not in self._backends:
            self._backends[name] = REGISTRY[name]()  # loads lazily; stays warm
        return self._backends[name]

    def backend_for(self, voice: str) -> str:
        for name, cls in REGISTRY.items():
            if cls.owns(voice):
                return name
        raise BackendError(f"no backend owns voice '{voice}'")


if __name__ == "__main__":
    raise SystemExit(Host().run())
