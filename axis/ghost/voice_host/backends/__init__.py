"""Backend registry for the ghost voice host.

Adding a model means adding a module here and one REGISTRY entry. Nothing in
Godot changes - that is the whole point of the subprocess design (VOICE_PLAN.md
section 2), and it is the test of whether the interface is drawn correctly.

A backend must not import its heavy dependencies at module import time. The host
enumerates every registered backend to answer `voices`, and one missing wheel
must degrade to "that backend is unavailable" rather than taking the process
down. Import inside __init__ or later, and raise BackendError.
"""

from __future__ import annotations

from typing import Any


class BackendError(RuntimeError):
    """Anything a user could plausibly be shown: a missing model, an absent
    dependency, a voice that will not load. Distinct from a genuine bug, which
    is allowed to raise normally and get a traceback on stderr."""


class Backend:
    """The contract. Deliberately narrow, and deliberately the same shape the
    procedural engine already satisfies, so both paths can sit behind one
    VoiceSession on the Godot side."""

    name = "base"

    # -- identity ----------------------------------------------------------

    @classmethod
    def describe(cls) -> dict:
        """Static capability report, answerable without loading anything.

        The UI is built from this rather than from assumptions: a backend that
        cannot control duration simply does not show that slider. Backends
        differ enormously in what they expose and pretending otherwise is how
        you end up with dead controls.
        """
        return {
            "name": cls.name,
            "streaming": False,        # incremental synthesis, or utterance-at-a-time
            "phoneme_input": False,    # accepts pre-computed phonemes (ghost's own G2P)
            "duration_control": False,  # global rate, at minimum
            "pitch_control": False,
            "reference_audio": False,   # voice cloning / style transfer
            "timings": False,           # returns word or phone boundaries
            "singing": False,
        }

    @classmethod
    def owns(cls, voice: str) -> bool:
        """Whether this backend recognizes a voice id, so callers need not
        track which backend a voice came from."""
        return False

    # -- work --------------------------------------------------------------

    def voices(self) -> list[dict]:
        """Installed and installable voices. Each entry MUST carry a `license`
        field: for most of these projects the code license and the checkpoint
        license differ, and the checkpoint is where the commercial risk sits
        (VOICE_PLAN.md section 8)."""
        raise NotImplementedError

    def ensure(self, voice: str) -> dict:
        """Download or prepare a voice. Separate from synthesize so a UI can
        show progress before the user is sitting waiting on audio."""
        raise NotImplementedError

    def synthesize(self, text: str, voice: str, out_path: str,
                   params: dict[str, Any], phonemes: Any = None) -> dict:
        """Write a WAV to out_path and return metadata.

        Must return: {"wav", "sample_rate", "duration"} and, where the model can
        supply them, {"words": [{text, t0, t1}], "phones": [{p, t0, t1}]}.
        ghost's karaoke subtitles and its audio-reactive scene pipeline both key
        off those timings, so a backend that can produce them should.
        """
        raise NotImplementedError


REGISTRY: dict[str, type[Backend]] = {}


def register(cls: type[Backend]) -> type[Backend]:
    REGISTRY[cls.name] = cls
    return cls


# Import for side effects. Each module registers itself and must not import its
# heavy dependencies at module scope.
from . import piper  # noqa: E402,F401
