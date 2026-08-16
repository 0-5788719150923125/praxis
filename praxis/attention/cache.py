"""Generation-time cache shared by all attention types.

Extends DynamicCache (K/V slots keyed by current_depth) with a per-slot
state dict for recurrent attentions (Infini/Arc), which carry compressive
memory + a partial-segment K/V tail instead of full-sequence K/V.
"""

from typing import Any, Dict, Optional

from transformers import DynamicCache


class PraxisCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        self.states: Dict[int, Dict[str, Any]] = {}
        # Head-side decode state, keyed by the owning module (a harmonic field's
        # running prefix mean and fast-weight bank, for instance). Kept apart
        # from ``states`` on purpose: ``past_length()`` reads the trunk's own
        # slots and must not see the head's bookkeeping, since the head derives
        # its position offset FROM past_length().
        self.head_states: Dict[str, Dict[str, Any]] = {}

    def get_state(self, slot: int) -> Optional[Dict[str, Any]]:
        return self.states.get(slot)

    def set_state(self, slot: int, state: Dict[str, Any]) -> None:
        self.states[slot] = state

    def get_head_state(self, key: str) -> Optional[Dict[str, Any]]:
        return self.head_states.get(key)

    def set_head_state(self, key: str, state: Dict[str, Any]) -> None:
        self.head_states[key] = state

    def past_length(self) -> int:
        """Tokens already cached, across both K/V slots and recurrent states.

        Zero when no attention layer wrote anything - generation then falls
        back to full-sequence recompute, so cache-less attentions stay correct.
        """
        length = self.get_seq_length()
        for state in self.states.values():
            length = max(length, state.get("pos", 0))
        return int(length)
