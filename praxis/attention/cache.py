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

    def get_state(self, slot: int) -> Optional[Dict[str, Any]]:
        return self.states.get(slot)

    def set_state(self, slot: int, state: Dict[str, Any]) -> None:
        self.states[slot] = state

    def past_length(self) -> int:
        """Tokens already cached, across both K/V slots and recurrent states.

        Zero when no attention layer wrote anything - generation then falls
        back to full-sequence recompute, so cache-less attentions stay correct.
        """
        length = self.get_seq_length()
        for state in self.states.values():
            length = max(length, state.get("pos", 0))
        return int(length)
