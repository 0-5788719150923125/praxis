"""Generation request data structure."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationRequest:
    """Represents a text generation request."""

    id: str
    prompt: str
    kwargs: Dict[str, Any]
    result: Optional[str] = None
