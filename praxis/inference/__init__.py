"""Text generation modules for Praxis.

The shape here is one loop, several decoding methods. :class:`Generator` owns
the request queue, the prompt, and the tool state machine; :class:`ModelBackend`
wraps ``model.generate``; and everything that is not next-token sampling -
speculative decoding, CALM's patch vote - is registered as a transformers
*decoding method*, so it inherits the prepared logits processors and stopping
criteria rather than re-deriving them. :mod:`praxis.inference.decoding` holds
the few pieces such a method still needs for itself.
"""

from praxis.inference.context_blocks import (
    DEFAULT_CONTEXT_BLOCKS,
    ContextBlock,
    ContextStreams,
)
from praxis.inference.decode_backend import DecodeBackend, ModelBackend
from praxis.inference.decoding import first_halt, is_halted, pick_next, trunk_hooks
from praxis.inference.generator import Generator
from praxis.inference.mono_forward_generator import MonoForwardGenerator
from praxis.inference.reply import EMPTY_REPLY_PLACEHOLDER, extract_assistant_reply
from praxis.inference.request import GenerationRequest, GenerationResult
from praxis.inference.runtime import bos_prompt, swap_inference_generator
from praxis.inference.speculative import speculative_decoding
from praxis.inference.streamers import ReplyStreamer
from praxis.inference.streaming import (
    StreamingContext,
    normalize_display_breaks,
    random_char_seed,
    random_text_seed,
)

__all__ = [
    # the loop
    "Generator",
    "GenerationRequest",
    "GenerationResult",
    "MonoForwardGenerator",
    "DecodeBackend",
    "ModelBackend",
    # decoding methods and the pieces they share
    "speculative_decoding",
    "pick_next",
    "first_halt",
    "is_halted",
    "trunk_hooks",
    # reply extraction + incremental publication
    "extract_assistant_reply",
    "EMPTY_REPLY_PLACEHOLDER",
    "ReplyStreamer",
    # rolling contexts (the Terminal's growing passages)
    "StreamingContext",
    "normalize_display_breaks",
    "random_char_seed",
    "random_text_seed",
    "ContextBlock",
    "ContextStreams",
    "DEFAULT_CONTEXT_BLOCKS",
    # wiring
    "bos_prompt",
    "swap_inference_generator",
]
