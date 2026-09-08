"""Text generation modules for Praxis.

The shape here is one loop, several decoding methods. :class:`Generator` owns
the request queue, the prompt, and the tool state machine; :class:`ModelBackend`
wraps ``model.generate``; and everything that is not next-token sampling -
speculative decoding, CALM's patch vote - is registered as a transformers
*decoding method*, so it inherits the prepared logits processors and stopping
criteria rather than re-deriving them. :mod:`praxis.generation.decoding` holds
the few pieces such a method still needs for itself.
"""

from praxis.generation.context_blocks import (
    DEFAULT_CONTEXT_BLOCKS,
    ContextBlock,
    ContextStreams,
)
from praxis.generation.decode_backend import DecodeBackend, ModelBackend
from praxis.generation.decoding import first_halt, is_halted, pick_next, trunk_hooks
from praxis.generation.generator import Generator
from praxis.generation.mono_forward_generator import MonoForwardGenerator
from praxis.generation.reply import EMPTY_REPLY_PLACEHOLDER, extract_assistant_reply
from praxis.generation.request import GenerationRequest, GenerationResult
from praxis.generation.runtime import bos_prompt, swap_inference_generator
from praxis.generation.speculative import speculative_decoding
from praxis.generation.streamers import ReplyStreamer
from praxis.generation.streaming import (
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
