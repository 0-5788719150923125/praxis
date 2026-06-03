"""Joke format: quality-filtered imitation of well-rated jokes (the dense signal).

The rated-jokes dataset (SeppeV/rated_jokes_dataset_from_jester) carries each
joke's human `rating`. We keep only above-median jokes so the supervised target is
"reproduce jokes humans liked" - the basic RL signal. The live human-approval
channel (Loop UI) layers the sustaining online signal on top (see JokePolicy).
"""

import random
from typing import Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt

# Jester ratings run roughly -10..10; keep the above-median (liked) jokes so the
# imitation target is quality-weighted. Skipped jokes are retried by the sampler.
JOKE_RATING_THRESHOLD = 0.0

_ASK_TEMPLATES = [
    "Tell me a joke.",
    "Got a joke for me?",
    "Make me laugh.",
    "Tell me something funny.",
]


def format_joke(
    document: Dict, keys: List[str], tokenizer: PreTrainedTokenizer
) -> Dict:
    """Build a joke imitation example, or an empty doc (skipped) for low-rated
    jokes. keys = [jokeText, rating]."""
    joke = (document.get(keys[0]) or "").strip()
    try:
        rating = float(document.get(keys[1], 0.0))
    except (TypeError, ValueError):
        rating = 0.0

    if not joke or rating <= JOKE_RATING_THRESHOLD:
        return {"messages": []}  # skip - sampler draws another

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": sample_developer_prompt("engage_conversation"),
        },
        {"role": "user", "content": random.choice(_ASK_TEMPLATES)},
        {"role": "assistant", "content": joke},
    ]
    return {
        "messages": messages,
        "metadata": {"format": "joke", "rating": rating},
    }
