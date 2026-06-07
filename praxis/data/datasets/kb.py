"""KB-backed training dataset.

Samples the knowledge base itself - docs, notes, crawled pages - as raw text.
It iterates KB sources directly (no FTS index needed), reshuffling per epoch
with a seeded RNG; on exhaustion it reloads, so newly crawled pages enter the
mixture as the spider finds them.
"""

import random
from typing import Dict, List

from transformers import PreTrainedTokenizer

from praxis.data.datasets.base import PraxisSampler

# Default sources: written + crawled prose. Configs/cards/agents are skipped
# (metadata, not text).
DEFAULT_SOURCES = ["docs", "notes", "pages"]


class KBDataset(PraxisSampler):
    """Yields one KB item per document: title and origin, then the body."""

    def __init__(self, tokenizer: PreTrainedTokenizer, seed: int, config: Dict):
        super().__init__(tokenizer)
        self.dataset_path = "kb"
        self.sources = config.get("sources", DEFAULT_SOURCES)
        self.rng = random.Random(seed)
        self._epoch: List[str] = []

    def _load_epoch(self) -> None:
        from praxis.kb.sources import KB_SOURCE_REGISTRY

        texts = []
        for name in self.sources:
            source = KB_SOURCE_REGISTRY[name]()
            for item in source.iter_items():
                body = (item.body or "").strip()
                if not body:
                    continue
                header = " · ".join(p for p in (item.title, item.origin) if p)
                texts.append(f"{header}\n\n{body}" if header else body)
        self.rng.shuffle(texts)
        self._epoch = texts

    def fill_sequence_cache(self):
        if not self._epoch:
            self._load_epoch()
        self.sequence_cache.append(self._epoch.pop() if self._epoch else "")
