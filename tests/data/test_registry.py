"""Tests for data sampling weight modes: static, dynamic, and novelty."""

from unittest.mock import Mock

import pytest

from praxis.data.datasets.novelty import NoveltyTracker

# ---------------------------------------------------------------------------
# Manager integration tests — helpers
# ---------------------------------------------------------------------------


def _make_tokenizer():
    """Create a minimal GPT-2 tokenizer with chat template."""
    from transformers import AutoTokenizer

    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOS]", "[SEP]", "[PAD]"]}
    )
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


def _make_sampler(name, get_document_fn):
    """Create a mock sampler."""
    sampler = Mock()
    sampler.dataset_path = name
    sampler.get_document = get_document_fn
    return sampler


def _simple_doc(content="Hello world"):
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "OK"},
        ],
        "metadata": {"source": "test"},
    }


def _reset_shared_state():
    """Reset class-level shared state between tests."""
    from praxis.data.datasets.manager import InterleaveDataManager

    InterleaveDataManager.shared_weights = None
    InterleaveDataManager.shared_weights_initialized = False


# ---------------------------------------------------------------------------
# Manager integration tests
# ---------------------------------------------------------------------------


class TestStaticMode:
    def setup_method(self):
        _reset_shared_state()

    def test_weights_unchanged(self):
        """In static mode, weights should remain exactly as given."""
        from praxis.data.datasets.manager import InterleaveDataManager

        tokenizer = _make_tokenizer()
        sampler = _make_sampler("ds", lambda: _simple_doc())
        manager = InterleaveDataManager(
            samplers=[sampler, sampler],
            weights=[0.8, 0.2],
            tokenizer=tokenizer,
            block_size=128,
            weighting_mode="static",
        )

        assert not manager._adaptive
        assert not hasattr(manager, "novelty_tracker")
        assert manager.weights == [0.8, 0.2]

        # Fetching a batch should not change weights
        manager.get_batch(batch_size=1)
        assert manager.weights == [0.8, 0.2]


class TestNoveltyMode:
    def setup_method(self):
        _reset_shared_state()

    def test_novelty_tracker_initialized(self):
        """Novelty mode should create a NoveltyTracker."""
        from praxis.data.datasets.manager import InterleaveDataManager

        tokenizer = _make_tokenizer()
        sampler = _make_sampler("ds", lambda: _simple_doc())
        manager = InterleaveDataManager(
            samplers=[sampler],
            weights=[1.0],
            tokenizer=tokenizer,
            block_size=128,
            weighting_mode="novelty",
        )

        assert manager._adaptive
        assert hasattr(manager, "novelty_tracker")
        assert isinstance(manager.novelty_tracker, NoveltyTracker)

        batch = manager.get_batch(batch_size=1)
        assert batch is not None
