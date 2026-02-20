"""Tests for data sampling weight modes: static, dynamic, and novelty."""

import numpy as np
import pytest
from unittest.mock import Mock

from praxis.data.datasets.novelty import CountMinSketch, NoveltyTracker


# ---------------------------------------------------------------------------
# Count-Min Sketch unit tests
# ---------------------------------------------------------------------------


class TestCountMinSketch:
    def test_add_and_query(self):
        """Basic add/query: inserted keys return correct counts."""
        cms = CountMinSketch(width=1024, depth=4)
        cms.add(42, count=5)
        cms.add(42, count=3)
        assert cms.query(42) == 8

    def test_unseen_key_returns_zero(self):
        """Querying a never-inserted key returns 0."""
        cms = CountMinSketch(width=1024, depth=4)
        assert cms.query(99999) == 0

    def test_decay(self):
        """Decay multiplies all counts down."""
        cms = CountMinSketch(width=1024, depth=4)
        cms.add(10, count=100)
        before = cms.query(10)
        cms.decay(0.5)
        after = cms.query(10)
        assert after == int(before * 0.5)

    def test_batch_operations(self):
        """Batch add/query produces consistent results."""
        cms = CountMinSketch(width=4096, depth=4)
        keys = np.array([1, 2, 3, 1, 2, 1], dtype=np.int64)
        cms.add_batch(keys)
        counts = cms.query_batch(np.array([1, 2, 3, 4], dtype=np.int64))
        assert counts[0] == 3  # key 1 appeared 3 times
        assert counts[1] == 2  # key 2 appeared 2 times
        assert counts[2] == 1  # key 3 appeared 1 time
        assert counts[3] == 0  # key 4 never appeared


# ---------------------------------------------------------------------------
# Novelty tracker unit tests
# ---------------------------------------------------------------------------


class TestNoveltyTracker:
    def test_diverse_vs_repetitive(self):
        """A dataset producing diverse documents should keep higher weight
        than one producing identical documents."""
        tracker = NoveltyTracker(
            num_datasets=2,
            cms_width=4096,
            warmup_samples=5,
        )
        rng = np.random.RandomState(42)

        repetitive_tokens = rng.randint(0, 100, size=200).tolist()
        for _ in range(100):
            diverse_tokens = rng.randint(0, 50000, size=200).tolist()
            tracker.score_and_update(0, diverse_tokens)
            tracker.score_and_update(1, repetitive_tokens)

        weights = tracker.get_target_weights([0.5, 0.5])
        assert weights[0] > weights[1], (
            f"Diverse dataset weight ({weights[0]:.4f}) should exceed "
            f"repetitive dataset weight ({weights[1]:.4f})"
        )

    def test_cold_start_stays_near_static(self):
        """During warmup, weights should stay close to static weights."""
        tracker = NoveltyTracker(num_datasets=2, warmup_samples=50)
        static = [0.7, 0.3]

        tracker.score_and_update(0, list(range(100)))
        tracker.score_and_update(1, list(range(100, 200)))

        weights = tracker.get_target_weights(static)
        assert abs(weights[0] - 0.7) < 0.1, (
            f"Weight[0]={weights[0]:.4f} should be near 0.7 during warmup"
        )
        assert abs(weights[1] - 0.3) < 0.1, (
            f"Weight[1]={weights[1]:.4f} should be near 0.3 during warmup"
        )

    def test_weight_floor(self):
        """No dataset weight should drop below 1% of its static weight."""
        tracker = NoveltyTracker(
            num_datasets=2, cms_width=4096, warmup_samples=0
        )

        same_tokens = [1, 2, 3, 4, 5] * 40
        for _ in range(200):
            tracker.score_and_update(1, same_tokens)
            tracker.score_and_update(0, np.random.randint(0, 100000, 200).tolist())

        weights = tracker.get_target_weights([0.5, 0.5])
        assert weights[1] > 0, "Repetitive dataset should still have positive weight"
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_bigram_extraction(self):
        """Bigram key packing works correctly."""
        tracker = NoveltyTracker(num_datasets=1)
        keys = tracker._extract_bigram_keys([10, 20, 30])
        assert len(keys) == 2
        assert keys[0] == 10 * 131072 + 20
        assert keys[1] == 20 * 131072 + 30

    def test_empty_and_short_inputs(self):
        """Empty and single-token inputs are handled gracefully."""
        tracker = NoveltyTracker(num_datasets=1)
        assert tracker.score_and_update(0, []) == 0.0
        assert tracker.score_and_update(0, [42]) == 0.0

    def test_numeric_normalization(self):
        """Numeric token IDs should be collapsed so random numbers
        don't inflate novelty scores."""
        # Token IDs 10-19 are "numeric"
        numeric_ids = set(range(10, 20))

        tracker_with = NoveltyTracker(
            num_datasets=2, cms_width=4096,
            warmup_samples=0, numeric_token_ids=numeric_ids,
        )
        tracker_without = NoveltyTracker(
            num_datasets=2, cms_width=4096, warmup_samples=0,
        )

        rng = np.random.RandomState(99)
        # Template: fixed structure with random "numeric" tokens injected
        template = [100, 101, 102]  # non-numeric structure
        for _ in range(50):
            # Same template, different random numbers in positions
            nums = rng.randint(10, 20, size=3).tolist()  # within numeric range
            doc = template + nums + template
            tracker_with.score_and_update(0, doc)
            tracker_without.score_and_update(0, doc)

        # With normalization, the template should be recognized as repetitive
        # (lower novelty). Without normalization, random numbers keep it novel.
        assert tracker_with.dataset_novelty[0] < tracker_without.dataset_novelty[0], (
            f"Normalized novelty ({tracker_with.dataset_novelty[0]:.4f}) should be "
            f"lower than raw ({tracker_without.dataset_novelty[0]:.4f})"
        )

    def test_decay_triggers(self):
        """Decay fires at the configured interval."""
        tracker = NoveltyTracker(
            num_datasets=1, cms_width=1024, decay_interval=10, decay_factor=0.5
        )
        tokens = list(range(50))
        for _ in range(10):
            tracker.score_and_update(0, tokens)

        count = tracker.global_cms.query(0 * 131072 + 1)
        assert count < 10


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
        original_mode = InterleaveDataManager.weighting_mode
        InterleaveDataManager.weighting_mode = "static"

        try:
            sampler = _make_sampler("ds", lambda: _simple_doc())
            manager = InterleaveDataManager(
                samplers=[sampler, sampler],
                weights=[0.8, 0.2],
                tokenizer=tokenizer,
                block_size=128,
            )

            assert not manager._adaptive
            assert not hasattr(manager, "novelty_tracker")
            assert manager.weights == [0.8, 0.2]

            # Fetching a batch should not change weights
            manager.get_batch(batch_size=1)
            assert manager.weights == [0.8, 0.2]
        finally:
            InterleaveDataManager.weighting_mode = original_mode


class TestDynamicMode:
    def setup_method(self):
        _reset_shared_state()

    def test_huge_docs_downweighted(self):
        """Dynamic mode should downweight datasets with huge documents."""
        from praxis.data.datasets.manager import InterleaveDataManager

        tokenizer = _make_tokenizer()
        original_mode = InterleaveDataManager.weighting_mode
        InterleaveDataManager.weighting_mode = "dynamic"

        try:
            call_count = [0, 0]

            def small_doc():
                call_count[0] += 1
                return _simple_doc(f"Short {call_count[0]}")

            def huge_doc():
                call_count[1] += 1
                messages = []
                for j in range(50):
                    messages.append({"role": "user", "content": f"Part {j}"})
                    messages.append({"role": "assistant", "content": f"Reply {j}"})
                return {"messages": messages, "metadata": {"source": "huge"}}

            manager = InterleaveDataManager(
                samplers=[
                    _make_sampler("small", small_doc),
                    _make_sampler("huge", huge_doc),
                ],
                weights=[0.5, 0.5],
                tokenizer=tokenizer,
                block_size=128,
            )

            assert manager._adaptive
            assert not hasattr(manager, "novelty_tracker")

            for _ in range(10):
                manager.get_batch(batch_size=2)

            # Small-doc dataset should have higher weight
            assert manager.weights[0] > manager.weights[1], (
                f"small={manager.weights[0]:.4f} should exceed "
                f"huge={manager.weights[1]:.4f}"
            )
        finally:
            InterleaveDataManager.weighting_mode = original_mode


class TestNoveltyMode:
    def setup_method(self):
        _reset_shared_state()

    def test_novelty_tracker_initialized(self):
        """Novelty mode should create a NoveltyTracker."""
        from praxis.data.datasets.manager import InterleaveDataManager

        tokenizer = _make_tokenizer()
        original_mode = InterleaveDataManager.weighting_mode
        InterleaveDataManager.weighting_mode = "novelty"

        try:
            sampler = _make_sampler("ds", lambda: _simple_doc())
            manager = InterleaveDataManager(
                samplers=[sampler],
                weights=[1.0],
                tokenizer=tokenizer,
                block_size=128,
            )

            assert manager._adaptive
            assert hasattr(manager, "novelty_tracker")
            assert isinstance(manager.novelty_tracker, NoveltyTracker)

            batch = manager.get_batch(batch_size=1)
            assert batch is not None
        finally:
            InterleaveDataManager.weighting_mode = original_mode
