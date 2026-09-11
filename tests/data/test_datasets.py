"""Tests for praxis/data/datasets: network retry and the offline latch, the
Hugging Face sampler's cache fallbacks, the interleave manager's weighting
modes, the novelty tracker, the message-queue packer, the KB sampler and the
governed dataset shape."""

import random
import sqlite3

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

import praxis.data.datasets.huggingface as hf
import praxis.data.datasets.network_retry as nr
from praxis.data import DATASETS
from praxis.data.batch_schedule import BatchSchedule
from praxis.data.datasets.kb import KBDataset
from praxis.data.datasets.manager import InterleaveDataManager
from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.data.datasets.novelty import CountMinSketch, NoveltyTracker
from praxis.data.datasets.weighted import WeightedIterableDataset
from praxis.interface.state.live_metrics import LiveMetrics
from praxis.tasks.weighter import DifficultyTaskLossWeighter
from praxis.tokenizers.chat_templates import chat_format_of

_CLOSED = "Cannot send a request, as the client has been closed."
_OFFLINE_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "PRAXIS_OFFLINE")


def _foreign_error(module):
    """An exception class that reports ``module`` as its origin."""
    cls = type("ForeignError", (Exception,), {})
    cls.__module__ = module
    return cls


_HttpxError = _foreign_error("httpx._transports")
_HubOfflineError = _foreign_error("huggingface_hub.errors")


@pytest.fixture(autouse=True)
def _hub_state(monkeypatch):
    """Start every test online and put the process-wide hub state back after.

    enter_offline_mode writes HF_*_OFFLINE into os.environ and flips the hub
    libraries' live flags. setenv-then-delenv makes monkeypatch record each
    variable, so teardown restores it even when it was unset. The TCP probe is
    stubbed to "down" so no test opens a socket or sleeps in a retry, and the
    sampler's session reset is stubbed so it never closes the real client.
    """
    import datasets.config as datasets_config
    import huggingface_hub.constants as hub_constants

    for var in _OFFLINE_VARS:
        monkeypatch.setenv(var, "0")
        monkeypatch.delenv(var)
    monkeypatch.setattr(nr, "_OFFLINE", False)
    monkeypatch.setattr(nr, "hub_reachable", lambda *a, **k: False)
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", hub_constants.HF_HUB_OFFLINE)
    for flag in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        if hasattr(datasets_config, flag):
            monkeypatch.setattr(datasets_config, flag, getattr(datasets_config, flag))
    monkeypatch.setattr(hf.HuggingfaceDataset, "transport_broken", False)
    monkeypatch.setattr(hf.HuggingfaceDataset, "counts", {})
    monkeypatch.setattr(hf, "reset_hub_session", lambda: None)


# ------------------------------------------------------------------------------
# network_retry.py
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,network,unrecoverable,skippable",
    [
        # httpx raises a plain RuntimeError for a closed client, invisible to
        # module sniffing, and waiting never revives it.
        pytest.param(RuntimeError(_CLOSED), True, True, True, id="closed client"),
        pytest.param(_HttpxError("connect failed"), True, False, True, id="httpx"),
        pytest.param(ConnectionError("reset"), True, False, True, id="connection"),
        # A corrupt or partial cache skips that dataset instead of aborting.
        pytest.param(
            ValueError("Couldn't find cache for tiiuae/falcon-refinedweb"),
            False,
            False,
            True,
            id="cache miss",
        ),
        pytest.param(
            FileNotFoundError(2, "No such file", "/cache/c735.incomplete/info.json"),
            False,
            False,
            True,
            id="partial download",
        ),
        # Programming errors still propagate.
        pytest.param(ValueError("bad value"), False, False, False, id="value"),
        pytest.param(RuntimeError("other"), False, False, False, id="runtime"),
        pytest.param(TypeError("bad config"), False, False, False, id="type"),
        pytest.param(KeyError("messages"), False, False, False, id="key"),
    ],
)
def test_error_classification(exc, network, unrecoverable, skippable):
    assert nr.is_network_error(exc) is network
    assert nr.is_unrecoverable(exc) is unrecoverable
    assert nr.is_skippable_load_error(exc) is skippable


def test_offline_env_detected(monkeypatch):
    assert not nr.hf_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert nr.hf_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert not nr.hf_offline()
    monkeypatch.setenv("PRAXIS_OFFLINE", "true")
    assert nr.hf_offline()


@pytest.mark.parametrize(
    "offline,hub_up,failures,max_attempts,calls,raises",
    [
        # Offline: one attempt, since waiting for connectivity never ends.
        pytest.param(True, True, [RuntimeError(_CLOSED)], 0, 1, True, id="offline"),
        # Load-time calls are bounded; the caller has a cache fallback.
        pytest.param(False, True, [ConnectionError()] * 5, 2, 2, True, id="bounded"),
        # Hub down: raise at once so this caller falls back to its cache.
        pytest.param(False, False, [ConnectionError()], 0, 1, True, id="hub down"),
        # Hub up: a blip is retried until the fetch succeeds.
        pytest.param(False, True, [ConnectionError()] * 2, 0, 3, False, id="blip"),
        # A dead client cannot heal by waiting.
        pytest.param(False, True, [RuntimeError(_CLOSED)], 0, 1, True, id="closed"),
    ],
)
def test_retry_on_network_error(
    monkeypatch, offline, hub_up, failures, max_attempts, calls, raises
):
    if offline:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(nr, "hub_reachable", lambda: hub_up)
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    made = []

    def fetch():
        made.append(1)
        if len(made) <= len(failures):
            raise failures[len(made) - 1]
        return "ok"

    if raises:
        with pytest.raises(type(failures[0])):
            nr.retry_on_network_error(fetch, max_attempts=max_attempts)
    else:
        assert nr.retry_on_network_error(fetch, max_attempts=max_attempts) == "ok"
    assert len(made) == calls
    # Only enter_offline_mode latches; one failing fetch must not take the
    # rest of the mixture offline.
    assert nr._OFFLINE is False


def test_enter_offline_mode_latches_and_notifies():
    nr.enter_offline_mode("test outage")
    assert nr.hf_offline()
    # The events deque is bounded, so check the newest entry, not the length.
    event = LiveMetrics().events[-1]
    assert "OFFLINE" in event["message"] and event["level"] == "warning"


def test_reset_hub_session_recovers_closed_client():
    """A closed huggingface_hub client stays referenced, so every later
    get_session() returns the dead client. reset_hub_session must drop it.
    Reads huggingface_hub's private _GLOBAL_CLIENT, so it breaks on a hub
    upgrade that changes the session cache - which is the point."""
    import huggingface_hub.utils._http as h

    client = h.get_session()
    client.close()
    assert h._GLOBAL_CLIENT is not None and client.is_closed

    nr.reset_hub_session()
    assert h._GLOBAL_CLIENT is None  # recreated lazily
    fresh = h.get_session()
    assert not fresh.is_closed and fresh is not client


# ------------------------------------------------------------------------------
# huggingface.py
# ------------------------------------------------------------------------------


class _FakeStream:
    """A dataset double: yields ``rows``, then raises ``die_with`` if given."""

    def __init__(self, rows=(), die_with=None):
        self.rows, self.die_with = list(rows), die_with

    def shuffle(self, **kwargs):
        return self

    def __iter__(self):
        yield from self.rows
        if self.die_with is not None:
            raise self.die_with


def _docs(text, n=5):
    return [{"text": text}] * n


def _sampler(monkeypatch, fake_load, **config):
    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    return hf.HuggingfaceDataset(
        tokenizer=None, seed=0, config={"path": "fake/ds", **config}
    )


def test_first_hub_failure_falls_back_to_cache(monkeypatch):
    """A streaming load failure reloads THIS dataset from the cache without
    latching the process offline - other datasets must keep streaming."""
    seen = []

    def fake_load(args):
        seen.append(dict(args))
        if args.get("streaming"):
            raise RuntimeError(_CLOSED)
        return _FakeStream()

    sampler = _sampler(monkeypatch, fake_load)
    assert sampler.is_streaming is False
    assert not nr.hf_offline()
    assert seen[0]["streaming"] is True and seen[-1]["streaming"] is False


@pytest.mark.parametrize("latched", [False, True])
def test_uncached_dataset_raises(monkeypatch, latched):
    """Uncached and unreachable: raise so the caller skips it, and leave the
    process latch as it was."""
    if latched:
        nr.enter_offline_mode("test")

    def fake_load(args):
        raise FileNotFoundError("not in cache")

    with pytest.raises(FileNotFoundError):
        _sampler(monkeypatch, fake_load)
    assert nr.hf_offline() is latched


def test_boot_dead_client_gets_one_fresh_load(monkeypatch):
    loads = []

    def fake_load(args):
        loads.append(1)
        if len(loads) == 1:
            raise RuntimeError(_CLOSED)
        return _FakeStream(_docs("ok", 1))

    s = _sampler(monkeypatch, fake_load)
    assert len(loads) == 2 and s.get_document()["messages"]


def test_configured_nonstreaming_still_downloads(monkeypatch):
    """streaming=False in the config is a deliberate one-time download, not a
    fallback, so it must not be forced to local_files_only."""
    seen = []

    def fake_load(args):
        seen.append(dict(args))
        return _FakeStream(_docs("doc", 3))

    _sampler(monkeypatch, fake_load, streaming=False)
    assert seen and all(not a.get("streaming") for a in seen)
    assert all("download_config" not in a for a in seen)


def _dies_midrun(monkeypatch, cache_works):
    """A sampler whose stream dies with an offline error after one document."""

    def fake_load(args):
        if args.get("streaming"):
            return _FakeStream(_docs("live doc", 1), _HubOfflineError("offline"))
        if cache_works:
            return _FakeStream(_docs("cached doc", 3))
        raise FileNotFoundError("not in cache")

    return _sampler(monkeypatch, fake_load)


def test_midrun_offline_falls_back_to_cached_loop(monkeypatch):
    s = _dies_midrun(monkeypatch, cache_works=True)
    assert s.get_document()["messages"]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # hub dies mid-run
    assert s.get_document()["messages"]  # served from the cache
    assert not s._retired and s.is_streaming is False


def test_midrun_offline_retires_quietly_when_uncached(monkeypatch, capsys):
    empty = {"messages": [], "metadata": {}}
    s = _dies_midrun(monkeypatch, cache_works=False)
    assert s.get_document()["messages"]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert s.get_document() == empty
    assert s._retired
    assert "Traceback" not in capsys.readouterr().out
    # Later picks are silent empties - no per-fetch spam.
    for _ in range(5):
        assert s.get_document() == empty
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(
            _FakeStream(_docs("doc before death", 1), RuntimeError(_CLOSED)),
            id="dead client",
        ),
        # Nothing even after the reshuffle: a dead transport, not exhaustion.
        pytest.param(_FakeStream(), id="empty after reshuffle"),
    ],
)
def test_dead_transport_rebuilds_stream(monkeypatch, first):
    loads = []

    def fake_load(args):
        loads.append(1)
        return first if len(loads) == 1 else _FakeStream(_docs("after rebuild"))

    s = _sampler(monkeypatch, fake_load)
    doc = s.get_document()
    if first.rows:  # the live document, then the death
        doc = s.get_document()
    assert doc["messages"] and not s._retired
    assert len(loads) == 2
    assert s._stream_rebuilds == 0  # a healthy fetch resets the budget


def test_exhausted_rebuilds_land_on_cache_without_download(monkeypatch):
    """Rebuilds keep dying, so the source loops the local cache instead of
    retiring - and that cache read can never become a full download."""
    import datasets.config as datasets_config

    seen = []

    def fake_load(args):
        # download_and_prepare can ignore local_files_only; the real guard is
        # the offline flag being live during the load.
        seen.append({**args, "_offline": bool(datasets_config.HF_DATASETS_OFFLINE)})
        if args.get("streaming") is False:
            return _FakeStream(_docs("cached doc", 10))
        return _FakeStream(die_with=RuntimeError(_CLOSED))

    offline_before = datasets_config.HF_DATASETS_OFFLINE
    s = _sampler(monkeypatch, fake_load)
    s._stream_rebuilds = 3  # budget exhausted
    assert s.get_document()["messages"]
    assert not s._retired and s.is_streaming is False

    cache_loads = [a for a in seen if not a.get("streaming")]
    assert cache_loads
    for args in cache_loads:
        assert args["download_config"].local_files_only is True
        assert args["_offline"] is True
    # The fallback forces offline for the load only; it does not latch.
    assert datasets_config.HF_DATASETS_OFFLINE == offline_before


def test_broken_transport_is_global(monkeypatch):
    """One failed rebuild condemns the shared client: later samplers go
    straight to their caches without burning their own rebuild budgets."""
    booting, streaming_loads = [True], []

    def fake_load(args):
        if args.get("streaming") is False:
            return _FakeStream(_docs("cached", 10))
        streaming_loads.append(1)
        if booting[0]:
            return _FakeStream(die_with=RuntimeError(_CLOSED))
        raise RuntimeError(_CLOSED)

    first, second = _sampler(monkeypatch, fake_load), _sampler(monkeypatch, fake_load)
    booting[0] = False
    streaming_loads.clear()

    assert first.get_document()["messages"]
    assert hf.HuggingfaceDataset.transport_broken
    assert len(streaming_loads) <= 2  # one rebuild (frugal, then plain)

    streaming_loads.clear()
    assert second.get_document()["messages"]
    assert streaming_loads == []


@pytest.mark.network
def test_huggingface_dataset_determinism():
    """Same seed, same sequences; a different seed, different ones."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    config = {**DATASETS["minipile-validation"], "streaming": False}

    def sequences(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return hf.HuggingfaceDataset(tokenizer, seed, config).get_sequences(5)

    first = sequences(42)
    assert sequences(42) == first
    assert sequences(43) != first


# ------------------------------------------------------------------------------
# manager.py: tasker mode
# ------------------------------------------------------------------------------
# Sampling driven by the model's learned per-task loss weights: a task the
# weighter (praxis/tasks/weighter.py) deems hard is upweighted in the loss and
# upsampled in the data.


def _tasker_manager(sampler_task_ids, static_weights, task_weights):
    """A manager wired only for the tasker weight calculation - skips the
    heavyweight __init__ (tokenizer, message queue, dataset fetches)."""
    m = object.__new__(InterleaveDataManager)
    m.weighting_mode = "tasker"
    m.samplers = [None] * len(sampler_task_ids)
    m.sampler_task_ids = list(sampler_task_ids)
    m.static_weights = list(static_weights)
    InterleaveDataManager.shared_task_weights = (
        None if task_weights is None else list(task_weights)
    )
    return m


def test_update_task_weights_noops_when_not_armed():
    InterleaveDataManager.update_task_weights([1.0, 2.0, 3.0])
    assert InterleaveDataManager.shared_task_weights is None


def test_update_task_weights_accepts_tensor_and_list():
    InterleaveDataManager.shared_task_weights = [1.0, 1.0]
    InterleaveDataManager.update_task_weights(torch.tensor([1.0, 4.0]))
    assert InterleaveDataManager.shared_task_weights == [1.0, 4.0]
    InterleaveDataManager.update_task_weights([2.0, 0.5])
    assert InterleaveDataManager.shared_task_weights == [2.0, 0.5]


def test_warmup_is_uniform_until_tasker_reports():
    m = _tasker_manager([0, 0, 1], [1.0, 1.0, 1.0], task_weights=None)
    assert m._calculate_target_weights() == [1 / 3, 1 / 3, 1 / 3]


def test_hard_task_is_upsampled():
    m = _tasker_manager([0, 0, 1], [1.0, 1.0, 1.0], task_weights=[1.0, 3.0])
    w = m._calculate_target_weights()
    assert abs(sum(w) - 1.0) < 1e-9  # uniform-floor mix preserves normalization
    assert w[2] > w[0]  # the hard task's dataset dominates
    assert abs(w[0] - w[1]) < 1e-9  # equal datasets within a task stay equal


def test_static_weights_scale_within_a_task():
    m = _tasker_manager([0, 0], [3.0, 1.0], task_weights=[1.0])
    w = m._calculate_target_weights()
    assert w[0] > w[1]


def test_uniform_floor_keeps_easy_task_from_starving():
    m = _tasker_manager([0, 1], [1.0, 1.0], task_weights=[0.0, 5.0])
    w = m._calculate_target_weights()
    assert 0.0 < w[0] < w[1]


def test_refill_in_tasker_mode_adapts_weights_and_logs_metrics(tmp_path, make_sampler):
    """The refill dispatch must run tasker mode: weights move toward the
    learned task targets and the sampling-weights card gets rows."""
    m = InterleaveDataManager(
        [make_sampler("easy_ds", task_type=0), make_sampler("hard_ds", task_type=1)],
        [0.5, 0.5],
        tokenizer=None,
        block_size=64,
        weighting_mode="tasker",
        run_dir=str(tmp_path),
        data_metrics_log_interval=10,
        enable_chat_validation=False,
    )
    InterleaveDataManager.update_task_weights([1.0, 3.0])  # task 1 is hard

    for _ in range(3):
        m._refill_message_queue(min_documents=64)
        m.message_queue.message_queue.clear()  # force the next refill

    assert m.dynamic_weights[1] > m.dynamic_weights[0]
    assert m.sampling_count > 0
    m.data_metrics_logger.close()
    con = sqlite3.connect(tmp_path / "data_metrics.db")
    rows = con.execute(
        "select count(*), max(sampling_weights) from data_metrics"
    ).fetchone()
    con.close()
    assert rows[0] > 0
    assert "hard_ds" in rows[1]


def test_difficulty_weighter_output_upsamples_its_hard_task():
    """A DifficultyTaskLossWeighter's effective_weights, routed through the
    sampler, upsample the task it finds hard."""
    weighter = DifficultyTaskLossWeighter(gamma=1.0)
    task_ids = torch.tensor([[0, 0, 1, 1]])
    losses = torch.tensor([[0.1, 0.1, 5.0, 5.0]])
    for _ in range(20):  # let the EMA settle
        weighter.observe(task_ids, losses)
    eff = weighter.effective_weights()
    assert eff[1] > eff[0]

    m = _tasker_manager([0, 1], [1.0, 1.0], task_weights=[float(x) for x in eff])
    w = m._calculate_target_weights()
    assert w[1] > w[0]


# ------------------------------------------------------------------------------
# manager.py: dynamic mode and determinism
# ------------------------------------------------------------------------------


def test_dynamic_mode_downweights_huge_docs(default_tokenizer, make_sampler):
    def huge_doc():
        messages = []
        for j in range(50):
            messages.append({"role": "user", "content": f"Part {j}"})
            messages.append({"role": "assistant", "content": f"Reply {j}"})
        return {"messages": messages, "metadata": {}}

    manager = InterleaveDataManager(
        samplers=[make_sampler("small"), make_sampler("huge", huge_doc)],
        weights=[0.5, 0.5],
        tokenizer=default_tokenizer,
        block_size=128,
        weighting_mode="dynamic",
    )
    for _ in range(10):
        manager.get_batch(batch_size=2)
    assert manager.weights[0] > manager.weights[1], manager.weights


@pytest.mark.network
def test_interleave_data_manager_determinism():
    """Two managers over the same seeded dataset build identical batches,
    including at a sequence multiplier."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    config = {**DATASETS["minipile-validation"], "streaming": False}

    def seed_all():
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

    seed_all()
    managers = [
        InterleaveDataManager(
            [hf.HuggingfaceDataset(tokenizer, 42, config)],
            [1.0],
            tokenizer,
            block_size=128,
        )
        for _ in range(2)
    ]
    for kwargs in ({"batch_size": 2}, {"batch_size": 4, "sequence_multiplier": 2}):
        batches = []
        for manager in managers:
            seed_all()
            batches.append(manager.get_batch(**kwargs)["batch"])
        assert len(batches[0]) == len(batches[1])
        for a, b in zip(*batches):
            torch.testing.assert_close(a, b)


# ------------------------------------------------------------------------------
# novelty.py
# ------------------------------------------------------------------------------


class TestCountMinSketch:
    def test_add_and_query(self):
        cms = CountMinSketch(width=1024, depth=4)
        cms.add(42, count=5)
        cms.add(42, count=3)
        assert cms.query(42) == 8

    def test_decay(self):
        cms = CountMinSketch(width=1024, depth=4)
        cms.add(10, count=100)
        before = cms.query(10)
        cms.decay(0.5)
        assert cms.query(10) == int(before * 0.5)

    def test_batch_operations(self):
        cms = CountMinSketch(width=4096, depth=4)
        cms.add_batch(np.array([1, 2, 3, 1, 2, 1], dtype=np.int64))
        counts = cms.query_batch(np.array([1, 2, 3, 4], dtype=np.int64))
        assert list(counts) == [3, 2, 1, 0]  # key 4 never inserted


class TestNoveltyTracker:
    def test_diverse_vs_repetitive(self):
        """A dataset producing diverse documents keeps a higher weight than one
        producing identical documents."""
        tracker = NoveltyTracker(num_datasets=2, cms_width=4096, warmup_samples=5)
        rng = np.random.RandomState(42)
        repetitive = rng.randint(0, 100, size=200).tolist()
        for _ in range(30):
            tracker.score_and_update(0, rng.randint(0, 50000, size=200).tolist())
            tracker.score_and_update(1, repetitive)
        weights = tracker.get_target_weights([0.5, 0.5])
        assert weights[0] > weights[1], weights

    def test_cold_start_stays_near_uniform(self):
        """During warmup, weights blend from uniform toward novelty-driven."""
        tracker = NoveltyTracker(num_datasets=2, warmup_samples=50)
        tracker.score_and_update(0, list(range(100)))
        tracker.score_and_update(1, list(range(100, 200)))
        weights = tracker.get_target_weights([0.7, 0.3])
        # 2 of 50 warmup docs: the blend factor is ~0.04.
        assert abs(weights[0] - 0.5) < 0.1 and abs(weights[1] - 0.5) < 0.1, weights

    def test_weight_floor(self):
        """A dataset whose novelty collapses keeps the floor share: 1% of
        uniform before renormalization."""
        tracker = NoveltyTracker(num_datasets=2, cms_width=4096, warmup_samples=0)
        rng = np.random.RandomState(0)
        same = [1, 2, 3, 4, 5] * 40
        for _ in range(50):
            tracker.score_and_update(1, same)
            tracker.score_and_update(0, rng.randint(0, 100000, 200).tolist())
        weights = tracker.get_target_weights([0.5, 0.5])
        assert abs(sum(weights) - 1.0) < 1e-6
        assert weights[1] >= 0.01 / 2 - 1e-9, weights

    def test_bigram_extraction(self):
        tracker = NoveltyTracker(num_datasets=1)
        keys = tracker._extract_bigram_keys([10, 20, 30])
        assert list(keys) == [10 * 131072 + 20, 20 * 131072 + 30]

    def test_empty_and_short_inputs(self):
        tracker = NoveltyTracker(num_datasets=1)
        assert tracker.score_and_update(0, []) == 0.0
        assert tracker.score_and_update(0, [42]) == 0.0

    def test_numeric_normalization(self):
        """Numeric token ids are collapsed, so a fixed template with random
        numbers in it reads as repetitive rather than novel."""
        kwargs = dict(num_datasets=2, cms_width=4096, warmup_samples=0)
        normalized = NoveltyTracker(numeric_token_ids=set(range(10, 20)), **kwargs)
        raw = NoveltyTracker(**kwargs)
        rng = np.random.RandomState(99)
        template = [100, 101, 102]
        for _ in range(50):
            doc = template + rng.randint(10, 20, size=3).tolist() + template
            normalized.score_and_update(0, doc)
            raw.score_and_update(0, doc)
        assert normalized.dataset_novelty[0] < raw.dataset_novelty[0]

    def test_decay_triggers(self):
        tracker = NoveltyTracker(
            num_datasets=1, cms_width=1024, decay_interval=10, decay_factor=0.5
        )
        for _ in range(10):
            tracker.score_and_update(0, list(range(50)))
        assert tracker.global_cms.query(0 * 131072 + 1) < 10


# ------------------------------------------------------------------------------
# message_queue.py
# ------------------------------------------------------------------------------

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]

MULTIBYTE = [
    {"role": "system", "content": "“quoted”"},
    {"role": "user", "content": "Calculate √1156 \U0001f600"},
    {"role": "assistant", "content": "— the answer is 34."},
    {"role": "user", "content": "and été?"},
    {"role": "assistant", "content": "Summer."},
]


def test_prose_survives_the_packer(prose_tokenizer):
    """The packer passes `omit_leading_bos` for docs appended mid-sequence. In
    prose the boundary IS the separator, so the template must ignore that flag
    rather than run one document's role name into the previous one's last
    word - and every id in the stream is a plain byte."""
    manager = MessageQueueManager(tokenizer=prose_tokenizer, block_size=512)
    for _ in range(6):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    batch = manager.get_batch(batch_size=2)

    assert (
        chat_format_of(prose_tokenizer).document_separator_id(prose_tokenizer) is None
    )
    assert len(batch["batch"]) == 2
    for seq, mask in zip(batch["batch"], batch["assistant_mask"]):
        assert seq.shape == mask.shape
        assert 0 <= int(seq.min()) and int(seq.max()) < 256
        # Doc-to-doc seams read as plain prose; block_ids carry the seam.
        text = prose_tokenizer.decode(seq, skip_special_tokens=False)
        assert "Tokyo.\n\nsystem\n\n" in text
    assert manager.get_validation_stats()["documents_skipped"] == 0


def test_packer_uses_the_exact_mask(prose_tokenizer):
    """No assistant content may be dropped and no prompt content admitted, on
    multi-byte text."""
    manager = MessageQueueManager(tokenizer=prose_tokenizer, block_size=4096)
    manager.add_document({"messages": MULTIBYTE, "metadata": {}})
    batch = manager.get_batch(batch_size=1)
    seq, mask = batch["batch"][0], batch["assistant_mask"][0]
    trained = prose_tokenizer.decode(
        [int(t) for t, m in zip(seq, mask) if m], skip_special_tokens=False
    )
    assert "— the answer is 34." in trained
    assert "Summer." in trained
    assert "1156" not in trained


@pytest.mark.parametrize("fmt", ["prose", "default"])
def test_packer_emits_block_ids_for_every_document(
    fmt, prose_tokenizer, default_tokenizer
):
    """block_ids segment the local encoder's attention so it cannot read across
    unrelated documents. Only the packer knows where documents meet, so this
    holds whether the format writes a separator id or nothing at all."""
    tok = prose_tokenizer if fmt == "prose" else default_tokenizer
    manager = MessageQueueManager(
        tokenizer=tok, block_size=4096, enable_chat_validation=False
    )
    for _ in range(3):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    batch = manager.get_batch(batch_size=1)
    seq, blocks = batch["batch"][0], batch["block_ids"][0]

    assert blocks.shape == seq.shape
    # Three documents, each its own block, plus a block for the pad tail.
    assert sorted(set(blocks.tolist())) == [1, 2, 3, 4]
    # Blocks are contiguous runs: a document never resumes after another.
    runs = [int(blocks[0])]
    for prev, cur in zip(blocks[:-1], blocks[1:]):
        if cur != prev:
            runs.append(int(cur))
    assert runs == sorted(set(runs))


def test_prose_documents_end_on_their_own_text(prose_tokenizer):
    """No separator is appended, so a document ends where its text ends: the
    trailing boundary the template emits is the halt signal, and it is inside
    the generated turn's span."""
    manager = MessageQueueManager(
        tokenizer=prose_tokenizer, block_size=4096, enable_chat_validation=False
    )
    ends_generated = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    ids, mask = manager._tokenize_doc(
        {"messages": ends_generated, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("yo\n\n")
    assert int(mask[-1]) == 1  # the halting boundary is a trained target

    ends_prompt = [{"role": "user", "content": "unanswered"}]
    ids, mask = manager._tokenize_doc(
        {"messages": ends_prompt, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("unanswered\n\n")
    assert int(mask[-1]) == 0


def test_per_document_tokenization(default_tokenizer):
    """Documents are tokenized one by one, and a document appended mid-sequence
    drops its leading [BOS] (the previous document's [SEP] already opens it)."""
    manager = MessageQueueManager(default_tokenizer, block_size=256)
    for tag in ("ONE", "TWO"):
        manager.add_document(
            {
                "messages": [
                    {"role": "user", "content": f"DOCUMENT_{tag}"},
                    {"role": "assistant", "content": f"Response {tag}"},
                ],
                "metadata": {},
            }
        )
    packed = manager.get_batch(batch_size=1, sequence_multiplier=1)["batch"][0]
    text = default_tokenizer.decode(packed, skip_special_tokens=False)
    assert "DOCUMENT_ONE" in text and "DOCUMENT_TWO" in text
    # Doc 1: one [BOS] per message. Doc 2: only its assistant message's.
    assert text.count("[BOS]") == 3, text


@pytest.mark.parametrize("enabled", [True, False])
def test_message_queue_validation_stats(default_tokenizer, enabled):
    queue = MessageQueueManager(
        default_tokenizer, block_size=512, enable_chat_validation=enabled
    )
    assert (queue.chat_validator is not None) is enabled
    queue.add_document(
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "metadata": {},
        }
    )
    queue.get_batch(batch_size=1, sequence_multiplier=1)
    stats = queue.get_validation_stats()
    assert stats["documents_validated"] == (1 if enabled else 0)
    assert stats["documents_failed"] == 0 and stats["documents_skipped"] == 0


@pytest.mark.parametrize("strict", [True, False])
def test_failed_validation_raises_in_strict_mode_else_skips(
    default_tokenizer, monkeypatch, strict
):
    queue = MessageQueueManager(
        default_tokenizer,
        block_size=512,
        enable_chat_validation=True,
        strict_chat_validation=strict,
    )
    verdicts = [(False, "bad")]  # the first document fails, the rest pass
    monkeypatch.setattr(
        queue.chat_validator,
        "validate_and_report",
        lambda *a, **k: verdicts.pop() if verdicts else (True, ""),
    )
    for _ in range(2):
        queue.add_document({"messages": CONVERSATION, "metadata": {}})
    if strict:
        with pytest.raises(ValueError, match="validation failed"):
            queue.get_batch(batch_size=1)
    else:
        queue.get_batch(batch_size=1)
        assert queue.get_validation_stats()["documents_skipped"] == 1


# ------------------------------------------------------------------------------
# kb.py
# ------------------------------------------------------------------------------


@pytest.fixture
def docs_tree(tmp_path, monkeypatch):
    """A three-page docs/ tree standing in for the generated wiki."""
    import praxis.kb.sources as sources

    (tmp_path / "docs").mkdir()
    for i in range(3):
        (tmp_path / "docs" / f"page{i}.md").write_text(f"# Page {i}\n\nBody {i}.\n")
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    return tmp_path


def test_kb_dataset_yields_seeded_docs(docs_tree):
    def draw(seed):
        ds = KBDataset(tokenizer=None, seed=seed, config={"sources": ["docs"]})
        return ds.get_sequences(5)

    seqs = draw(7)
    assert all(isinstance(s, str) and s for s in seqs)
    assert draw(7) == seqs


def test_kb_dataset_reloads_on_exhaustion(docs_tree):
    ds = KBDataset(None, 0, {"sources": ["docs"]})
    ds._load_epoch()
    n = len(ds._epoch)
    assert n == 3
    assert all(ds.get_sequences(n + 2))  # one wrap, no stall


# ------------------------------------------------------------------------------
# weighted.py: the governed shape
# ------------------------------------------------------------------------------


def _weighted(batch_size, governed):
    ds = object.__new__(WeightedIterableDataset)
    ds.batch_size, ds.sequence_multiplier_tiers, ds.governed = batch_size, (), governed
    return ds


def test_dataset_falls_back_to_static_shape_without_a_governor():
    # Governed, but no governor has enabled the schedule.
    assert _weighted(16, governed=True)._next_shape() == (16, 1)


def test_validation_loader_keeps_a_fixed_shape():
    """The val loader is the same class. Following the governed plan would make
    val loss incomparable across steps and, because the plan counts
    microbatches to find a cycle boundary, shift the training cycle."""
    train, val = _weighted(64, governed=True), _weighted(64, governed=False)
    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    assert val._next_shape() == (64, 1)
    assert BatchSchedule.current() is None  # val did not open a cycle
    assert train._next_shape() == (4, 1)


def test_validation_draws_do_not_shift_the_training_cycle():
    train, val = _weighted(64, governed=True), _weighted(64, governed=False)
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=())  # accum 4
    train._next_shape()
    train._next_shape()
    for _ in range(10):  # a whole validation pass mid-cycle
        val._next_shape()
    assert BatchSchedule._micro_index == 2  # still two into the open cycle
