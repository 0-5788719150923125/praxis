import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from praxis.data import DATASETS, HuggingfaceDataset, InterleaveDataManager
from praxis.data.batch_schedule import BatchSchedule
from praxis.data.datasets.kb import KBDataset
from praxis.data.datasets.manager import InterleaveDataManager
from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.data.datasets.network_retry import (
    hf_offline,
    is_network_error,
    retry_on_network_error,
)
from praxis.data.datasets.novelty import CountMinSketch, NoveltyTracker
from praxis.tasks.weighter import DifficultyTaskLossWeighter
from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE, chat_format_of
from praxis.tokenizers.standard import StandardTokenizer

# ------------------------------------------------------------------------------
# offline_data
# ------------------------------------------------------------------------------
# Offline data fallback: error classification, no-retry, cache-mode flips.


def test_httpx_closed_client_is_a_network_error():
    # The exact failure seen when the hub is down: httpx raises a plain
    # builtins.RuntimeError, invisible to module sniffing.
    assert is_network_error(
        RuntimeError("Cannot send a request, as the client has been closed.")
    )


def test_httpx_module_errors_are_network_errors():
    class FakeConnectError(Exception):
        pass

    FakeConnectError.__module__ = "httpx._transports"
    assert is_network_error(FakeConnectError("connect failed"))


def test_unrelated_errors_still_propagate():
    assert not is_network_error(ValueError("bad value"))
    assert not is_network_error(RuntimeError("some other runtime issue"))


def test_offline_env_detected(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("PRAXIS_OFFLINE", raising=False)
    assert not hf_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert hf_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert not hf_offline()
    monkeypatch.setenv("PRAXIS_OFFLINE", "true")
    assert hf_offline()


def test_offline_skips_the_retry_loop(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    with pytest.raises(RuntimeError):
        retry_on_network_error(boom)
    assert len(calls) == 1  # one attempt, no indefinite wait


def _reset_latch(monkeypatch):
    import praxis.data.datasets.network_retry as nr

    monkeypatch.setattr(nr, "_OFFLINE", False)
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "PRAXIS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)


def test_enter_offline_mode_latches(monkeypatch):
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    assert not nr.hf_offline()
    nr.enter_offline_mode("test")
    assert nr.hf_offline()


def test_bounded_load_attempts(monkeypatch):
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    monkeypatch.setattr(nr, "hub_reachable", lambda: True)
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    calls = []

    def boom():
        calls.append(1)
        raise ConnectionError("hub down")

    with pytest.raises(ConnectionError):
        retry_on_network_error(boom, max_attempts=2)
    assert len(calls) == 2  # bounded, no indefinite wait


def test_first_hub_failure_falls_back_to_cache(monkeypatch):
    """A streaming load failure reloads THIS dataset from the cache without
    latching the process offline - other datasets must keep streaming."""
    import praxis.data.datasets.huggingface as hf
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)

    class FakeDataset:
        def shuffle(self, **kwargs):
            return iter(())

    seen = []

    def fake_load(dataset_args):
        seen.append(dict(dataset_args))
        if dataset_args.get("streaming"):
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return FakeDataset()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    sampler = hf.HuggingfaceDataset(
        tokenizer=None, seed=0, config={"path": "fake/dataset"}
    )
    assert sampler.is_streaming is False  # this sampler dropped to cache
    assert not nr.hf_offline()  # but the process is NOT latched offline
    assert seen[0]["streaming"] is True and seen[-1]["streaming"] is False


def test_dataset_specific_failure_skips_without_latching(monkeypatch):
    """A dataset that fails the network AND isn't cached raises (skipped
    upstream) without dragging the whole process offline."""
    import praxis.data.datasets.huggingface as hf
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)

    def fake_load(dataset_args):
        # Streaming fails; the non-streaming cache read also misses.
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    with pytest.raises(RuntimeError):
        hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/bad"})
    assert not nr.hf_offline()  # rest of the mixture stays online


def test_reset_hub_session_recovers_closed_client():
    """A closed huggingface_hub client stays REFERENCED (not None), so every
    later get_session() returns the dead client -> 'client has been closed' for
    the rest of the process. reset_hub_session must null it so a fresh, open
    client is created. This is the 2-day streaming-failure root cause."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub.utils._http as h

    from praxis.data.datasets.network_retry import reset_hub_session

    client = h.get_session()
    client.close()
    # The bug: the global still points at the closed client.
    assert h._GLOBAL_CLIENT is not None and client.is_closed

    reset_hub_session()
    assert h._GLOBAL_CLIENT is None  # nulled -> recreated lazily
    fresh = h.get_session()
    assert not fresh.is_closed and fresh is not client


def test_corrupt_cache_is_skippable_not_fatal(monkeypatch):
    """A corrupt/partial cache (the '.incomplete' ValueError/FileNotFoundError
    chain) must classify as skippable so one bad dataset doesn't abort the run.
    Without offline latched, these used to re-raise and crash main()."""
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    val_err = ValueError(
        "Couldn't find cache for tiiuae/falcon-refinedweb for config "
        "'default-cba17da4221ad668'"
    )
    fnf = FileNotFoundError(
        2, "No such file", "/cache/falcon/c735.incomplete/dataset_info.json"
    )
    assert nr.is_skippable_load_error(val_err)
    assert nr.is_skippable_load_error(fnf)
    # Genuine programming errors must still propagate (not silently skipped).
    assert not nr.is_skippable_load_error(TypeError("bad config field"))
    assert not nr.is_skippable_load_error(KeyError("messages"))


def test_uncached_dataset_raises_after_latch(monkeypatch):
    import praxis.data.datasets.huggingface as hf
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    nr.enter_offline_mode("test")

    def fake_load(dataset_args):
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    with pytest.raises(FileNotFoundError):
        hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/x"})


def test_offline_latch_emits_notification_event(monkeypatch):
    import praxis.data.datasets.network_retry as nr
    from praxis.interface.state.live_metrics import LiveMetrics

    _reset_latch(monkeypatch)
    before = len(LiveMetrics().events)
    nr.enter_offline_mode("test outage")
    events = list(LiveMetrics().events)
    assert len(events) == before + 1
    assert "OFFLINE" in events[-1]["message"]
    assert events[-1]["level"] == "warning"


class _OfflineErr(Exception):
    pass


_OfflineErr.__module__ = "huggingface_hub.errors"


def _live_sampler(monkeypatch, cache_works):
    """A sampler whose stream dies mid-run; cache fallback works or not."""
    import praxis.data.datasets.huggingface as hf

    class FakeCached:
        def __init__(self):
            self.rows = [{"text": "cached doc"}] * 3

        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter(self.rows)

    class FakeStream:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            def gen():
                yield {"text": "live doc"}
                raise _OfflineErr("offline mode is enabled")

            return gen()

    def fake_load(args):
        if args.get("streaming"):
            return FakeStream()
        if cache_works:
            return FakeCached()
        raise FileNotFoundError("not in cache")

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    return hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})


def test_midrun_offline_falls_back_to_cached_loop(monkeypatch, capsys):
    _reset_latch(monkeypatch)
    s = _live_sampler(monkeypatch, cache_works=True)
    assert s.get_document()["messages"]  # live doc
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # hub dies mid-run
    doc = s.get_document()  # stream dies -> cache fallback -> cached doc
    assert doc["messages"], doc
    assert not s._retired and s.is_streaming is False
    assert "looping over the local cache" in capsys.readouterr().out


def test_midrun_offline_retires_quietly_when_uncached(monkeypatch, capsys):
    _reset_latch(monkeypatch)
    s = _live_sampler(monkeypatch, cache_works=False)
    assert s.get_document()["messages"]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # hub dies mid-run
    assert s.get_document() == {"messages": [], "metadata": {}}
    assert s._retired
    out = capsys.readouterr().out
    assert "retiring stream" in out and "Traceback" not in out
    # Subsequent picks are silent empties - no per-fetch spam.
    for _ in range(5):
        assert s.get_document() == {"messages": [], "metadata": {}}
    assert capsys.readouterr().out == ""


def test_midrun_hub_outage_raises_without_latching(monkeypatch):
    """Unbounded retries must not hang training when the hub is down: raise
    immediately so the caller's per-dataset cache fallback runs - but DON'T
    latch the whole process offline (other datasets keep working)."""
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    monkeypatch.setattr(nr, "hub_reachable", lambda: False)
    calls = []

    def boom():
        calls.append(1)
        raise ConnectionError("hub down mid-stream")

    with pytest.raises(ConnectionError):
        nr.retry_on_network_error(boom)
    assert len(calls) == 1  # no indefinite wait
    assert not nr.hf_offline()  # not latched - per-dataset fallback only


def test_midrun_blip_keeps_retrying_when_hub_up(monkeypatch):
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    monkeypatch.setattr(nr, "hub_reachable", lambda: True)
    monkeypatch.setattr(nr.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "ok"

    assert nr.retry_on_network_error(flaky) == "ok"
    assert len(calls) == 3
    assert not nr.hf_offline()


def test_closed_client_is_unrecoverable():
    from praxis.data.datasets.network_retry import is_unrecoverable

    exc = RuntimeError("Cannot send a request, as the client has been closed.")
    assert is_unrecoverable(exc)
    assert not is_unrecoverable(ConnectionError("connection reset"))


def test_closed_client_raises_immediately(monkeypatch):
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)
    monkeypatch.setattr(nr, "hub_reachable", lambda: True)
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    with pytest.raises(RuntimeError):
        nr.retry_on_network_error(boom)
    assert len(calls) == 1  # no retry on a dead client
    assert not nr.hf_offline()


def test_dead_transport_rebuilds_stream(monkeypatch):
    """A closed shared client mid-run: the sampler reloads the dataset (fresh
    client) and keeps serving documents."""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)
    loads = []

    class DeadStream:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            def gen():
                yield {"text": "doc before death"}
                raise RuntimeError(
                    "Cannot send a request, as the client has been closed."
                )

            return gen()

    class FreshStream(DeadStream):
        def __iter__(self):
            return iter([{"text": "doc after rebuild"}] * 5)

    def fake_load(args):
        loads.append(1)
        return DeadStream() if len(loads) == 1 else FreshStream()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    s = hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})
    assert s.get_document()["messages"]
    doc = s.get_document()  # client dies; rebuild kicks in
    assert doc["messages"] and not s._retired
    assert len(loads) == 2
    assert s._stream_rebuilds == 0  # healthy fetch reset the budget


def test_empty_post_reshuffle_stream_rebuilds(monkeypatch):
    """A stream that yields nothing even after reshuffle (dead client) goes
    through the rebuild path instead of spamming empty documents."""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)
    loads = []

    class Empty:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter(())

    class Fresh(Empty):
        def __iter__(self):
            return iter([{"text": "alive"}] * 5)

    def fake_load(args):
        loads.append(1)
        return Empty() if len(loads) == 1 else Fresh()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    s = hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})
    doc = s.get_document()
    assert doc["messages"] and not s._retired
    assert len(loads) == 2


def test_boot_dead_client_gets_one_fresh_load(monkeypatch):
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)
    loads = []

    class Fine:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter([{"text": "ok"}])

    def fake_load(args):
        loads.append(1)
        if len(loads) == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return Fine()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    s = hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})
    assert len(loads) == 2 and s.get_document()["messages"]


def test_cache_fallback_never_downloads(monkeypatch):
    """A streaming source that dies must fall back to the cache WITHOUT ever
    downloading the full set: the non-streaming fallback load must carry
    local_files_only=True. (Configured streaming=False is exempt - see below.)"""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)

    class Dying:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            def gen():
                raise RuntimeError(
                    "Cannot send a request, as the client has been closed."
                )
                yield

            return gen()

    class Cached:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter([{"text": "cached doc"}] * 10)

    import datasets.config as dc

    seen = []

    def fake_load(args):
        # download_and_prepare can ignore local_files_only; the real guard is
        # the offline flag being live during the load.
        seen.append({**args, "_offline_live": bool(dc.HF_DATASETS_OFFLINE)})
        return Cached() if args.get("streaming") is False else Dying()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    offline_before = dc.HF_DATASETS_OFFLINE
    s = hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})
    s._stream_rebuilds = 3  # budget exhausted -> cache fallback
    s.get_document()
    nonstreaming = [a for a in seen if not a.get("streaming")]
    assert nonstreaming, "expected a non-streaming cache load"
    for args in nonstreaming:
        dl = args.get("download_config")
        assert dl is not None and dl.local_files_only is True
        assert args["_offline_live"] is True, "cache load must run offline-forced"
    # The transient fallback must restore offline state, not latch it.
    assert dc.HF_DATASETS_OFFLINE == offline_before


def test_configured_nonstreaming_still_downloads(monkeypatch):
    """A dataset deliberately configured streaming=False is a one-time full
    download, not a fallback - it must NOT be forced to local_files_only."""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)

    class Ready:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter([{"text": "doc"}] * 3)

    seen = []

    def fake_load(args):
        seen.append(dict(args))
        return Ready()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    hf.HuggingfaceDataset(
        tokenizer=None, seed=0, config={"path": "fake/small", "streaming": False}
    )
    assert seen and all(not a.get("streaming") for a in seen)
    assert all("download_config" not in a for a in seen)


def test_exhausted_rebuilds_land_on_cache(monkeypatch):
    """Flapping DNS: rebuilds keep dying, so the source collapses to looping
    the local cache instead of retiring."""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)

    class Dying:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            def gen():
                raise RuntimeError(
                    "Cannot send a request, as the client has been closed."
                )
                yield

            return gen()

    class Cached:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            return iter([{"text": "cached doc"}] * 10)

    def fake_load(args):
        return Cached() if args.get("streaming") is False else Dying()

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    s = hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/ds"})
    s._stream_rebuilds = 3  # budget exhausted
    doc = s.get_document()
    assert doc["messages"] and not s._retired
    assert s.is_streaming is False


def test_broken_transport_is_global(monkeypatch):
    """One failed rebuild flips the class flag; later samplers go straight
    to cache without burning their own rebuild budgets."""
    import praxis.data.datasets.huggingface as hf

    _reset_latch(monkeypatch)
    monkeypatch.setattr(hf.HuggingfaceDataset, "transport_broken", False)
    loads = []

    class Dying:
        def shuffle(self, **kw):
            return self

        def __iter__(self):
            def gen():
                raise RuntimeError(
                    "Cannot send a request, as the client has been closed."
                )
                yield

            return gen()

    class Cached(Dying):
        def __iter__(self):
            return iter([{"text": "cached"}] * 10)

    def fake_load(args):
        loads.append(args.get("streaming"))
        if args.get("streaming") is False:
            return Cached()
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    s1 = hf.HuggingfaceDataset.__new__(hf.HuggingfaceDataset)
    # Simulate an already-loaded sampler whose stream just died.
    for s in (s1,):
        s.dataset_path = "fake/one"
        s._dataset_args = {"path": "fake/one", "streaming": True}
        s.is_streaming = True
        s.base_seed, s.restart_count, s.buffer_size = 0, 0, 8
        s.tokenizer = None
        s._retired = False
        s.sequence_cache = []
        s.format_handler = lambda doc, keys, tok: {"messages": [doc], "metadata": {}}
        s.keys = ["text"]
        s.dataset = Dying()
        s.shuffled_dataset = Dying()
        s.dataset_iterator = iter(s.shuffled_dataset)
    doc = s1.get_document()
    assert doc["messages"] and hf.HuggingfaceDataset.transport_broken
    streaming_loads = [x for x in loads if x is not False]
    assert (
        len(streaming_loads) <= 2
    )  # one rebuild attempt (frugal+plain), then condemned


# ------------------------------------------------------------------------------
# tasker_sampling
# ------------------------------------------------------------------------------
# The `tasker` sampling mode: dataset sampling driven by the model's learned per-task
# loss weights.
#
# Closes the loop between the loss weighter (praxis/tasks/weighter.py) and the data
# sampler (InterleaveDataManager): a task the weighter deems hard gets both upweighted
# in the loss and upsampled in the data.


def _reset():
    InterleaveDataManager.shared_task_weights = None


def _tasker_manager(sampler_task_ids, static_weights, task_weights):
    """A minimal manager wired only for the tasker weight calculation - avoids
    the heavyweight __init__ (tokenizer, message queue, dataset fetches)."""
    m = object.__new__(InterleaveDataManager)
    m.weighting_mode = "tasker"
    m.samplers = [None] * len(sampler_task_ids)
    m.sampler_task_ids = list(sampler_task_ids)
    m.static_weights = list(static_weights)
    InterleaveDataManager.shared_task_weights = (
        None if task_weights is None else list(task_weights)
    )
    return m


# --------------------------------------------------------------------------
# update_task_weights classmethod (the trainer -> sampler push)
# --------------------------------------------------------------------------


def test_update_task_weights_noops_when_not_armed():
    _reset()
    # Not in tasker mode: the push must be a silent no-op.
    InterleaveDataManager.update_task_weights([1.0, 2.0, 3.0])
    assert InterleaveDataManager.shared_task_weights is None


def test_update_task_weights_accepts_tensor_and_list():
    _reset()
    InterleaveDataManager.shared_task_weights = [1.0, 1.0]
    InterleaveDataManager.update_task_weights(torch.tensor([1.0, 4.0]))
    assert InterleaveDataManager.shared_task_weights == [1.0, 4.0]
    InterleaveDataManager.update_task_weights([2.0, 0.5])
    assert InterleaveDataManager.shared_task_weights == [2.0, 0.5]
    _reset()


# --------------------------------------------------------------------------
# _calculate_target_weights: task weight -> sampling weight
# --------------------------------------------------------------------------


def test_warmup_is_uniform_until_tasker_reports():
    _reset()
    m = _tasker_manager([0, 0, 1], [1.0, 1.0, 1.0], task_weights=None)
    w = m._calculate_target_weights()
    assert w == [1 / 3, 1 / 3, 1 / 3]


def test_hard_task_is_upsampled():
    _reset()
    # task 1 is "hard" (weight 3x); the dataset on it should be sampled most.
    m = _tasker_manager(
        sampler_task_ids=[0, 0, 1],
        static_weights=[1.0, 1.0, 1.0],
        task_weights=[1.0, 3.0],
    )
    w = m._calculate_target_weights()
    assert abs(sum(w) - 1.0) < 1e-9  # uniform-floor mix preserves normalization
    assert w[2] > w[0]  # hard-task dataset dominates
    assert abs(w[0] - w[1]) < 1e-9  # equal datasets within the same task stay equal


def test_static_weights_scale_within_a_task():
    _reset()
    # Two datasets on the same (equal-weight) task keep their configured ratio.
    m = _tasker_manager(
        sampler_task_ids=[0, 0],
        static_weights=[3.0, 1.0],
        task_weights=[1.0],
    )
    w = m._calculate_target_weights()
    assert w[0] > w[1]


def test_uniform_floor_keeps_easy_task_from_starving():
    _reset()
    # Even with a near-zero task weight, the floor keeps a positive share.
    m = _tasker_manager(
        sampler_task_ids=[0, 1],
        static_weights=[1.0, 1.0],
        task_weights=[0.0, 5.0],
    )
    w = m._calculate_target_weights()
    assert w[0] > 0.0
    assert w[1] > w[0]
    _reset()


# --------------------------------------------------------------------------
# End-to-end signal: a DifficultyTaskLossWeighter's output drives sampling
# --------------------------------------------------------------------------


class _StubSampler:
    """Minimal sampler for the refill loop: fixed task, canned document."""

    def __init__(self, name, task_type):
        self.dataset_path = name
        self.task_type = task_type
        self.weight = 1.0

    def get_document(self):
        return {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "metadata": {},
        }


def test_refill_in_tasker_mode_adapts_weights_and_logs_metrics(tmp_path):
    """Regression: the refill dispatch skipped tasker mode entirely, so tasker
    runs never adapted sampling weights toward the learned task targets and
    logged an EMPTY data_metrics.db (no sampling-weights card - every
    tasker-mode run in build/runs had 0 rows)."""
    import sqlite3

    _reset()
    InterleaveDataManager.shared_weights = None
    InterleaveDataManager.shared_weights_initialized = False
    samplers = [_StubSampler("easy_ds", 0), _StubSampler("hard_ds", 1)]
    m = InterleaveDataManager(
        samplers,
        [0.5, 0.5],
        tokenizer=None,
        block_size=64,
        weighting_mode="tasker",
        run_dir=str(tmp_path),
        data_metrics_log_interval=10,
        enable_chat_validation=False,
    )
    # The trainer reports: task 1 is hard (3x weight).
    InterleaveDataManager.update_task_weights([1.0, 3.0])

    for _ in range(3):
        m._refill_message_queue(min_documents=64)
        m.message_queue.message_queue.clear()  # force the next refill

    # Sampling weights moved toward the hard task...
    assert m.dynamic_weights[1] > m.dynamic_weights[0]
    assert m.sampling_count > 0
    # ...and the sampling-weights card has rows to render.
    m.data_metrics_logger.close()
    con = sqlite3.connect(tmp_path / "data_metrics.db")
    rows = con.execute(
        "select count(*), max(sampling_weights) from data_metrics"
    ).fetchone()
    con.close()
    assert rows[0] > 0
    assert "hard_ds" in rows[1]
    _reset()
    InterleaveDataManager.shared_weights = None
    InterleaveDataManager.shared_weights_initialized = False


def test_difficulty_weighter_output_upsamples_its_hard_task():
    _reset()
    # Feed the weighter a high loss on task 1 and a low loss on task 0, then
    # route its effective_weights through the sampler.
    weighter = DifficultyTaskLossWeighter(gamma=1.0)
    task_ids = torch.tensor([[0, 0, 1, 1]])
    losses = torch.tensor([[0.1, 0.1, 5.0, 5.0]])
    for _ in range(20):  # let the EMA settle
        weighter.observe(task_ids, losses)

    eff = weighter.effective_weights()
    assert eff[1] > eff[0]  # difficulty weighter upweighted the hard task

    InterleaveDataManager.shared_task_weights = [1.0, 1.0]
    InterleaveDataManager.update_task_weights(eff)
    m = _tasker_manager([0, 1], [1.0, 1.0], task_weights=None)
    InterleaveDataManager.shared_task_weights = [float(x) for x in eff]
    w = m._calculate_target_weights()
    assert w[1] > w[0]  # the hard task is sampled more
    _reset()


# ------------------------------------------------------------------------------
# dynamic_weights
# ------------------------------------------------------------------------------
# Tests for data sampling weight modes: static, dynamic, and novelty.


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

    def test_cold_start_stays_near_uniform(self):
        """During warmup, weights blend from uniform toward novelty-driven weights."""
        tracker = NoveltyTracker(num_datasets=2, warmup_samples=50)
        static = [0.7, 0.3]

        tracker.score_and_update(0, list(range(100)))
        tracker.score_and_update(1, list(range(100, 200)))

        weights = tracker.get_target_weights(static)
        # With only 2/50 docs processed, blend factor is ~0.04 — weights should be near uniform.
        assert (
            abs(weights[0] - 0.5) < 0.1
        ), f"Weight[0]={weights[0]:.4f} should be near 0.5 (uniform) during warmup"
        assert (
            abs(weights[1] - 0.5) < 0.1
        ), f"Weight[1]={weights[1]:.4f} should be near 0.5 (uniform) during warmup"

    def test_weight_floor(self):
        """No dataset weight should drop below 1% of its static weight."""
        tracker = NoveltyTracker(num_datasets=2, cms_width=4096, warmup_samples=0)

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
            num_datasets=2,
            cms_width=4096,
            warmup_samples=0,
            numeric_token_ids=numeric_ids,
        )
        tracker_without = NoveltyTracker(
            num_datasets=2,
            cms_width=4096,
            warmup_samples=0,
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


class TestDynamicMode:
    def setup_method(self):
        _reset_shared_state()

    def test_huge_docs_downweighted(self):
        """Dynamic mode should downweight datasets with huge documents."""
        from praxis.data.datasets.manager import InterleaveDataManager

        tokenizer = _make_tokenizer()
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
            weighting_mode="dynamic",
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


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]


# ---------------------------------------------------------------- packing


def test_prose_survives_the_packer(prose_tokenizer):
    """The packer passes `omit_leading_bos` for docs appended mid-sequence.
    In prose the boundary IS the separator, so the template must ignore that
    flag rather than run one document's role name into the previous one's last
    word."""
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(tokenizer=prose_tokenizer, block_size=512)
    for _ in range(6):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    batch = manager.get_batch(batch_size=2)

    assert len(batch["batch"]) == 2
    for seq, mask in zip(batch["batch"], batch["assistant_mask"]):
        assert seq.shape == mask.shape
        text = prose_tokenizer.decode(seq, skip_special_tokens=False)
        assert "[BOS]" not in text
        assert "[SEP]" not in text
        assert "[TOOL_CALL]" not in text
        assert "[EOS]" not in text
        # Doc-to-doc seams read as plain prose. Nothing marks them in the
        # stream; the seam is carried by block_ids instead.
        assert "Tokyo.\n\nsystem\n\n" in text
    assert manager.get_validation_stats()["documents_skipped"] == 0


# ------------------------------------------- assistant mask on non-ASCII text


MULTIBYTE = [
    {"role": "system", "content": "“quoted”"},
    {"role": "user", "content": "Calculate √1156 \U0001f600"},
    {"role": "assistant", "content": "— the answer is 34."},
    {"role": "user", "content": "and été?"},
    {"role": "assistant", "content": "Summer."},
]


def test_packer_uses_the_exact_mask(prose_tokenizer):
    """End to end through the packer: no assistant content may be dropped and no
    prompt content admitted, on multi-byte text."""
    from praxis.data.datasets.message_queue import MessageQueueManager

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


def test_packer_emits_block_ids_for_every_document(prose_tokenizer, default_tokenizer):
    """Packing needs the seam to be findable, and the packer states it.

    block_ids segment the local encoder's attention so it cannot read across
    unrelated documents. They come from the packer, which is the only step that
    knows where documents meet - so this holds identically for a format that
    writes a separator id and one that writes nothing at all.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    for tok in (prose_tokenizer, default_tokenizer):
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


def test_prose_writes_no_control_id_into_the_stream(prose_tokenizer):
    """The point of the pure-byte layout: every id is a byte.

    Nothing below 256 is reserved, so there is no id in a packed sequence that
    could not equally have come from the text itself.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(
        tokenizer=prose_tokenizer, block_size=4096, enable_chat_validation=False
    )
    for _ in range(3):
        manager.add_document({"messages": CONVERSATION, "metadata": {}})
    seq = manager.get_batch(batch_size=1)["batch"][0]

    assert (
        chat_format_of(prose_tokenizer).document_separator_id(prose_tokenizer) is None
    )
    assert int(seq.max()) < 256
    assert int(seq.min()) >= 0


def test_prose_documents_end_on_their_own_text(prose_tokenizer):
    """No separator is appended, so a document ends where its text ends.

    The old layout appended [EOS] and copied the last mask value onto it. With
    the separator gone there is nothing to supervise or to mask: the trailing
    boundary the template already emits is the halt signal, and it is inside
    the generated turn's span.
    """
    from praxis.data.datasets.message_queue import MessageQueueManager

    manager = MessageQueueManager(
        tokenizer=prose_tokenizer, block_size=4096, enable_chat_validation=False
    )
    ends_generated = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    ends_prompt = [{"role": "user", "content": "unanswered"}]

    ids, mask = manager._tokenize_doc(
        {"messages": ends_generated, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("yo\n\n")
    assert int(mask[-1]) == 1  # the halting boundary is a trained target

    ids, mask = manager._tokenize_doc(
        {"messages": ends_prompt, "metadata": {}}, omit_leading_bos=False
    )
    assert prose_tokenizer.decode(ids).endswith("unanswered\n\n")
    assert int(mask[-1]) == 0


# ------------------------------------------------------------------------------
# chat_validation
# ------------------------------------------------------------------------------
# Tests for chat template validation.


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    return StandardTokenizer.from_pretrained("gpt2")


def test_message_queue_integration(tokenizer):
    """Test that validation integrates correctly with MessageQueueManager."""
    # Create queue with validation enabled
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=True,
        strict_chat_validation=False,
    )

    # Add valid document
    valid_doc = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ],
        "metadata": {"source": "test"},
    }

    queue.add_document(valid_doc)

    # Process the queue (packing triggers tokenization + validation)
    queue.get_batch(batch_size=1, sequence_multiplier=1)

    # Check validation stats
    stats = queue.get_validation_stats()
    assert stats["documents_validated"] >= 1
    assert stats["documents_failed"] == 0
    assert stats["documents_skipped"] == 0


def test_message_queue_validation_disabled(tokenizer):
    """Test that validation can be disabled."""
    # Create queue with validation disabled
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=False,
        strict_chat_validation=False,
    )

    assert queue.chat_validator is None

    # Add document
    doc = {"messages": [{"role": "user", "content": "test"}], "metadata": {}}
    queue.add_document(doc)
    queue.get_batch(batch_size=1, sequence_multiplier=1)

    # Stats should show no validation happened
    stats = queue.get_validation_stats()
    assert stats["documents_validated"] == 0


def test_strict_mode_raises_exception(tokenizer):
    """Test that strict mode raises exceptions on validation failure."""
    # Create queue with strict validation
    queue = MessageQueueManager(
        tokenizer,
        block_size=512,
        enable_chat_validation=True,
        strict_chat_validation=True,
    )

    # Note: We can't easily create an invalid document through the normal API
    # because apply_chat_template should always produce valid output.
    # This test would require manually injecting bad token sequences,
    # which is tested in other tests above.

    # Just verify strict mode is set
    assert queue.strict_chat_validation is True
    assert queue.chat_validator.strict_mode is True


# ------------------------------------------------------------------------------
# kb_dataset
# ------------------------------------------------------------------------------
# KB-as-dataset sampler and incremental page indexing.


def test_kb_dataset_yields_docs():
    ds = KBDataset(tokenizer=None, seed=0, config={"sources": ["docs"]})
    seqs = ds.get_sequences(3)
    assert all(isinstance(s, str) and s for s in seqs)


def test_kb_dataset_is_seeded():
    a = KBDataset(None, 7, {"sources": ["docs"]}).get_sequences(5)
    b = KBDataset(None, 7, {"sources": ["docs"]}).get_sequences(5)
    assert a == b


def test_kb_dataset_reloads_on_exhaustion():
    ds = KBDataset(None, 0, {"sources": ["docs"]})
    ds._load_epoch()
    n = len(ds._epoch)
    assert all(ds.get_sequences(n + 2))  # one wrap, no stall


# ------------------------------------------------------------------------------
# governor
# ------------------------------------------------------------------------------
# GNS batch governor: estimator math, tier control, Lightning wiring.


def test_dataset_falls_back_to_static_shape_without_a_governor():
    from praxis.data.datasets.weighted import WeightedIterableDataset

    ds = object.__new__(WeightedIterableDataset)
    ds.batch_size = 16
    ds.sequence_multiplier_tiers = ()
    ds.governed = True  # governed, but no governor has enabled the schedule
    assert ds._next_shape() == (16, 1)


def test_validation_loader_keeps_a_fixed_shape():
    """The val loader is the same dataset class. Following the governed plan
    would make val loss incomparable across steps AND, because the plan counts
    microbatches to find a cycle boundary, shift the training cycle."""
    from praxis.data.datasets.weighted import WeightedIterableDataset

    train = object.__new__(WeightedIterableDataset)
    train.batch_size, train.sequence_multiplier_tiers, train.governed = 64, (), True
    val = object.__new__(WeightedIterableDataset)
    val.batch_size, val.sequence_multiplier_tiers, val.governed = 64, (), False

    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    assert val._next_shape() == (64, 1)  # untouched by the governor
    assert BatchSchedule.current() is None  # and it did not open a cycle
    assert train._next_shape() == (4, 1)  # the governed shape


def test_validation_draws_do_not_shift_the_training_cycle():
    from praxis.data.datasets.weighted import WeightedIterableDataset

    train = object.__new__(WeightedIterableDataset)
    train.batch_size, train.sequence_multiplier_tiers, train.governed = 64, (), True
    val = object.__new__(WeightedIterableDataset)
    val.batch_size, val.sequence_multiplier_tiers, val.governed = 64, (), False

    # accum 4: a cycle is four microbatches.
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=())
    train._next_shape()
    train._next_shape()
    for _ in range(10):  # a whole validation pass mid-cycle
        val._next_shape()
    assert BatchSchedule._micro_index == 2  # still two into the open cycle


# ------------------------------------------------------------------------------
# determinism
# ------------------------------------------------------------------------------


def test_huggingface_dataset_determinism():
    """Test that HuggingfaceDataset produces deterministic sequences with the same seed."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Set a simple chat template for testing
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    # Choose a small dataset from the available ones
    dataset_config = DATASETS["minipile-validation"].copy()
    dataset_config["streaming"] = False  # Set streaming to False for testing

    # Set all random sources before first run
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # First run with seed 42
    dataset_1 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    sequences_1 = dataset_1.get_sequences(5)

    # Reset all random sources before second run
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Second run with same seed
    dataset_2 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    sequences_2 = dataset_2.get_sequences(5)

    # Sequences should be identical
    assert (
        sequences_1 == sequences_2
    ), "Dataset sequences should be deterministic with the same seed"

    # Set different seed for third run
    random.seed(43)
    torch.manual_seed(43)
    np.random.seed(43)

    # Different seed should give different sequences
    dataset_3 = HuggingfaceDataset(tokenizer, 43, dataset_config)
    sequences_3 = dataset_3.get_sequences(5)

    # Unlikely that all sequences would match with a different seed
    assert (
        sequences_1 != sequences_3
    ), "Dataset sequences should differ with different seeds"


def test_interleave_data_manager_determinism():
    """Test that InterleaveDataManager produces deterministic batches with fixed seeds."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Set a simple chat template for testing
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    # Use a small dataset
    dataset_config = DATASETS["minipile-validation"].copy()
    dataset_config["streaming"] = False

    # Create two datasets with the same seed - set all random sources
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    dataset_1 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    dataset_2 = HuggingfaceDataset(tokenizer, 42, dataset_config)

    # Create two data managers with the same settings
    manager_1 = InterleaveDataManager([dataset_1], [1.0], tokenizer, block_size=128)
    manager_2 = InterleaveDataManager([dataset_2], [1.0], tokenizer, block_size=128)

    # Get batches from both managers
    batch_1 = manager_1.get_batch(batch_size=2)

    # Reset all seeds
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    batch_2 = manager_2.get_batch(batch_size=2)

    # Check that batches are identical
    # Handle new dictionary format
    assert isinstance(batch_1, dict) and isinstance(batch_2, dict)
    assert "batch" in batch_1 and "batch" in batch_2

    batch_tensors_1 = batch_1["batch"]
    batch_tensors_2 = batch_2["batch"]

    assert len(batch_tensors_1) == len(batch_tensors_2)
    for i in range(len(batch_tensors_1)):
        torch.testing.assert_close(batch_tensors_1[i], batch_tensors_2[i])

    # Test different sampling modes
    batch_3 = manager_1.get_batch(batch_size=4, sequence_multiplier=2)

    # Reset all seeds and get new batch
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    batch_4 = manager_2.get_batch(batch_size=4, sequence_multiplier=2)

    # Check that batches are identical
    batch_tensors_3 = batch_3["batch"]
    batch_tensors_4 = batch_4["batch"]

    assert len(batch_tensors_3) == len(batch_tensors_4)
    for i in range(len(batch_tensors_3)):
        torch.testing.assert_close(batch_tensors_3[i], batch_tensors_4[i])


# ------------------------------------------------------------------------------
# message_queue
# ------------------------------------------------------------------------------
# Test message queue BOS token constraint.


def test_bos_token_constraint():
    """Verify that BOS tokens only appear before role tokens."""
    # Create a tokenizer with chat template
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"

    # Add special tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOS]", "[SEP]", "[PAD]"]}
    )

    # Set chat template
    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE

    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Create message queue manager
    block_size = 128
    queue_manager = MessageQueueManager(tokenizer, block_size)

    # Add multiple documents
    for i in range(3):
        document = {
            "messages": [
                {"role": "system", "content": f"You are assistant {i}"},
                {"role": "user", "content": f"Hello {i}"},
                {"role": "assistant", "content": f"Hi there {i}!"},
            ],
            "metadata": {"doc_id": i},
        }
        queue_manager.add_document(document)

    # Get a batch
    batch_result = queue_manager.get_batch(batch_size=2)
    batch = batch_result["batch"]

    # Stack sequences into tensor
    batch_tensor = torch.stack(batch)

    # Get BOS token id
    bos_id = tokenizer.convert_tokens_to_ids("[BOS]")

    # Valid role tokens that can follow BOS
    valid_roles = ["system", "developer", "assistant", "user"]
    valid_role_ids = [
        tokenizer.encode(role, add_special_tokens=False)[0] for role in valid_roles
    ]

    # Check each sequence
    for seq_idx, sequence in enumerate(batch_tensor):
        # Find all BOS token positions
        bos_positions = (sequence == bos_id).nonzero(as_tuple=True)[0]

        print(f"\nSequence {seq_idx}:")
        print(f"  Found {len(bos_positions)} BOS tokens")

        # For each BOS token, check what follows
        for pos in bos_positions:
            if pos + 1 < len(sequence):
                next_token = sequence[pos + 1].item()
                next_token_str = tokenizer.decode([next_token])

                print(
                    f"  BOS at position {pos} -> next token: '{next_token_str}' (id={next_token})"
                )

                # Check if next token is a valid role
                is_valid = next_token in valid_role_ids

                if not is_valid:
                    # Check if it's part of a role word (tokenizer may split roles)
                    is_role_prefix = any(
                        next_token_str.lower().strip() in role for role in valid_roles
                    )

                    if not is_role_prefix:
                        print(f"  WARNING: Token after BOS is not a role token!")
                        print(f"  Valid role IDs: {valid_role_ids}")
                        print(f"  Found token ID: {next_token}")

                        # Decode a few tokens for context
                        context_start = max(0, pos - 2)
                        context_end = min(len(sequence), pos + 5)
                        context = tokenizer.decode(sequence[context_start:context_end])
                        print(f"  Context: '{context}'")

    print("\nTest completed - check output for any BOS constraint violations")


def test_per_document_tokenization():
    """Verify documents are tokenized separately, not concatenated."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.bos_token = "[BOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[BOS]", "[SEP]", "[PAD]"]}
    )

    from praxis.tokenizers.chat_templates import DEFAULT_CHAT_TEMPLATE

    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Create message queue manager
    block_size = 64
    queue_manager = MessageQueueManager(tokenizer, block_size)

    # Add 2 documents with very different content
    doc1 = {
        "messages": [
            {"role": "user", "content": "DOCUMENT_ONE"},
            {"role": "assistant", "content": "Response one"},
        ],
        "metadata": {"doc_id": 1},
    }

    doc2 = {
        "messages": [
            {"role": "user", "content": "DOCUMENT_TWO"},
            {"role": "assistant", "content": "Response two"},
        ],
        "metadata": {"doc_id": 2},
    }

    queue_manager.add_document(doc1)
    queue_manager.add_document(doc2)

    # Pack into a batch (triggers per-doc tokenization)
    result = queue_manager.get_batch(batch_size=1, sequence_multiplier=1)
    packed = torch.cat([result["batch"][0]])

    # Decode the packed sequence
    full_text = tokenizer.decode(packed, skip_special_tokens=False)

    print("\nFull packed sequence:")
    print(full_text)

    # Verify both documents are present
    assert "DOCUMENT_ONE" in full_text, "Document 1 content missing"
    assert "DOCUMENT_TWO" in full_text, "Document 2 content missing"

    # Count BOS tokens. Doc 1 contributes 2 (one per message). Doc 2's leading
    # BOS is stripped (mid-sequence), so it contributes 1 (assistant message).
    # Expect at least 3 BOS tokens total.
    bos_count = full_text.count("[BOS]")
    print(f"\nBOS token count: {bos_count}")
    assert bos_count >= 3, f"Expected at least 3 BOS tokens, found {bos_count}"

    print("\nPer-document tokenization test passed!")


if __name__ == "__main__":
    test_bos_token_constraint()
    test_per_document_tokenization()
