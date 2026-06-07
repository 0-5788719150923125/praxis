"""Offline data fallback: error classification, no-retry, cache-mode flips."""

import pytest
import torch

from praxis.data.datasets.network_retry import (
    hf_offline,
    is_network_error,
    retry_on_network_error,
)


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
    """The automatic retry step: a network failure on the streaming load
    latches offline mode and reloads the same dataset non-streaming (cache)."""
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
    monkeypatch.setattr(hf, "hub_reachable", lambda: False)
    sampler = hf.HuggingfaceDataset(
        tokenizer=None, seed=0, config={"path": "fake/dataset"}
    )
    assert sampler.is_streaming is False
    assert nr.hf_offline()  # latched for the rest of boot
    assert seen[0]["streaming"] is True and seen[-1]["streaming"] is False


def test_dataset_specific_failure_skips_without_latching(monkeypatch):
    """When the hub is reachable, one bad dataset raises (skipped upstream)
    without dragging the whole process offline."""
    import praxis.data.datasets.huggingface as hf
    import praxis.data.datasets.network_retry as nr

    _reset_latch(monkeypatch)

    def fake_load(dataset_args):
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(hf, "load_dataset_smart", fake_load)
    monkeypatch.setattr(hf, "hub_reachable", lambda: True)
    with pytest.raises(RuntimeError):
        hf.HuggingfaceDataset(tokenizer=None, seed=0, config={"path": "fake/bad"})
    assert not nr.hf_offline()  # rest of the mixture stays online


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


def test_midrun_hub_outage_latches_offline_and_propagates(monkeypatch):
    """Unbounded retries must not hang training when the hub is down: latch
    offline and raise so get_document's cache fallback can run."""
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
    assert nr.hf_offline()


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
