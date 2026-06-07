"""Tests for the Print mechanism: model-led question -> user answer -> reward."""

import sys

# The web import chain lazily parses argv; keep it benign so importing the routes
# under pytest doesn't trip argparse on pytest's own flags.
sys.argv = ["praxis"]

import flask  # noqa: E402
import pytest  # noqa: E402

from praxis.policies.engagement_channel import LIVE_ENGAGEMENT  # noqa: E402
from praxis.web.routes import print as print_route  # noqa: E402
from praxis.web.routes import register_routes  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_pending():
    """The pending slots are process-global; clear them between tests."""
    with print_route._lock:
        print_route._pending.clear()
        print_route._loop_pending.clear()
    yield


class _FakeGen:
    """Returns a model-led 'question\\nanswer' in the chat-template envelope."""

    def __init__(self, reply):
        self._reply = reply

    def request_generation(self, prompt, kwargs):
        return "rid"

    def get_result(self, rid):
        return self._reply


class _FakeTok:
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "[BOS]system\n..."


def _client(reply="[BOS]assistant\nWhat is the capital of France?\nParis[SEP]"):
    app = flask.Flask(__name__)
    app.config.update(generator=_FakeGen(reply), tokenizer=_FakeTok())
    register_routes(app)
    return app.test_client()


def test_button_is_conditional_until_asked():
    c = _client()
    assert c.get("/api/print/pending").get_json() == {"available": False}
    ask = c.post("/api/print/ask", json={}).get_json()
    assert ask["available"] is True
    assert ask["question"] == "What is the capital of France?"
    assert c.get("/api/print/pending").get_json()["available"] is True


def test_ask_is_idempotent_while_pending():
    c = _client()
    a1 = c.post("/api/print/ask", json={}).get_json()
    a2 = c.post("/api/print/ask", json={}).get_json()
    assert a1["id"] == a2["id"]


def test_respond_scores_and_clears():
    c = _client()
    ask = c.post("/api/print/ask", json={}).get_json()
    r = c.post(
        "/api/print/respond", json={"id": ask["id"], "response": "Is it Paris?"}
    ).get_json()
    assert r["status"] == "ok"
    assert r["activation"] == 1.0  # 'Paris?' matches predicted 'Paris'
    assert r["recall"] == 1.0
    assert r["predicted_answer"] == "Paris"
    # Slot cleared after answering.
    assert c.get("/api/print/pending").get_json() == {"available": False}


def test_stale_id_is_rejected():
    c = _client()
    c.post("/api/print/ask", json={})
    resp = c.post("/api/print/respond", json={"id": "nope", "response": "x"})
    assert resp.status_code == 409


def test_unavailable_when_generator_missing():
    app = flask.Flask(__name__)
    app.config["tokenizer"] = _FakeTok()  # no generator
    register_routes(app)
    out = app.test_client().post("/api/print/ask", json={}).get_json()
    assert out["available"] is False


def test_channel_buffers_for_trainer_drain():
    before = LIVE_ENGAGEMENT.snapshot()["count"]
    LIVE_ENGAGEMENT.submit(["paris"], ["paris", "i", "think"])
    assert LIVE_ENGAGEMENT.snapshot()["count"] == before + 1
    drained = LIVE_ENGAGEMENT.drain()
    assert drained and drained[-1]["recall"] == 1.0
    assert LIVE_ENGAGEMENT.snapshot()["buffered"] == 0


def test_loop_approve_records_joke_reward():
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = _client()
    approve = c.post("/api/loop/approve", json={"score": 1.0}).get_json()
    assert approve["status"] == "ok"
    assert approve["activation"] == 1.0
    # A rejection still sustains energy (engagement alone counts), but its
    # valence lives in the signed reward, not the activation.
    reject = c.post("/api/loop/approve", json={"approve": False}).get_json()
    assert reject["activation"] == pytest.approx(0.8)
    assert reject["score"] == -1.0 and reject["reward"] == -1.0
    assert c.get("/api/loop/energy").get_json()["count"] >= 2
    # Both events buffered for the joke drain callback.
    assert len(LIVE_JOKES.drain()) >= 2


def test_format_joke_quality_filters():
    from praxis.data.formatters import format_joke

    keys = ["jokeText", "rating"]
    good = format_joke({"jokeText": "knock knock", "rating": 5.0}, keys, None)
    assert [m["role"] for m in good["messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
    ]
    # The assistant turn ends with the normalized want->need score on its own
    # line - the dense grounding for the calibration loop mode.
    assert good["messages"][-1]["content"] == "knock knock\n+0.5"
    # Below-median (disliked) jokes are skipped (empty -> sampler retries).
    assert (
        format_joke({"jokeText": "bad", "rating": -2.0}, keys, None)["messages"] == []
    )


class TestLoopModes:
    """The loop-mode registry: parse + score contracts."""

    def test_registry_and_default(self):
        from praxis.policies.loop_modes import LOOP_MODE_REGISTRY, get_loop_mode

        assert set(LOOP_MODE_REGISTRY) == {"calibration", "approval"}
        assert get_loop_mode().name == "calibration"
        assert get_loop_mode("approval").name == "approval"
        assert get_loop_mode("nonsense").name == "calibration"  # safe fallback

    def test_calibration_parse(self):
        from praxis.policies.loop_modes import get_loop_mode

        m = get_loop_mode("calibration")
        assert m.parse("A pun!\n+0.6") == ("A pun!", 0.6)
        # No trailing score line -> the whole reply is content.
        assert m.parse("just a joke") == ("just a joke", None)
        # A bare number far outside the slider range is content (e.g. a year),
        # not a wildly-clamped prediction.
        assert m.parse("In the year\n1942") == ("In the year\n1942", None)

    def test_calibration_score_rewards_small_corrections(self):
        from praxis.policies.loop_modes import get_loop_mode

        m = get_loop_mode("calibration")
        perfect = m.score(0.6, 0.6)
        wrong = m.score(-0.4, 0.6)
        assert perfect["activation"] == 1.0
        assert perfect["extra"]["correction"] == 0.0
        assert wrong["activation"] == 0.5  # 1 - |corr|/2 = 1 - 0.5
        assert wrong["reward"] == -0.4  # valence stays the user's signed score
        # No parseable prediction -> degrade to approval semantics.
        assert m.score(1.0, None)["activation"] == 1.0
        assert m.score(1.0, None)["extra"] == {}


def test_loop_generate_then_calibrated_approve():
    from praxis.policies.engagement_channel import LIVE_JOKES

    LIVE_JOKES.drain()
    c = _client(reply="[BOS]assistant\nA pun!\n+0.6[SEP]")
    gen = c.post("/api/loop/generate", json={"task": "joke"}).get_json()
    assert gen["available"] is True
    assert gen["mode"] == "calibration"
    assert gen["text"] == "A pun!"  # prediction parsed off the display text
    assert gen["predicted"] == pytest.approx(0.6)

    # Confirming the model's guess = zero correction = full activation.
    ok = c.post(
        "/api/loop/approve", json={"id": gen["id"], "score": 0.6}
    ).get_json()
    assert ok["correction"] == pytest.approx(0.0)
    assert ok["activation"] == 1.0

    # A large correction shrinks the activation; valence keeps the user's sign.
    bad = c.post(
        "/api/loop/approve", json={"id": gen["id"], "score": -0.4}
    ).get_json()
    assert bad["correction"] == pytest.approx(1.0)
    assert bad["reward"] == pytest.approx(-0.4)
    assert bad["activation"] < ok["activation"]
    assert len(LIVE_JOKES.drain()) == 2
