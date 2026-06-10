"""`Print` mechanism: the model leads with a question; the user answers; the
response becomes a live engagement-prediction reward (PLAN.md P4/P5).

Conditional by design - a question is only "available" to the UI after the model
has actually been asked to lead with one (no user prompt). The frontend polls
``/api/print/ask`` (the environment-level hook), and the Print button stays hidden
until a question exists to present.
"""

import logging
import re
import threading
import uuid

from flask import Blueprint, current_app, jsonify, request

from praxis.data.config import SYSTEM_PROMPT, sample_developer_prompt
from praxis.policies.engagement_channel import LIVE_ENGAGEMENT, LIVE_JOKES
from praxis.policies.loop_modes import get_loop_mode

from ..utils import generate_from_messages

print_bp = Blueprint("print", __name__)
api_logger = logging.getLogger("praxis.web")

# Single-slot pending question: {id, question, predicted_answer}. One model-led
# question is presented at a time; answering it (or asking again) replaces it.
_lock = threading.Lock()
_pending: dict = {}

# Single-slot pending loop section: {id, text, predicted, mode}. Scoring it (or
# generating again) replaces it.
_loop_pending: dict = {}


def _tokens(text: str):
    """Word-token proxy for the live reward: lowercased alphanumeric words, so
    'Paris?' matches 'Paris' (reward fires on mention, not exact phrasing)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_generator():
    cfg = current_app.config
    if "api_server" in cfg:
        return cfg["api_server"].generator
    return cfg.get("generator")


def _max_new_tokens(data, default=256):
    """Per-request generation budget: the UI sends its Max Tokens setting;
    clamp to a sane band and fall back to a generous default."""
    try:
        return max(16, min(1024, int(data.get("max_new_tokens") or default)))
    except (TypeError, ValueError):
        return default


@print_bp.route("/api/print/ask", methods=["POST"])
def print_ask():
    """Ask the model to lead with a question; stash its predicted answer and
    expose the question. Idempotent while one is already pending, unless the
    caller sends ``{"reroll": true}`` to discard it and generate a fresh one."""
    data = request.get_json(silent=True) or {}
    reroll = bool(data.get("reroll"))
    with _lock:
        if reroll:
            _pending.clear()
        if _pending:
            return jsonify(
                {
                    "status": "ok",
                    "available": True,
                    "id": _pending["id"],
                    "question": _pending["question"],
                }
            )

    generator = _get_generator()
    tokenizer = current_app.config.get("tokenizer")
    if generator is None or tokenizer is None:
        return jsonify(
            {"status": "ok", "available": False, "reason": "generator not ready"}
        )

    # Model leads: system + developer only, then add_generation_prompt drives the
    # assistant turn. The print format is "question\nanswer", so the first line is
    # the question and the remainder is the model's predicted answer (A_hat).
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": sample_developer_prompt("engage_conversation"),
        },
    ]
    try:
        reply = generate_from_messages(
            messages=messages,
            generator=generator,
            tokenizer=tokenizer,
            max_new_tokens=_max_new_tokens(data),
            temperature=0.7,
            repetition_penalty=1.15,
            do_sample=True,
        )
    except Exception as e:
        api_logger.error(f"[print] generation failed: {e}")
        reply = None

    if not reply:
        return jsonify(
            {"status": "ok", "available": False, "reason": "empty generation"}
        )

    head, _, tail = reply.partition("\n")
    question = head.strip()
    predicted = tail.strip()
    # Keep it a short, single question: cut at the first '?' if present, and cap
    # length so an undertrained model's run-on output stays presentable.
    qmark = question.find("?")
    if qmark != -1:
        question = question[: qmark + 1]
    question = question[:200].strip()
    if not question:
        return jsonify({"status": "ok", "available": False, "reason": "no question"})

    rid = str(uuid.uuid4())
    with _lock:
        _pending.clear()
        _pending.update(
            {"id": rid, "question": question, "predicted_answer": predicted}
        )
    return jsonify({"status": "ok", "available": True, "id": rid, "question": question})


@print_bp.route("/api/print/pending", methods=["GET"])
def print_pending():
    """The currently presentable question, if any (no generation)."""
    with _lock:
        if _pending:
            return jsonify(
                {
                    "available": True,
                    "id": _pending["id"],
                    "question": _pending["question"],
                }
            )
    return jsonify({"available": False})


@print_bp.route("/api/print/respond", methods=["POST"])
def print_respond():
    """Score the user's response against the model's stashed predicted answer,
    fold it into the live engagement energy, and clear the slot."""
    data = request.get_json() or {}
    rid = data.get("id")
    response = (data.get("response") or "").strip()

    with _lock:
        pending = dict(_pending) if _pending and _pending.get("id") == rid else None

    if pending is None:
        return (
            jsonify({"status": "error", "error": "no matching pending question"}),
            409,
        )

    event = LIVE_ENGAGEMENT.submit(
        _tokens(pending["predicted_answer"]), _tokens(response)
    )
    with _lock:
        if _pending.get("id") == rid:
            _pending.clear()

    return jsonify(
        {"status": "ok", "predicted_answer": pending["predicted_answer"], **event}
    )


@print_bp.route("/api/print/energy", methods=["GET"])
def print_energy():
    """Live engagement energy snapshot (for a dashboard badge)."""
    return jsonify(LIVE_ENGAGEMENT.snapshot())


def _resolve_loop_mode():
    """The active loop query mode: explicit app config, else the RL policy's
    ``loop_mode`` attribute (found on the live model, like the drain callback
    does), else the registry default."""
    cfg = current_app.config
    name = cfg.get("loop_mode")
    if not name:
        model = getattr(_get_generator(), "model", None)
        model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        for policy in (
            getattr(model, "recall_policies", {}).values() if model is not None else []
        ):
            name = getattr(policy, "loop_mode", None)
            if name:
                break
    return get_loop_mode(name)


@print_bp.route("/api/loop/generate", methods=["POST"])
def loop_generate():
    """Run one looped task through the active loop mode: build the prompt,
    generate (short, time-capped), parse off any self-predicted score, and stash
    the section for scoring. Never 500s - baby models produce gibberish or
    nothing; an empty generation returns text "" for the UI to caption."""
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "joke").strip() or "joke"
    mode = _resolve_loop_mode()

    generator = _get_generator()
    tokenizer = current_app.config.get("tokenizer")
    if generator is None or tokenizer is None:
        return jsonify(
            {"status": "ok", "available": False, "reason": "generator not ready"}
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "developer",
            "content": sample_developer_prompt("engage_conversation"),
        },
        {"role": "user", "content": mode.task_prompt(task)},
    ]
    try:
        reply = generate_from_messages(
            messages=messages,
            generator=generator,
            tokenizer=tokenizer,
            max_new_tokens=_max_new_tokens(data),
            temperature=0.7,
            repetition_penalty=1.15,
            do_sample=True,
            # Still time-capped so the UI never hangs, but roomy enough for
            # the full token budget on a slow box.
            timeout=60.0,
        )
    except Exception as e:
        api_logger.error(f"[loop] generation failed: {e}")
        reply = None

    text, predicted = mode.parse(reply or "")
    rid = str(uuid.uuid4())
    with _lock:
        _loop_pending.clear()
        _loop_pending.update(
            {"id": rid, "text": text, "predicted": predicted, "mode": mode.name}
        )
    return jsonify(
        {
            "status": "ok",
            "available": True,
            "id": rid,
            "text": text,
            "predicted": predicted,
            "mode": mode.name,
        }
    )


@print_bp.route("/api/loop/approve", methods=["POST"])
def loop_approve():
    """Record a human score for a looped output - the live joke reward. `score`
    is a signed -1..1 want->need judgement (or `approve` bool shorthand). The
    active loop mode converts (score, stashed prediction) into the channel's
    (activation, reward): in calibration mode the correction magnitude is the
    signal (less correction = more energy); approval mode takes the score at
    face value. An `id` ties the score to a stashed /api/loop/generate section;
    without one (or unmatched) there is no prediction and scoring degrades to
    approval semantics."""
    data = request.get_json() or {}
    score = data.get("score")
    if score is None:
        score = 1.0 if data.get("approve") else -1.0
    try:
        score = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0

    # NOT cleared on scoring: the slider can be re-adjusted (each change is its
    # own event, as before); the slot is replaced by the next /api/loop/generate.
    rid = data.get("id")
    with _lock:
        pending = (
            dict(_loop_pending)
            if rid and _loop_pending and _loop_pending.get("id") == rid
            else None
        )

    mode = get_loop_mode(pending["mode"]) if pending else _resolve_loop_mode()
    predicted = pending.get("predicted") if pending else None
    result = mode.score(score, predicted)
    event = LIVE_JOKES.submit_scalar(
        result["activation"], reward=result["reward"], extra=result["extra"]
    )
    return jsonify({"status": "ok", "score": score, "mode": mode.name, **event})


@print_bp.route("/api/loop/energy", methods=["GET"])
def loop_energy():
    """Live joke-approval energy snapshot."""
    return jsonify(LIVE_JOKES.snapshot())
