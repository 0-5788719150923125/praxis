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
from praxis.policies.engagement_channel import LIVE_ENGAGEMENT

from ..utils import generate_from_messages

print_bp = Blueprint("print", __name__)
api_logger = logging.getLogger("praxis.web")

# Single-slot pending question: {id, question, predicted_answer}. One model-led
# question is presented at a time; answering it (or asking again) replaces it.
_lock = threading.Lock()
_pending: dict = {}


def _tokens(text: str):
    """Word-token proxy for the live reward: lowercased alphanumeric words, so
    'Paris?' matches 'Paris' (reward fires on mention, not exact phrasing)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_generator():
    cfg = current_app.config
    if "api_server" in cfg:
        return cfg["api_server"].generator
    return cfg.get("generator")


@print_bp.route("/api/print/ask", methods=["POST"])
def print_ask():
    """Ask the model to lead with a question; stash its predicted answer and
    expose the question. Idempotent while one is already pending."""
    with _lock:
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
            max_new_tokens=64,
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
