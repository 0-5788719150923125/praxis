"""`/messages`: how a run's inference defaults and a client's overrides combine.

One precedence rule, checked from the outside: the run's values apply, and
anything the request sends replaces them key by key. The web app's Settings
form and its editable developer prompt are both just "the request sent it",
which is why there is nothing else to reconcile server-side.
"""

import json

import pytest

from praxis.tokenizers import create_tokenizer


class _StubGenerator:
    """Records the prompt and kwargs the route handed to the generator."""

    def __init__(self):
        self.prompt = None
        self.kwargs = None

    def request_generation(self, prompt, kwargs, deadline=None, **callbacks):
        self.prompt = prompt
        self.kwargs = dict(kwargs)
        return "request-id"

    def get_result(self, request_id):
        return f"{self.prompt}a reply"


@pytest.fixture(scope="module")
def tokenizer():
    """A real Praxis tokenizer, because the prompt rules are format-aware."""
    return create_tokenizer(vocab_size=4096, tokenizer_type="char_level")


@pytest.fixture
def client(tokenizer):
    """A throwaway app carrying the two blueprints under test.

    Deliberately NOT `praxis.web.app.app`: mutating that singleton's config, or
    registering blueprints on it a second time, leaks into every later test.
    """
    import os

    from flask import Flask

    import praxis.web
    from praxis.web.routes.core import core_bp
    from praxis.web.routes.generation import generation_bp

    templates = os.path.join(os.path.dirname(praxis.web.__file__), "templates")
    app = Flask(__name__, template_folder=templates)
    app.config["TESTING"] = True
    generator = _StubGenerator()
    app.config.update(
        generator=generator,
        tokenizer=tokenizer,
        system_prompt="You are terse.",
        developer_prompt="chat casually",
        generation_kwargs={"max_new_tokens": 64, "temperature": 0.9},
    )
    app.register_blueprint(generation_bp)
    app.register_blueprint(core_bp)

    test_client = app.test_client()
    test_client.generator = generator
    return test_client


def _post(client, **payload):
    payload.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post("/messages", json=payload)


def test_run_defaults_reach_the_generator(client):
    _post(client)
    assert client.generator.kwargs["max_new_tokens"] == 64
    assert client.generator.kwargs["temperature"] == 0.9


def test_run_prompts_reach_the_model(client):
    _post(client)
    assert "You are terse." in client.generator.prompt
    assert "chat casually" in client.generator.prompt


def test_request_kwargs_override_per_key(client):
    """The Settings form is an override, not a replacement: a key it does not
    mention keeps the run's value."""
    _post(client, generation_kwargs=["temperature=0.2", "top_p=0.8"])
    assert client.generator.kwargs["temperature"] == 0.2
    assert client.generator.kwargs["top_p"] == 0.8
    assert client.generator.kwargs["max_new_tokens"] == 64


def test_an_edited_developer_prompt_replaces_the_runs(client):
    _post(
        client,
        messages=[
            {"role": "developer", "content": "BE LOUD"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert "BE LOUD" in client.generator.prompt
    assert "chat casually" not in client.generator.prompt
    # The system prompt is a separate role and is untouched by that.
    assert "You are terse." in client.generator.prompt


def test_an_unknown_kwarg_is_rejected_not_dropped(client):
    response = _post(client, generation_kwargs=["temperture=0.2"])
    assert response.status_code == 400
    assert "temperture" in json.loads(response.data)["error"]


def test_legacy_top_level_fields_still_work(client):
    """Older clients sent these beside the messages rather than in a mapping."""
    _post(client, max_new_tokens=32, temperature=0.1)
    assert client.generator.kwargs["max_new_tokens"] == 32
    assert client.generator.kwargs["temperature"] == 0.1


def test_timeout_never_reaches_generate(client):
    """It bounds the decode through the request deadline; handing it to
    `generate` as a kwarg would be a parameter transformers does not have."""
    _post(client, generation_kwargs=["timeout=5"])
    assert "timeout" not in client.generator.kwargs


def test_the_page_serves_the_run_defaults(client):
    """The web app seeds its forms from these before the first paint."""
    import html as html_module
    import re

    page = client.get("/").data.decode()
    match = re.search(r'name="praxis-defaults" content="([^"]*)"', page)
    assert match, "the page must carry the run's inference defaults"
    defaults = json.loads(html_module.unescape(match.group(1)))
    assert defaults["developerPrompt"] == "chat casually"
    assert defaults["generationKwargs"]["max_new_tokens"] == 64
