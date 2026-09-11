"""Integration middleware (praxis/web/middleware): hooks registered at runtime
must reach real responses - ngrok's browser-warning bypass rides on this."""

import requests

from praxis.web import middleware, register_request_middleware, register_response_header


def test_registered_hooks_reach_the_response(api_url, monkeypatch):
    # The registries are process-global lists; give this test its own copies.
    monkeypatch.setattr(middleware, "_response_headers", [])
    monkeypatch.setattr(middleware, "_request_middleware", [])

    seen = []
    register_response_header("X-Test-Header", "test-value")
    register_request_middleware(
        lambda request, response=None: seen.append(request.path)
    )

    response = requests.get(f"{api_url}/api/ping")

    assert response.headers["X-Test-Header"] == "test-value"
    assert "/api/ping" in seen
