"""The shared Flask app (praxis/web/app.py): CORS on every route."""

import pytest
import requests


@pytest.mark.parametrize("path", ["/api/ping", "/input", "/messages", "/api/agents"])
def test_options_preflight_carries_cors(api_url, path):
    response = requests.options(f"{api_url}{path}")
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_simple_requests_carry_cors(api_url):
    response = requests.get(f"{api_url}/api/ping")
    assert response.headers["Access-Control-Allow-Origin"] == "*"
