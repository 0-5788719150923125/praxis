"""Comprehensive test suite for the Praxis API server."""

import json
import os
import sys
import time
import threading
from unittest.mock import Mock, MagicMock, patch
import pytest
import requests
from typing import Generator, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from praxis.api import APIServer, app


class MockGenerator:
    """Mock generator for testing."""

    def __init__(self):
        self.model = Mock()
        self.request_counter = 0

    def request_generation(self, prompt: str, kwargs: dict) -> str:
        """Mock generation request."""
        self.request_counter += 1
        return f"request_{self.request_counter}"

    def get_result(self, request_id: str) -> str:
        """Mock getting generation result."""
        if "request_" in request_id:
            return f"Generated response for {request_id}"
        return None


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Mock chat template application."""
        result = ""
        for msg in messages:
            result += f"{self.bos_token}{msg.get('role', 'user')}\n{msg.get('content', '')}{self.sep_token}\n"
        if add_generation_prompt:
            result += f"{self.bos_token}assistant\n"
        return result


@pytest.fixture
def mock_generator():
    """Provide a mock generator."""
    return MockGenerator()


@pytest.fixture
def mock_tokenizer():
    """Provide a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def api_server(mock_generator, mock_tokenizer):
    """Create and start an API server for testing."""
    # Create server with mock components
    server = APIServer(
        generator=mock_generator,
        host="localhost",
        port=9999,  # Use a high port to avoid conflicts
        tokenizer=mock_tokenizer,
        param_stats={"total": 1000000, "trainable": 900000},
        seed=42,
        truncated_hash="test12345",
        full_hash="test1234567890abcdef",
        dev_mode=True,
        launch_command="python test.py"
    )

    # Start server in a thread
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{server.port}/api/ping")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        pytest.fail("Server did not start within timeout")

    yield server

    # Stop server after test
    server.stop()


@pytest.fixture
def api_url(api_server):
    """Provide the API URL."""
    return f"http://localhost:{api_server.port}"


class TestCoreRoutes:
    """Test core API routes."""

    def test_ping_endpoint(self, api_url):
        """Test /api/ping endpoint."""
        response = requests.get(f"{api_url}/api/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Praxis API server is running" in data["message"]

    def test_ping_post(self, api_url):
        """Test /api/ping with POST method."""
        response = requests.post(f"{api_url}/api/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_ping_options(self, api_url):
        """Test /api/ping with OPTIONS method."""
        response = requests.options(f"{api_url}/api/ping")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_spec_endpoint(self, api_url):
        """Test /api/spec endpoint."""
        response = requests.get(f"{api_url}/api/spec")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "truncated_hash" in data
        assert data["truncated_hash"] == "test12345"
        assert "full_hash" in data
        assert data["full_hash"] == "test1234567890abcdef"
        assert "args" in data
        assert "param_stats" in data
        assert data["param_stats"]["total"] == 1000000
        assert "seed" in data
        assert data["seed"] == 42

    def test_home_page(self, api_url):
        """Test home page returns HTML."""
        response = requests.get(f"{api_url}/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("Content-Type", "")
        # Check for CSP header
        assert "Content-Security-Policy" in response.headers

        # Check that it's actually HTML content
        assert "<!DOCTYPE html>" in response.text
        assert "<title>Praxis</title>" in response.text


class TestGenerationRoutes:
    """Test generation API routes."""

    def test_input_generation(self, api_url):
        """Test /input endpoint for string-based generation."""
        payload = {
            "prompt": "Hello, world!",
            "max_new_tokens": 50,
            "temperature": 0.7
        }
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "Generated response" in data["response"]

    def test_input_missing_prompt(self, api_url):
        """Test /input endpoint with missing prompt."""
        payload = {"max_new_tokens": 50}
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "prompt" in data["error"].lower()

    def test_input_with_messages_error(self, api_url):
        """Test /input endpoint rejects messages."""
        payload = {
            "prompt": "test",
            "messages": [{"role": "user", "content": "test"}]
        }
        response = requests.post(f"{api_url}/input", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "/messages endpoint" in data["error"]

    def test_messages_generation(self, api_url):
        """Test /messages endpoint for chat-based generation."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_new_tokens": 50
        }
        response = requests.post(f"{api_url}/messages", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        # Response should contain generated text
        assert len(data["response"]) > 0

    def test_messages_missing_messages(self, api_url):
        """Test /messages endpoint with missing messages."""
        payload = {"max_new_tokens": 50}
        response = requests.post(f"{api_url}/messages", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "messages" in data["error"].lower()

    def test_generation_options(self, api_url):
        """Test generation endpoints with OPTIONS method."""
        response = requests.options(f"{api_url}/input")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

        response = requests.options(f"{api_url}/messages")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


class TestAgentsRoute:
    """Test agent discovery route."""

    @patch('subprocess.run')
    def test_agents_endpoint(self, mock_run, api_url):
        """Test /api/agents endpoint."""
        # Mock git commands
        mock_run.return_value = Mock(
            returncode=0,
            stdout="origin\thttps://github.com/test/repo.git\t(fetch)\n",
            stderr=""
        )

        response = requests.get(f"{api_url}/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)

        # Should at least have "self" agent
        agent_names = [agent["name"] for agent in data["agents"]]
        assert "self" in agent_names

    def test_agents_options(self, api_url):
        """Test /api/agents with OPTIONS method."""
        response = requests.options(f"{api_url}/api/agents")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers


class TestStaticRoutes:
    """Test static file serving routes."""

    def test_favicon(self, api_url):
        """Test favicon.ico endpoint."""
        response = requests.get(f"{api_url}/favicon.ico")
        # Should return 204 No Content if favicon doesn't exist
        assert response.status_code in [200, 204]

    def test_static_files(self, api_url):
        """Test static file serving."""
        # Try to get a non-existent static file
        response = requests.get(f"{api_url}/static/nonexistent.js")
        # Should return 404
        assert response.status_code == 404


class TestGitRoutes:
    """Test Git HTTP backend routes."""

    @patch('subprocess.run')
    def test_git_info_refs(self, mock_run, api_url):
        """Test git info/refs endpoint."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=b"abc123\tHEAD\n",
            stderr=b""
        )

        response = requests.get(f"{api_url}/praxis/info/refs?service=git-upload-pack")
        assert response.status_code == 200
        assert "application/x-git-upload-pack-advertisement" in response.headers.get("Content-Type", "")

    def test_git_invalid_service(self, api_url):
        """Test git endpoint with invalid service."""
        response = requests.get(f"{api_url}/praxis/info/refs?service=invalid")
        assert response.status_code == 400

    def test_git_write_denied(self, api_url):
        """Test git write access is denied."""
        response = requests.get(f"{api_url}/praxis/info/refs?service=git-receive-pack")
        assert response.status_code == 403


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_headers(self, api_url):
        """Test CORS headers are present."""
        response = requests.get(f"{api_url}/api/ping")
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_custom_middleware_registration(self):
        """Test custom middleware can be registered."""
        from praxis.api import register_request_middleware, register_response_header

        # Register a test header
        register_response_header("X-Test-Header", "test-value")

        # Register a test middleware
        def test_middleware(request, response=None):
            if response:
                response.headers["X-Middleware-Test"] = "passed"
            return None

        register_request_middleware(test_middleware)

        # Headers should be registered
        from praxis.api.middleware import get_response_headers
        headers = get_response_headers()
        assert ("X-Test-Header", "test-value") in headers


class TestServerManagement:
    """Test server lifecycle management."""

    def test_server_start_stop(self, mock_generator, mock_tokenizer):
        """Test server can be started and stopped."""
        server = APIServer(
            generator=mock_generator,
            host="localhost",
            port=9998,  # Different port from fixture
            tokenizer=mock_tokenizer
        )

        # Start server
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True
        server_thread.start()

        # Wait for server to be ready
        time.sleep(2)

        # Check server is running
        try:
            response = requests.get(f"http://localhost:9998/api/ping", timeout=1)
            assert response.status_code == 200
        except:
            pytest.fail("Server did not respond")

        # Stop server
        server.stop()
        time.sleep(1)

        # Server should eventually stop responding
        # (Note: may take time for thread to fully terminate)

    def test_port_in_use_handling(self, mock_generator, mock_tokenizer):
        """Test server finds next available port if specified port is in use."""
        # Start first server
        server1 = APIServer(
            generator=mock_generator,
            host="localhost",
            port=9997,
            tokenizer=mock_tokenizer
        )

        # Actually start the first server so it binds to the port
        server1_thread = threading.Thread(target=server1.start)
        server1_thread.daemon = True
        server1_thread.start()

        # Wait for first server to start
        time.sleep(2)

        # Start second server with same port - should auto-increment
        server2 = APIServer(
            generator=mock_generator,
            host="localhost",
            port=9997,  # Same port
            tokenizer=mock_tokenizer
        )

        # Second server should have different port
        assert server2.port != server1.port
        assert server2.port > server1.port  # Should increment to a higher port

        # Clean up
        server1.stop()
        server2.stop()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])