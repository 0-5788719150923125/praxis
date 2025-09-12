"""Ngrok integration class for Praxis."""

import asyncio
import os
import threading
from typing import Any, Dict, Optional

from praxis.integrations.base import BaseIntegration, IntegrationSpec


class NgrokTunnel:
    """Manages ngrok tunnel for the API server using the Python SDK."""

    def __init__(self, host="localhost", port=5000, auth_token=None):
        self.host = host
        self.port = port
        self.auth_token = auth_token or os.getenv("NGROK_AUTHTOKEN")
        self.listener = None
        self.session = None
        self.public_url = None
        self.webhook_secret = None
        self.error = None
        self._loop = None
        self._thread = None

    def start(self):
        """Start the ngrok tunnel using the Python SDK."""
        try:
            import ngrok
        except ImportError:
            return False

        try:
            # Start ngrok in a separate thread with its own event loop
            self._thread = threading.Thread(target=self._run_ngrok_async, daemon=True)
            self._thread.start()

            # Wait for tunnel to be established
            import time

            max_wait = 10  # seconds
            wait_time = 0
            while not self.public_url and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5

            if self.public_url is not None:
                return True
            else:
                if self.error:
                    print(f"❌ NGROK ERROR: {self.error}")
                return False

        except Exception as e:
            print(f"❌ NGROK ERROR: {e}")
            return False

    def _run_ngrok_async(self):
        """Run ngrok in async event loop."""
        import ngrok

        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # Run the async tunnel setup
            self._loop.run_until_complete(self._setup_tunnel())
        except Exception as e:
            # Store error for main thread to see
            self.error = str(e)
            print(f"❌ NGROK ERROR: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _setup_tunnel(self):
        """Set up the ngrok tunnel asynchronously."""
        import secrets
        import string

        import ngrok

        # Generate webhook secret (deterministic if seed provided)
        secret_seed = os.getenv("NGROK_SEED")
        if secret_seed:
            # Use deterministic generation with seed
            import base64
            import hashlib

            # Create a hash from the seed and take first 32 chars (base64 safe)
            hash_bytes = hashlib.sha256(secret_seed.encode()).digest()
            self.webhook_secret = base64.urlsafe_b64encode(hash_bytes)[:32].decode()
        else:
            # Use secure random generation
            alphabet = string.ascii_letters + string.digits
            self.webhook_secret = "".join(secrets.choice(alphabet) for _ in range(32))

        # Create session with auth token
        self.session = await ngrok.SessionBuilder().authtoken(self.auth_token).connect()

        # Try to use a hardcoded static domain (you can set this in .env)
        static_domain = os.getenv("NGROK_DOMAIN")

        # Build the listener WITHOUT traffic policy - Flask will handle auth
        listener_builder = self.session.http_endpoint()
        if static_domain:
            listener_builder = listener_builder.domain(static_domain)

        self.listener = await listener_builder.listen()
        self.public_url = self.listener.url()

        # Forward traffic to local server
        await self.listener.forward(f"http://{self.host}:{self.port}")

    def stop(self):
        """Stop the ngrok tunnel."""
        try:
            if self._loop:
                # Check if loop is actually usable
                try:
                    is_closed = self._loop.is_closed()
                except (RuntimeError, AttributeError):
                    # Loop is in an invalid state
                    is_closed = True

                if not is_closed:
                    try:
                        # Schedule cleanup in the event loop with very short timeout
                        future = asyncio.run_coroutine_threadsafe(
                            self._cleanup_async(), self._loop
                        )
                        future.result(timeout=0.5)  # Very short timeout during shutdown
                    except (asyncio.TimeoutError, RuntimeError, AttributeError):
                        # These are all expected during shutdown
                        pass
                    except Exception as e:
                        # Only log unexpected errors
                        if "Event loop is closed" not in str(e):
                            pass  # Silently ignore

                    try:
                        # Try to stop the event loop if it's still running
                        if (
                            hasattr(self._loop, "is_running")
                            and self._loop.is_running()
                        ):
                            self._loop.call_soon_threadsafe(self._loop.stop)
                    except (RuntimeError, AttributeError):
                        # Loop might be closed or stopped already
                        pass
        except Exception as e:
            # Completely silence "Event loop is closed" errors
            if "Event loop is closed" not in str(e):
                # Only print real unexpected errors
                pass
        finally:
            if self.public_url:
                print(f"🔒 Ngrok tunnel closed")
                self.public_url = None

    async def _cleanup_async(self):
        """Clean up ngrok resources asynchronously."""
        if self.listener:
            await self.listener.close()
        if self.session:
            await self.session.close()


# Global tunnel instance
_tunnel = None


def setup_ngrok_routes(app, ngrok_secret):
    """Setup catch-all routes for ngrok secret-prefixed paths.

    Args:
        app: Flask application instance
        ngrok_secret: The secret to use for the URL prefix
    """
    from flask import request as flask_request

    # Create a catch-all route for the secret prefix
    @app.route(
        f"/{ngrok_secret}/",
        defaults={"path": ""},
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    @app.route(
        f"/{ngrok_secret}/<path:path>",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    def ngrok_proxy(path):
        """Proxy requests from ngrok secret URLs to the real endpoints."""
        # Build the real path
        real_path = "/" + path if path else "/"

        # Get the view function for the real path
        try:
            # Build the full URL with query string
            if flask_request.query_string:
                real_url = f"{real_path}?{flask_request.query_string.decode()}"
            else:
                real_url = real_path

            # Use Flask's test client to make internal request
            with app.test_client() as client:
                # Copy headers but remove ngrok-identifying ones
                clean_headers = {}
                for key, value in flask_request.headers:
                    if key.lower() not in [
                        "x-forwarded-host",
                        "ngrok-skip-browser-warning",
                        "host",
                    ]:
                        clean_headers[key] = value

                # Copy the request method and data
                response = client.open(
                    real_url,
                    method=flask_request.method,
                    data=flask_request.get_data(),
                    headers=clean_headers,
                    follow_redirects=False,
                )

                # Get response data
                data = response.get_data()

                # If it's HTML or JavaScript, rewrite URLs to include the secret prefix
                content_type = response.headers.get("Content-Type", "")
                if (
                    "text/html" in content_type
                    or "application/javascript" in content_type
                    or "text/javascript" in content_type
                ) and isinstance(data, bytes):
                    try:
                        html = data.decode("utf-8")
                        replacements = [
                            ('src="/static/', f'src="/{ngrok_secret}/static/'),
                            ('href="/static/', f'href="/{ngrok_secret}/static/'),
                            ('url("/static/', f'url("/{ngrok_secret}/static/'),
                            ("url('/static/", f"url('/{ngrok_secret}/static/"),
                            ('src="/socket.io/', f'src="/{ngrok_secret}/socket.io/'),
                            ("fetch('/", f"fetch('/{ngrok_secret}/"),
                            ('fetch("/', f'fetch("/{ngrok_secret}/'),
                            ('"/api/', f'"/{ngrok_secret}/api/'),
                            ("'/api/", f"'/{ngrok_secret}/api/"),
                        ]
                        for old, new in replacements:
                            html = html.replace(old, new)
                        data = html.encode("utf-8")
                    except:
                        pass

                # Return the response
                return data, response.status_code, response.headers

        except Exception as e:
            from flask import abort

            return abort(404)


class Integration(BaseIntegration):
    """Ngrok tunnel integration for exposing the API server."""

    def __init__(self, spec: IntegrationSpec):
        """Initialize the ngrok integration."""
        super().__init__(spec)
        self.tunnel = None

    def add_cli_args(self, parser) -> None:
        """Add ngrok CLI arguments to the parser."""
        networking_group = None

        # Find the 'networking' argument group
        for group in parser._action_groups:
            if group.title == "networking":
                networking_group = group
                break

        if networking_group is None:
            networking_group = parser.add_argument_group("networking")

        networking_group.add_argument(
            "--ngrok",
            action="store_true",
            default=False,
            help="Expose the API server via ngrok tunnel",
        )
        networking_group.add_argument(
            "--ngrok-auth-token",
            type=str,
            default=None,
            help="Ngrok auth token (can also be set via NGROK_AUTHTOKEN env var)",
        )

    def on_api_server_start(self, app: Any, args: Any) -> None:
        """Hook called when API server starts.

        Args:
            app: Flask application instance
            args: Command-line arguments
        """
        # Extract host and port from the app or args as needed
        host = getattr(args, "host_name", "localhost")
        port = getattr(args, "port", 2100)
        global _tunnel

        # Check if ngrok is actually enabled
        try:
            from cli import get_cli_args

            args = get_cli_args()
            if not getattr(args, "ngrok", False):
                return
        except:
            return

        if _tunnel is not None:
            print("⚠️  Ngrok tunnel already running")
            return

        # Load from .env file first
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        # Get auth token from args or environment
        auth_token = getattr(args, "ngrok_auth_token", None) or os.getenv(
            "NGROK_AUTHTOKEN"
        )

        if not auth_token:
            print(
                "❌ NGROK ERROR: No auth token provided. Set NGROK_AUTHTOKEN env var or use --ngrok-auth-token"
            )
            return

        print(f"🚀 Starting ngrok tunnel for {host}:{port}")
        _tunnel = NgrokTunnel(host, port, auth_token)
        success = _tunnel.start()

        if success:
            base_url = _tunnel.public_url
            protected_url = f"{base_url}/{_tunnel.webhook_secret}"

            print(f"🌐 Ngrok tunnel active: {base_url}")
            print(f"🔐 Protected URL: {protected_url}")
            print(f"📡 Local server: http://{host}:{port}")

            # Store ngrok info in Flask app config for middleware to access
            try:
                import api

                # Store the ngrok information in app config
                api.app.config["ngrok_url"] = base_url
                api.app.config["ngrok_secret"] = _tunnel.webhook_secret
                api.app.config["ngrok_protected_url"] = protected_url

                # Register the ngrok header using the generic hook
                api.register_response_header("ngrok-skip-browser-warning", "true")

                # Set up Flask routes with the secret
                setup_ngrok_routes(api.app, _tunnel.webhook_secret)

            except Exception as e:
                print(f"Could not set up routes: {e}")
        else:
            print("\n❌ NGROK ERROR: Failed to establish tunnel")
            import sys

            sys.exit(1)

    def request_middleware(self, request: Any, response: Any = None) -> Any:
        """Middleware for modifying requests/responses."""
        if response is not None:
            # This is the after_request phase - headers are now added via register_response_header
            return None

        # This is the before_request phase - check authorization for ngrok requests
        # Check if this request is coming through ngrok
        is_ngrok_request = (
            "ngrok-skip-browser-warning" in request.headers
            or "X-Forwarded-Host" in request.headers
            or request.host.endswith(".ngrok-free.app")
            or request.host.endswith(".ngrok.io")
            or request.host.endswith(".src.eco")
        )

        if not is_ngrok_request:
            return None  # Allow all local requests

        # For ngrok requests, check if they have the secret prefix
        # Get the secret from Flask app config
        try:
            from flask import current_app

            ngrok_secret = current_app.config.get("ngrok_secret")
        except:
            ngrok_secret = None

        if not ngrok_secret:
            # No secret configured, block the request
            from flask import abort

            return abort(404)

        path = request.path

        # Allow requests that start with the secret
        if path.startswith(f"/{ngrok_secret}"):
            return None  # Authorized request

        # Block unauthorized ngrok requests
        from flask import abort

        return abort(404)

    def cleanup(self) -> None:
        """Clean up ngrok resources."""
        global _tunnel

        if _tunnel is not None:
            _tunnel.stop()
            _tunnel = None
