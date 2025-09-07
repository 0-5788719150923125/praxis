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
            if self._loop and not self._loop.is_closed():
                # Schedule cleanup in the event loop
                future = asyncio.run_coroutine_threadsafe(
                    self._cleanup_async(), self._loop
                )
                future.result(timeout=5)  # Wait for cleanup to complete
                # Stop the event loop
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception as e:
            print(f"Error stopping ngrok: {e}")
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


# Global tunnel instance and secret
_tunnel = None
_ngrok_secret = None


def load_ngrok_secret():
    """Load ngrok secret from file."""
    global _ngrok_secret
    ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
    try:
        if os.path.exists(ngrok_info_path):
            with open(ngrok_info_path, "r") as f:
                lines = f.read().strip().split("\n")
                if len(lines) >= 2:
                    _ngrok_secret = lines[1]  # Second line is the secret
                    return True
    except Exception as e:
        pass
    return False


def setup_ngrok_routes(app):
    """Setup catch-all routes for ngrok secret-prefixed paths."""
    global _ngrok_secret
    
    # Load secret if not loaded
    if not _ngrok_secret:
        load_ngrok_secret()
    
    if not _ngrok_secret:
        return
    
    from flask import request as flask_request
    
    # Create a catch-all route for the secret prefix
    @app.route(
        f"/{_ngrok_secret}/",
        defaults={"path": ""},
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    @app.route(
        f"/{_ngrok_secret}/<path:path>",
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
                            ('src="/static/', f'src="/{_ngrok_secret}/static/'),
                            ('href="/static/', f'href="/{_ngrok_secret}/static/'),
                            ('url("/static/', f'url("/{_ngrok_secret}/static/'),
                            ("url('/static/", f"url('/{_ngrok_secret}/static/"),
                            ('src="/socket.io/', f'src="/{_ngrok_secret}/socket.io/'),
                            ("fetch('/", f"fetch('/{_ngrok_secret}/"),
                            ('fetch("/', f'fetch("/{_ngrok_secret}/'),
                            ('"/api/', f'"/{_ngrok_secret}/api/'),
                            ("'/api/", f"'/{_ngrok_secret}/api/"),
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

    def api_server_hook(self, host: str, port: int) -> None:
        """Hook called when API server starts.
        
        Args:
            host: The host the API server is running on
            port: The port the API server is running on
        """
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
        auth_token = getattr(args, "ngrok_auth_token", None) or os.getenv("NGROK_AUTHTOKEN")
        
        if not auth_token:
            print("❌ NGROK ERROR: No auth token provided. Set NGROK_AUTHTOKEN env var or use --ngrok-auth-token")
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
            
            # Save ngrok info to a file
            ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
            os.makedirs(os.path.dirname(ngrok_info_path), exist_ok=True)
            with open(ngrok_info_path, "w") as f:
                f.write(f"{base_url}\n{_tunnel.webhook_secret}")
            
            # Set up Flask routes - import the app from api module
            try:
                import api
                setup_ngrok_routes(api.app)
            except Exception as e:
                print(f"Could not set up routes: {e}")
        else:
            print("\n❌ NGROK ERROR: Failed to establish tunnel")
            import sys
            sys.exit(1)

    def request_middleware(self, request: Any, response: Any = None) -> Any:
        """Middleware for modifying requests/responses."""
        if response is not None:
            # This is the after_request phase - add ngrok headers
            response.headers["ngrok-skip-browser-warning"] = "true"
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
        global _ngrok_secret
        if not _ngrok_secret:
            load_ngrok_secret()
        
        path = request.path
        
        # Allow requests that start with the secret
        if _ngrok_secret and path.startswith(f"/{_ngrok_secret}"):
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
        
        # Remove ngrok info file
        try:
            ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
            if os.path.exists(ngrok_info_path):
                os.remove(ngrok_info_path)
        except:
            pass