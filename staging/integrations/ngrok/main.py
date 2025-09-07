"""Ngrok integration implementation for Praxis."""

import asyncio
import os
import threading
from typing import Optional


def add_cli_args(parser):
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
            # Using deterministic secret from seed
        else:
            # Use secure random generation
            alphabet = string.ascii_letters + string.digits
            self.webhook_secret = "".join(secrets.choice(alphabet) for _ in range(32))
            # Using random secret

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
                future.result(timeout=5)  # Wait up to 5 seconds for cleanup
        except Exception:
            pass  # Silent cleanup
        finally:
            if self.public_url:
                print(f"🔒 Ngrok tunnel closed")
                self.public_url = None

    async def _cleanup_async(self):
        """Clean up ngrok resources asynchronously."""
        try:
            if self.listener:
                await self.listener.close()
                self.listener = None

            if self.session:
                await self.session.close()
                self.session = None
        except Exception:
            pass  # Silent cleanup

    def get_public_url(self):
        """Get the public URL of the tunnel."""
        return self.public_url


# Global tunnel instance
_tunnel = None


def api_server_hook(host, port):
    """Hook function called when API server starts."""
    global _tunnel

    # Check if ngrok is actually enabled
    try:
        from cli import get_cli_args

        args = get_cli_args()
        if not getattr(args, "ngrok", False):
            # Ngrok not enabled, don't start tunnel
            return
    except:
        # If we can't get args, don't start tunnel
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

    print(f"🚀 Starting ngrok tunnel for {host}:{port}")
    _tunnel = NgrokTunnel(host, port, auth_token)
    success = _tunnel.start()

    if success:
        base_url = _tunnel.get_public_url()
        protected_url = f"{base_url}/{_tunnel.webhook_secret}"

        print(f"🌐 Ngrok tunnel active: {base_url}")
        print(f"🔐 Protected URL: {protected_url}")
        print(f"📡 Local server: http://{host}:{port}")
        print(f"ℹ️  Flask will handle authentication")
        print(f"     All requests to /{_tunnel.webhook_secret}/* are authorized")
        print(f"     All other requests return 404 (silent drop)")
        print(f"✅ No ngrok policy quota usage!")

        # Save ngrok info to a file for other processes to read
        ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
        os.makedirs(os.path.dirname(ngrok_info_path), exist_ok=True)
        with open(ngrok_info_path, "w") as f:
            f.write(f"{base_url}\n{_tunnel.webhook_secret}")

        # Now that we have the secret, set up Flask routes
        # We need to get the Flask app instance
        try:
            # Import the api module to get the app
            import api

            setup_ngrok_routes(api.app)

            # Patch Socket.IO to handle prefixed paths
            if hasattr(api, "socketio"):
                socketio = api.socketio
                try:
                    # Get the ngrok secret for patching
                    ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
                    if os.path.exists(ngrok_info_path):
                        with open(ngrok_info_path, "r") as f:
                            lines = f.read().strip().split("\n")
                            if len(lines) >= 2:
                                secret = lines[1]

                                # Patch the underlying engineio server to handle prefixed paths
                                if hasattr(socketio, "server") and hasattr(
                                    socketio.server, "eio"
                                ):
                                    eio = socketio.server.eio
                                    original_handle_request = eio.handle_request

                                    def patched_handle_request(environ, start_response):
                                        path = environ.get("PATH_INFO", "")
                                        # Strip the ngrok secret prefix from Socket.IO paths
                                        if path.startswith(f"/{secret}/socket.io"):
                                            environ["PATH_INFO"] = path[
                                                len(f"/{secret}") :
                                            ]
                                            environ["SCRIPT_NAME"] = f"/{secret}"
                                            pass  # Stripped Socket.IO prefix
                                        return original_handle_request(
                                            environ, start_response
                                        )

                                    eio.handle_request = patched_handle_request
                                    pass  # Patched Socket.IO
                except Exception as e:
                    pass  # Could not patch Socket.IO
        except Exception as e:
            pass  # Could not set up routes
    else:
        print("\n❌ NGROK ERROR: Failed to establish tunnel")
        import sys

        sys.exit(1)


def initialize(args, cache_dir, ckpt_path=None, truncated_hash=None):
    """Initialize ngrok module when conditions are met."""
    # Check SDK availability immediately at init
    try:
        import ngrok
    except ImportError:
        print("\n❌ NGROK ERROR: Python SDK not installed")
        print("   The module loader should have installed it automatically.")
        print("   Try: pip install ngrok")
        import sys

        sys.exit(1)

    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()  # Load .env from current directory
    except ImportError:
        pass  # python-dotenv not available, skip

    # Check for auth token in priority order:
    # 1. CLI argument --ngrok-auth-token
    # 2. Environment variable NGROK_AUTHTOKEN (including from .env)
    auth_token = getattr(args, "ngrok_auth_token", None) or os.getenv("NGROK_AUTHTOKEN")

    if not auth_token:
        print("\n❌ NGROK ERROR: Auth token required")
        print("   Choose one of these options:")
        print("   1. Create .env file with: NGROK_AUTHTOKEN=your_token_here")
        print("   2. Set environment: export NGROK_AUTHTOKEN=your_token_here")
        print("   3. Use CLI flag: --ngrok-auth-token your_token_here")
        print("\n   Get your token:")
        print("   • Sign up: https://dashboard.ngrok.com/signup")
        print("   • Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
        import sys

        sys.exit(1)

    # Register WSGI middleware early (before API server is created)
    # This ensures it's applied before Socket.IO is initialized
    try:
        import api

        def create_ngrok_wsgi_middleware_early(original_wsgi):
            """Create WSGI middleware to strip ngrok prefix from Socket.IO paths"""

            def ngrok_wsgi_wrapper(environ, start_response):
                # Check if we have ngrok info
                ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
                if os.path.exists(ngrok_info_path):
                    try:
                        with open(ngrok_info_path, "r") as f:
                            lines = f.read().strip().split("\n")
                            if len(lines) >= 2:
                                secret = lines[1]
                                path = environ.get("PATH_INFO", "")
                                if path.startswith(f"/{secret}/socket.io"):
                                    environ["PATH_INFO"] = path[len(f"/{secret}") :]
                                    pass  # Stripped WSGI prefix
                    except:
                        pass
                return original_wsgi(environ, start_response)

            return ngrok_wsgi_wrapper

        api.register_wsgi_middleware(create_ngrok_wsgi_middleware_early)
        pass  # Registered early WSGI middleware
    except:
        pass  # API module not yet available

    print("🔐 Will auto-generate secret with complete URL protection")
    print(f"ℹ️  URLs will use format: /SECRET instead of /webhook/SECRET")
    print("✅ Ngrok ready")
    return {}


# Ngrok authentication state
_ngrok_secret = None
_last_check_time = 0


def load_ngrok_secret():
    """Load the ngrok secret from the info file"""
    global _ngrok_secret
    try:
        ngrok_info_path = os.path.join("build", "praxis", "NGROK_INFO.txt")
        if os.path.exists(ngrok_info_path):
            with open(ngrok_info_path, "r") as f:
                lines = f.read().strip().split("\n")
                if len(lines) >= 2:
                    _ngrok_secret = lines[1]  # Second line is the secret
                    # Loaded ngrok secret
                    return True
    except Exception as e:
        pass  # Error loading secret
    return False


def setup_ngrok_routes(app):
    """Setup catch-all routes for ngrok secret-prefixed paths"""
    global _ngrok_secret

    # Load secret if not loaded
    if not _ngrok_secret:
        load_ngrok_secret()

    if not _ngrok_secret:
        return  # No secret loaded

    from flask import redirect, request as flask_request
    import werkzeug

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
        """Proxy requests from ngrok secret URLs to the real endpoints"""
        # Build the real path
        real_path = "/" + path if path else "/"

        pass  # Proxying request

        # Get the view function for the real path
        try:
            # Build the full URL with query string
            if flask_request.query_string:
                real_url = real_path + "?" + flask_request.query_string.decode()
            else:
                real_url = real_path

            # Use Flask's test client to internally call the real endpoint
            with app.test_client() as client:
                # Copy headers but remove ngrok-identifying ones to avoid recursive auth checks
                clean_headers = {}
                for key, value in flask_request.headers:
                    # Skip headers that would identify this as an ngrok request
                    if key.lower() not in [
                        "x-forwarded-host",
                        "ngrok-skip-browser-warning",
                        "host",
                    ]:
                        clean_headers[key] = value

                # Copy the original request method and data
                response = client.open(
                    real_url,
                    method=flask_request.method,
                    data=flask_request.get_data(),
                    headers=clean_headers,
                    follow_redirects=False,
                )

                # Get response data
                data = response.get_data()

                # If it's HTML or JavaScript content, rewrite URLs to include the secret prefix
                content_type = response.headers.get("Content-Type", "")
                if (
                    "text/html" in content_type
                    or "application/javascript" in content_type
                    or "text/javascript" in content_type
                ) and isinstance(data, bytes):
                    try:
                        html = data.decode("utf-8")
                        # Rewrite absolute URLs to include the secret prefix
                        replacements = [
                            ('src="/static/', f'src="/{_ngrok_secret}/static/'),
                            ('href="/static/', f'href="/{_ngrok_secret}/static/'),
                            ('url("/static/', f'url("/{_ngrok_secret}/static/'),
                            ("url('/static/", f"url('/{_ngrok_secret}/static/"),
                            ('src="/socket.io/', f'src="/{_ngrok_secret}/socket.io/'),
                            # Socket.IO specific replacements
                            (
                                "socketPath = '/socket.io'",
                                f"socketPath = '/{_ngrok_secret}/socket.io'",
                            ),
                            (
                                'socketPath = "/socket.io"',
                                f'socketPath = "/{_ngrok_secret}/socket.io"',
                            ),
                            # This is key - the connect call doesn't need rewriting but path does
                            # Line will be: io.connect('/live-reload', { path: socketPath })
                            # socketPath is already rewritten above
                            ("fetch('/", f"fetch('/{_ngrok_secret}/"),
                            ('fetch("/', f'fetch("/{_ngrok_secret}/'),
                            ('"/api/', f'"/{_ngrok_secret}/api/'),
                            ("'/api/", f"'/{_ngrok_secret}/api/"),
                        ]
                        for old, new in replacements:
                            html = html.replace(old, new)
                        data = html.encode("utf-8")
                    except:
                        pass  # If rewriting fails, return original

                # If it's JSON content from API endpoints, rewrite git_url
                elif "application/json" in content_type and isinstance(data, bytes):
                    try:
                        import json

                        json_data = json.loads(data.decode("utf-8"))

                        # For /api/spec endpoint, fix the git_url to use the ngrok URL with secret
                        if "git_url" in json_data:
                            # We know this is coming through ngrok, so construct the URL from NGROK_INFO
                            try:
                                ngrok_info_path = os.path.join(
                                    "build", "praxis", "NGROK_INFO.txt"
                                )
                                if os.path.exists(ngrok_info_path):
                                    with open(ngrok_info_path, "r") as f:
                                        lines = f.read().strip().split("\n")
                                        if len(lines) >= 1:
                                            base_url = lines[
                                                0
                                            ]  # First line is base URL
                                            # Replace any localhost URL with the proper ngrok URL
                                            json_data["git_url"] = (
                                                f"{base_url}/{_ngrok_secret}"
                                            )
                                            # Also update masked_git_url if present
                                            if "masked_git_url" in json_data:
                                                from praxis.utils import mask_git_url

                                                json_data["masked_git_url"] = (
                                                    mask_git_url(json_data["git_url"])
                                                )
                            except:
                                pass

                        # For /api/agents endpoint, fix agent URLs
                        if "agents" in json_data:
                            try:
                                ngrok_info_path = os.path.join(
                                    "build", "praxis", "NGROK_INFO.txt"
                                )
                                if os.path.exists(ngrok_info_path):
                                    with open(ngrok_info_path, "r") as f:
                                        lines = f.read().strip().split("\n")
                                        if len(lines) >= 1:
                                            base_url = lines[
                                                0
                                            ]  # First line is base URL
                                            for agent in json_data["agents"]:
                                                # Fix self agents that point to localhost
                                                if agent.get("name", "").startswith(
                                                    "self"
                                                ) and "localhost" in agent.get(
                                                    "url", ""
                                                ):
                                                    agent["url"] = (
                                                        f"{base_url}/{_ngrok_secret}"
                                                    )
                                                    if "masked_url" in agent:
                                                        from praxis.utils import (
                                                            mask_git_url,
                                                        )

                                                        agent["masked_url"] = (
                                                            mask_git_url(agent["url"])
                                                        )
                                                # Origin and other remotes should remain as-is
                            except:
                                pass

                        data = json.dumps(json_data).encode("utf-8")
                    except:
                        pass  # If JSON rewriting fails, return original

                # Return the response with potentially rewritten content
                return data, response.status_code, response.headers

        except Exception as e:
            pass  # Proxy error
            from flask import abort

            return abort(404)

    pass  # Set up catch-all route


def request_middleware(request, response=None):
    """Middleware to handle ngrok bypass headers and block unauthorized requests."""
    if response is not None:
        # This is the after_request phase - add the bypass header
        response.headers["ngrok-skip-browser-warning"] = "true"
        return None

    # This is the before_request phase - only check if ngrok request is authorized
    # Check if this request is coming through ngrok
    is_ngrok_request = (
        "ngrok-skip-browser-warning" in request.headers
        or "X-Forwarded-Host" in request.headers
        or request.host.endswith(".ngrok-free.app")
        or request.host.endswith(".ngrok.io")
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
        return None  # Authorized request  # Let the catch-all route handle it

    # Block unauthorized ngrok requests
    pass  # Blocking unauthorized request
    from flask import abort

    return abort(404)


def cleanup():
    """Cleanup ngrok resources."""
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
