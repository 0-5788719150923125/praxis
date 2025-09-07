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
        except Exception as e:
            print(f"Error stopping ngrok: {e}")

    async def _cleanup_async(self):
        """Clean up ngrok resources asynchronously."""
        if self.listener:
            await self.listener.close()
        if self.session:
            await self.session.close()


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
        """Hook called when API server starts."""
        # Only start tunnel if ngrok flag is set
        if not getattr(args, "ngrok", False):
            return

        print("🚀 STARTING NGROK TUNNEL...")

        # Get auth token from args or environment
        auth_token = getattr(args, "ngrok_auth_token", None) or os.getenv("NGROK_AUTHTOKEN")

        if not auth_token:
            print("❌ NGROK ERROR: No auth token provided. Set NGROK_AUTHTOKEN env var or use --ngrok-auth-token")
            return

        # Create and start tunnel
        self.tunnel = NgrokTunnel(
            host="localhost",
            port=5000,
            auth_token=auth_token
        )

        if self.tunnel.start():
            print(f"✅ NGROK TUNNEL ACTIVE: {self.tunnel.public_url}")
            
            # Store tunnel info in app config for other components to use
            app.config["ngrok_url"] = self.tunnel.public_url
            app.config["ngrok_webhook_secret"] = self.tunnel.webhook_secret
            
            # Print webhook information if available
            if self.tunnel.webhook_secret:
                print(f"📝 WEBHOOK SECRET: {self.tunnel.webhook_secret}")
                print(f"   Use this secret to validate webhook requests")
        else:
            print("❌ Failed to start ngrok tunnel")

    def request_middleware(self, request: Any, response: Any) -> None:
        """Middleware for modifying requests/responses."""
        # Add ngrok-specific headers if tunnel is active
        if self.tunnel and self.tunnel.public_url:
            # Add the public URL to response headers for client discovery
            response.headers["X-Ngrok-Public-URL"] = self.tunnel.public_url
            
            # If webhook secret exists, validate incoming webhook requests
            if self.tunnel.webhook_secret and request.path.startswith("/webhook"):
                provided_secret = request.headers.get("X-Webhook-Secret")
                if provided_secret != self.tunnel.webhook_secret:
                    # Invalid or missing webhook secret
                    response.status_code = 401
                    response.data = "Unauthorized: Invalid webhook secret"

    def cleanup(self) -> None:
        """Clean up ngrok resources."""
        if self.tunnel:
            print("🛑 Stopping ngrok tunnel...")
            self.tunnel.stop()
            self.tunnel = None