"""Ngrok integration module for Praxis."""

import os
import asyncio
import threading
from typing import Optional


def add_cli_args(parser):
    """Add ngrok CLI arguments to the parser."""
    networking_group = None
    
    # Find the 'networking' argument group
    for group in parser._action_groups:
        if group.title == 'networking':
            networking_group = group
            break
    
    if networking_group is None:
        networking_group = parser.add_argument_group('networking')
    
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
        import ngrok
        import secrets
        import string
        
        # Generate webhook secret (deterministic if seed provided)
        secret_seed = os.getenv("NGROK_SECRET_SEED")
        if secret_seed:
            # Use deterministic generation with seed
            import hashlib
            import base64
            
            # Create a hash from the seed and take first 32 chars (base64 safe)
            hash_bytes = hashlib.sha256(secret_seed.encode()).digest()
            self.webhook_secret = base64.urlsafe_b64encode(hash_bytes)[:32].decode()
            print(f"🔑 Using deterministic secret from seed")
        else:
            # Use secure random generation
            alphabet = string.ascii_letters + string.digits
            self.webhook_secret = ''.join(secrets.choice(alphabet) for _ in range(32))
            print(f"🎲 Using random secret")
        
        # Create session with auth token
        self.session = await ngrok.SessionBuilder().authtoken(self.auth_token).connect()
        
        # Try to use a hardcoded static domain (you can set this in .env)
        static_domain = os.getenv("NGROK_STATIC_DOMAIN")
        if static_domain:
            print(f"🔍 Found static domain in environment: {static_domain}")
        else:
            print("ℹ️  No static domain configured, using random URL")
        
        # Traffic policy: check authorization first, then rewrite if authorized
        traffic_policy = {
            "inbound": [
                {
                    "expressions": [f"!req.url.path.startsWith('/{self.webhook_secret}')"],
                    "actions": [
                        {
                            "type": "deny",
                            "config": {
                                "status_code": 401
                            }
                        }
                    ]
                },
                {
                    "expressions": [f"req.url.path == '/{self.webhook_secret}'"],
                    "actions": [
                        {
                            "type": "url-rewrite",
                            "config": {
                                "from": f"/{self.webhook_secret}",
                                "to": "/"
                            }
                        },
                        {
                            "type": "add-headers",
                            "config": {
                                "headers": {
                                    "X-Forwarded-Host": f"{self.host}:{self.port}",
                                    "X-Original-Host": "req.host"
                                }
                            }
                        }
                    ]
                },
                {
                    "expressions": [f"req.url.path.startsWith('/{self.webhook_secret}/')"],
                    "actions": [
                        {
                            "type": "url-rewrite",
                            "config": {
                                "from": f"/{self.webhook_secret}(.*)",
                                "to": "$1"
                            }
                        },
                        {
                            "type": "add-headers",
                            "config": {
                                "headers": {
                                    "X-Forwarded-Host": f"{self.host}:{self.port}",
                                    "X-Original-Host": "req.host"
                                }
                            }
                        }
                    ]
                }
            ]
        }
        
        # Create HTTP listener with traffic policy (must be JSON string)
        import json
        traffic_policy_json = json.dumps(traffic_policy)
        
        # Build the listener with optional domain
        listener_builder = self.session.http_endpoint().traffic_policy(traffic_policy_json)
        if static_domain:
            listener_builder = listener_builder.domain(static_domain)
            print(f"🌐 Using static domain: {static_domain}")
        
        self.listener = await listener_builder.listen()
        self.public_url = self.listener.url()
        
        # Forward traffic to local server
        await self.listener.forward(f"http://{self.host}:{self.port}")
    
    
    def stop(self):
        """Stop the ngrok tunnel."""
        try:
            if self._loop and not self._loop.is_closed():
                # Schedule cleanup in the event loop
                future = asyncio.run_coroutine_threadsafe(self._cleanup_async(), self._loop)
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
    
    if _tunnel is not None:
        print("⚠️  Ngrok tunnel already running")
        return
    
    # Load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get auth token from global args or environment
    auth_token = None
    try:
        from cli import get_cli_args
        args = get_cli_args()
        auth_token = getattr(args, 'ngrok_auth_token', None) or os.getenv("NGROK_AUTHTOKEN")
    except:
        auth_token = os.getenv("NGROK_AUTHTOKEN")
    
    print(f"🚀 Starting ngrok tunnel for {host}:{port}")
    _tunnel = NgrokTunnel(host, port, auth_token)
    success = _tunnel.start()
    
    if success:
        base_url = _tunnel.get_public_url()
        protected_url = f"{base_url}/{_tunnel.webhook_secret}"
        
        print(f"🌐 Ngrok tunnel active: {base_url}")
        print(f"🔐 Protected URL: {protected_url}")
        print(f"📡 Local server: http://{host}:{port}")
        print(f"ℹ️  URL rewriting:")
        print(f"     /{_tunnel.webhook_secret} -> /")
        print(f"     /{_tunnel.webhook_secret}/input/ -> /input/")
        print(f"🛡️  All other requests blocked with 401")
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
    auth_token = getattr(args, 'ngrok_auth_token', None) or os.getenv("NGROK_AUTHTOKEN")
    
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
    
    print("🔐 Will auto-generate secret with complete URL protection")
    print(f"ℹ️  URLs will use format: /SECRET instead of /webhook/SECRET")
    print("✅ Ngrok ready")
    return {}


def request_middleware(request, response=None):
    """Middleware to add ngrok bypass headers."""
    if response is not None:
        # This is the after_request phase - add the bypass header
        response.headers['ngrok-skip-browser-warning'] = 'true'
    # In before_request phase (response=None), we don't need to do anything
    

def cleanup():
    """Cleanup ngrok resources."""
    global _tunnel
    
    if _tunnel is not None:
        _tunnel.stop()
        _tunnel = None