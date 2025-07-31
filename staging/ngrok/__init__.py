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
    networking_group.add_argument(
        "--ngrok-domain",
        type=str,
        default=None,
        help="Custom ngrok domain (e.g., my-app.ngrok-free.app). If not specified, will auto-detect your reserved domains.",
    )


class NgrokTunnel:
    """Manages ngrok tunnel for the API server using the Python SDK."""
    
    def __init__(self, host="localhost", port=5000, auth_token=None, domain=None):
        self.host = host
        self.port = port
        self.auth_token = auth_token or os.getenv("NGROK_AUTHTOKEN")
        self.domain = domain
        self.listener = None
        self.session = None
        self.public_url = None
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
            
            return self.public_url is not None
                
        except Exception as e:
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
        except Exception:
            # Silently fail - error handling is done at the higher level
            pass
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
    
    async def _setup_tunnel(self):
        """Set up the ngrok tunnel asynchronously."""
        import ngrok
        
        # Create session with auth token
        self.session = await ngrok.SessionBuilder().authtoken(self.auth_token).connect()
        
        # Create HTTP endpoint with optional domain
        endpoint_builder = self.session.http_endpoint()
        
        if self.domain:
            # Use specified domain
            endpoint_builder = endpoint_builder.domain(self.domain)
        else:
            # Try to auto-detect reserved domains
            try:
                domains = await self._get_reserved_domains()
                if domains:
                    # Use the first available domain
                    endpoint_builder = endpoint_builder.domain(domains[0])
                    print(f"🎯 Using reserved domain: {domains[0]}")
                else:
                    print("💡 No reserved domains found, using ephemeral URL")
            except Exception:
                # Fall back to ephemeral URL if domain detection fails
                print("💡 Using ephemeral URL (domain detection failed)")
        
        # Create HTTP listener
        self.listener = await endpoint_builder.listen()
        self.public_url = self.listener.url()
        
        # Forward traffic to local server
        await self.listener.forward(f"http://{self.host}:{self.port}")
    
    async def _get_reserved_domains(self):
        """Get list of reserved domains for this account."""
        try:
            import ngrok
            # List reserved domains through the session
            domains = await self.session.reserved_domains().list()
            return [domain.domain for domain in domains if domain.domain]
        except Exception:
            return []
    
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
    
    # Get auth token and domain from global args or environment
    auth_token = None
    domain = None
    try:
        from cli import get_cli_args
        args = get_cli_args()
        auth_token = getattr(args, 'ngrok_auth_token', None) or os.getenv("NGROK_AUTHTOKEN")
        domain = getattr(args, 'ngrok_domain', None)
    except:
        auth_token = os.getenv("NGROK_AUTHTOKEN")
    
    print(f"🚀 Starting ngrok tunnel for {host}:{port}")
    _tunnel = NgrokTunnel(host, port, auth_token, domain)
    success = _tunnel.start()
    
    if success:
        print(f"🌐 Ngrok tunnel active: {_tunnel.get_public_url()}")
        print(f"📡 Local server: http://{host}:{port}")
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
    
    # Check for custom domain (optional)
    domain = getattr(args, 'ngrok_domain', None) or os.getenv("NGROK_DOMAIN")
    if domain:
        print(f"🎯 Will use custom domain: {domain}")
    else:
        print("🔍 Will auto-detect reserved domains")
    
    print("✅ Ngrok ready")
    return {}


def cleanup():
    """Cleanup ngrok resources."""
    global _tunnel
    
    if _tunnel is not None:
        _tunnel.stop()
        _tunnel = None