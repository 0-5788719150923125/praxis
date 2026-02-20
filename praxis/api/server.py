"""API server management."""

import logging
import os
import time
from datetime import datetime
from threading import Event, Thread
from typing import Any, Optional

from .app import api_logger, app, socketio, werkzeug_logger
from .config import (
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    DEV_LOG_LEVEL,
    SERVER_START_TIMEOUT,
)
from .middleware import (
    apply_wsgi_middleware,
    register_request_middleware,
    register_response_middleware,
)
from .routes import register_routes
from .utils import find_available_port
from .utils.file_watcher import TemplateWatcher
from .websocket import setup_live_reload, setup_metrics_live_namespace


class APIServer:
    """API server for the Praxis model."""

    def __init__(
        self,
        generator,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        tokenizer=None,
        integration_loader=None,
        param_stats=None,
        seed=None,
        truncated_hash=None,
        full_hash=None,
        dev_mode: bool = False,
        dashboard=None,
        launch_command=None,
        config_file=None,
    ):
        """Initialize the API server.

        Args:
            generator: Generator instance for model inference
            host: Host to bind to
            port: Port to bind to (will find next available if in use)
            tokenizer: Tokenizer instance
            integration_loader: Integration loader for middleware
            param_stats: Parameter statistics
            seed: Random seed
            truncated_hash: Truncated model hash
            full_hash: Full model hash
            dev_mode: Whether to run in development mode
            dashboard: Dashboard instance for terminal streaming
            launch_command: Command used to launch the model
            config_file: Path to experiment config file
        """
        self.generator = generator
        self.dashboard = dashboard
        self.host = host
        self.port = find_available_port(port)
        self.parent_pid = os.getppid()
        self.seed = seed
        self.tokenizer = tokenizer
        self.integration_loader = integration_loader
        self.param_stats = param_stats if param_stats else {}
        self.truncated_hash = truncated_hash
        self.full_hash = full_hash
        self.launch_command = launch_command
        self.config_file = config_file
        self.dev_mode = dev_mode
        self.launch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.server_thread = None
        self.server = None
        self.started = Event()
        self.shutdown_event = Event()

        # Configure logging based on dashboard and dev_mode
        self._configure_logging()

        # Initialize template watcher
        self.template_watcher = TemplateWatcher(socketio)

        # Set up WebSocket namespaces
        setup_live_reload(socketio)
        setup_metrics_live_namespace(socketio)

    def _configure_logging(self) -> None:
        """Configure logging based on dashboard and dev mode."""
        if self.dashboard:
            # When dashboard is active, route logs through dashboard handler
            from praxis.interface.io.handlers import DashboardStreamHandler

            for logger in [
                api_logger,
                werkzeug_logger,
                logging.getLogger("socketio"),
                logging.getLogger("engineio"),
            ]:
                logger.handlers = []
                dashboard_handler = DashboardStreamHandler(self.dashboard)
                dashboard_handler.setFormatter(
                    logging.Formatter("[%(name)s] %(message)s")
                )
                logger.addHandler(dashboard_handler)
                logger.propagate = False

            # Set appropriate log levels
            log_level = DEV_LOG_LEVEL if self.dev_mode else DEFAULT_LOG_LEVEL
            api_logger.setLevel(log_level)

            # Werkzeug should use WARNING to avoid request spam
            werkzeug_logger.setLevel(logging.WARNING)

            # SocketIO/EngineIO should only log errors
            for logger in [
                logging.getLogger("socketio"),
                logging.getLogger("engineio"),
                logging.getLogger("socketio.server"),
                logging.getLogger("engineio.server"),
            ]:
                logger.setLevel(logging.ERROR)

            if self.dev_mode:
                api_logger.info("API server running in development mode with dashboard")
        else:
            # Normal console logging when no dashboard
            log_level = DEV_LOG_LEVEL if self.dev_mode else DEFAULT_LOG_LEVEL
            api_logger.setLevel(log_level)

            # Werkzeug should use WARNING to avoid request spam
            werkzeug_logger.setLevel(logging.WARNING)

            # SocketIO/EngineIO should only log errors
            for logger in [
                logging.getLogger("socketio"),
                logging.getLogger("engineio"),
                logging.getLogger("socketio.server"),
                logging.getLogger("engineio.server"),
            ]:
                logger.setLevel(logging.ERROR)

            if self.dev_mode:
                api_logger.info("API server running in development mode")

    def update_param_stats(self, param_stats: dict) -> None:
        """Update the parameter statistics after optimizer creation.

        Args:
            param_stats: Updated parameter statistics
        """
        self.param_stats = param_stats

    def _monitor_parent(self) -> None:
        """Monitor thread that checks if parent process is alive."""
        while not self.shutdown_event.is_set():
            try:
                os.kill(self.parent_pid, 0)
                time.sleep(1)
            except OSError:
                # Parent process died, shutting down
                self.stop()
                break

    def start(self) -> None:
        """Start the API server."""
        if self.server_thread is not None:
            return  # Already started

        # Start the template watcher
        self.template_watcher.start()

        # Register routes FIRST (before any requests)
        with app.app_context():
            # Only register routes if not already registered
            if not hasattr(app, "_routes_registered"):
                register_routes(app)
                app._routes_registered = True

        # Set up Flask app config
        app.config["api_server"] = self
        app.config["configured_host"] = self.host
        app.config["configured_port"] = self.port
        app.config["generator"] = self.generator
        app.config["tokenizer"] = self.tokenizer
        app.config["integration_loader"] = self.integration_loader
        app.config["seed"] = self.seed
        app.config["truncated_hash"] = self.truncated_hash
        app.config["full_hash"] = self.full_hash
        app.config["launch_command"] = self.launch_command
        app.config["launch_timestamp"] = self.launch_timestamp
        app.config["param_stats"] = self.param_stats
        app.config["config_file"] = self.config_file
        app.config["repo_root"] = os.getcwd()  # Store the repository root at startup

        # Start the server thread
        self.server_thread = Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Start the monitor thread
        self.monitor_thread = Thread(target=self._monitor_parent)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.started.wait(timeout=SERVER_START_TIMEOUT)
        if not self.started.is_set():
            raise RuntimeError("Server failed to start within the timeout period")

        addr = self.get_api_addr()
        url = f"{addr}/" if addr.startswith(("http://", "https://")) else f"http://{addr}/"
        api_logger.info(f"API Server started at {url}")
        print(f"[API] Server started at {url}")

    def stop(self) -> None:
        """Stop the API server."""
        self.shutdown_event.set()

        if hasattr(self, "template_watcher"):
            try:
                self.template_watcher.stop()
            except:
                pass

    def _run_server(self) -> None:
        """Run the server in a thread."""
        try:
            api_logger.info("Starting API server thread...")

            # Apply WSGI middleware
            apply_wsgi_middleware(app)

            with app.app_context():
                # Register integration middleware
                if self.integration_loader:
                    for (
                        middleware_func
                    ) in self.integration_loader.get_request_middleware():

                        def create_wrapper(func):
                            def request_wrapper(req, resp):
                                return func(req, resp)

                            return request_wrapper

                        wrapper = create_wrapper(middleware_func)
                        # Only register as request middleware (not response)
                        # Registering as both causes "write() before start_response" errors
                        register_request_middleware(wrapper)

                # Signal that the server will start
                self.started.set()

                # Configure and run server
                if self.dev_mode:
                    api_logger.info(f"Starting dev mode server on port {self.port}")
                    app.debug = True
                    socketio.run(
                        app,
                        host="0.0.0.0",
                        port=self.port,
                        debug=True,
                        use_reloader=False,
                        allow_unsafe_werkzeug=True,
                    )
                else:
                    api_logger.info(
                        f"Starting production mode server on port {self.port}"
                    )
                    app.debug = False
                    socketio.run(
                        app,
                        host="0.0.0.0",
                        port=self.port,
                        debug=False,
                        use_reloader=False,
                        allow_unsafe_werkzeug=True,
                    )
        except Exception as e:
            api_logger.error(f"API server crashed with exception: {e}", exc_info=True)
            raise

    def get_api_addr(self) -> str:
        """Get the API server address.

        Returns:
            Server address as host:port string
        """
        return f"{self.host}:{self.port}"
