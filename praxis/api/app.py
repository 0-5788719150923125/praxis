"""Flask application setup and configuration."""

import logging
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO

from .config import CORS_ORIGINS, SOCKETIO_ASYNC_MODE
from .middleware import process_request_middleware, process_response_middleware

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    force=True,
    handlers=[logging.StreamHandler()],
)

# Create loggers
api_logger = logging.getLogger("praxis.api")
api_logger.setLevel(logging.WARNING)

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)

# Set SocketIO/EngineIO logging to ERROR only
logging.getLogger("socketio").setLevel(logging.ERROR)
logging.getLogger("engineio").setLevel(logging.ERROR)
logging.getLogger("socketio.server").setLevel(logging.ERROR)
logging.getLogger("engineio.server").setLevel(logging.ERROR)

# Create Flask app with proper template and static paths
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
app = Flask(__name__,
            template_folder=os.path.join(project_root, "templates"),
            static_folder=os.path.join(project_root, "static"))
app.debug = False

# Enable CORS
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# Create SocketIO instance
socketio = SocketIO(
    app,
    async_mode=SOCKETIO_ASYNC_MODE,
    cors_allowed_origins=CORS_ORIGINS,
    logger=True,
    engineio_logger=True,
)


# Register middleware hooks
@app.before_request
def before_request_handler():
    """Process request middleware."""
    return process_request_middleware(request)


@app.after_request
def after_request_handler(response):
    """Process response middleware."""
    return process_response_middleware(request, response)