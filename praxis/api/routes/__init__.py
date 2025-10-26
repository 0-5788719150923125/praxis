"""API route modules."""

from flask import Flask
from .core import core_bp
from .generation import generation_bp
from .agents import agents_bp
from .git import git_bp
from .metrics import metrics_bp
from .data_metrics import data_metrics_bp
from .static import static_bp


def register_routes(app: Flask) -> None:
    """Register all route blueprints with the Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(core_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(agents_bp)
    app.register_blueprint(git_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(data_metrics_bp)
    app.register_blueprint(static_bp)
