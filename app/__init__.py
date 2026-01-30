# app/__init__.py
"""
Keeling Schedule Service Flask Application.
Provides the application factory for creating Flask instances with proper
configuration for running behind a TLS-terminating proxy.
"""
import os
from flask import Flask
from app.logging import debug_logger
from app.logging.debug_logger import get_logger

# Use the structured logger
logger = get_logger(__name__)


def create_app(config_name: str = None) -> Flask:
    """
    Application factory for creating Flask instances.

    Args:
        config_name: Optional configuration name for different environments

    Returns:
        Configured Flask application instance
    """
    # Configure debug logging
    debug_logger.configure_root_logging()

    # Only print banner in the main process, not the reloader's parent
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_ENV") == "production":
        logger.info("=" * 80)
        logger.info("Keeling Schedule Service starting up")
        logger.info("=" * 80)

    app = Flask(__name__)

    _configure_app(app)
    _register_blueprints(app)

    return app


def _configure_app(app: Flask) -> None:
    """Configure Flask application settings."""
    # Configure Flask to work properly behind a TLS-terminating proxy
    app.config["PREFERRED_URL_SCHEME"] = "https"

    # Trust the X-Forwarded-* headers from the proxy
    preferred_host = os.environ.get("PREFERRED_HOST", None)
    if preferred_host:
        app.config["PREFERRED_HOST"] = preferred_host
        logger.info(f"Configured preferred host: {preferred_host}")


def _register_blueprints(app: Flask) -> None:
    """Register all application blueprints."""
    from .routes import api

    app.register_blueprint(api)
