"""
Unit tests for the Flask application factory and configuration.
"""

import os
from unittest.mock import patch, call
from app import create_app, _configure_app, _register_blueprints


class TestAppInit:
    """Test cases for app initialisation functions."""

    def test_create_app_basic(self):
        """Test basic app creation."""
        app = create_app()
        assert app is not None
        assert app.config["PREFERRED_URL_SCHEME"] == "https"

    def test_create_app_with_config_name(self):
        """Test app creation with config name (even though it's not used)."""
        app = create_app(config_name="testing")
        assert app is not None

    @patch.dict(os.environ, {"WERKZEUG_RUN_MAIN": "true"})
    def test_create_app_with_banner_werkzeug(self):
        """Test that banner is printed when WERKZEUG_RUN_MAIN is true."""
        with patch("app.logger") as mock_logger:
            app = create_app()

            # Verify banner was printed
            expected_calls = [call("=" * 80), call("Keeling Schedule Service starting up"), call("=" * 80)]
            mock_logger.info.assert_has_calls(expected_calls)
            assert app is not None

    @patch.dict(os.environ, {"FLASK_ENV": "production"})
    def test_create_app_with_banner_production(self):
        """Test that banner is printed when FLASK_ENV is production."""
        with patch("app.logger") as mock_logger:
            app = create_app()

            # Verify banner was printed
            expected_calls = [call("=" * 80), call("Keeling Schedule Service starting up"), call("=" * 80)]
            mock_logger.info.assert_has_calls(expected_calls)
            assert app is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_create_app_no_banner(self):
        """Test that banner is NOT printed in development mode."""
        # Ensure neither env var is set
        if "WERKZEUG_RUN_MAIN" in os.environ:
            del os.environ["WERKZEUG_RUN_MAIN"]
        if "FLASK_ENV" in os.environ:
            del os.environ["FLASK_ENV"]

        with patch("app.logger") as mock_logger:
            app = create_app()

            # Verify banner was NOT printed
            banner_calls = [call("=" * 80), call("Keeling Schedule Service starting up"), call("=" * 80)]
            for banner_call in banner_calls:
                assert banner_call not in mock_logger.info.call_args_list
            assert app is not None

    @patch.dict(os.environ, {"PREFERRED_HOST": "example.com"})
    def test_configure_app_with_preferred_host(self):
        """Test app configuration with PREFERRED_HOST environment variable."""
        from flask import Flask

        app = Flask(__name__)

        # Mock the module-level logger
        with patch("app.logger") as mock_logger:
            _configure_app(app)

            assert app.config["PREFERRED_HOST"] == "example.com"
            mock_logger.info.assert_called_once_with("Configured preferred host: example.com")

    def test_configure_app_without_preferred_host(self):
        """Test app configuration without PREFERRED_HOST environment variable."""
        from flask import Flask

        app = Flask(__name__)

        # Ensure PREFERRED_HOST is not set
        with patch.dict(os.environ, {}, clear=True):
            _configure_app(app)

            assert "PREFERRED_HOST" not in app.config
            assert app.config["PREFERRED_URL_SCHEME"] == "https"

    def test_register_blueprints(self):
        """Test blueprint registration."""
        from flask import Flask

        app = Flask(__name__)

        _register_blueprints(app)

        # Check that the api blueprint was registered
        assert len(app.blueprints) == 1
        assert "api" in app.blueprints

    def test_full_app_creation_integration(self):
        """Integration test for full app creation process."""
        app = create_app()

        # Verify app is properly configured
        assert app is not None
        assert app.config["PREFERRED_URL_SCHEME"] == "https"
        assert "api" in app.blueprints

        # Test with PREFERRED_HOST
        with patch.dict(os.environ, {"PREFERRED_HOST": "test.example.com"}):
            app2 = create_app()
            assert app2.config["PREFERRED_HOST"] == "test.example.com"
