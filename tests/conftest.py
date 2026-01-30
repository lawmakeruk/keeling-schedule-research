# tests/conftest.py

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """A test client for the app."""
    with app.test_client() as test_client:
        yield test_client
