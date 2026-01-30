import os
import pytest
from app.config import Config


@pytest.fixture
def config():
    """Fixture to initialise Config before each test."""
    return Config()


def test_config_paths(config):
    """Test that Config initialises log and plugin paths correctly."""
    expected_log_path = os.path.join(os.getcwd(), "app", ".log")
    expected_plugin_path = os.path.join(os.getcwd(), "app", "kernel", "plugins")

    assert config.LOG_PATH == expected_log_path
    assert config.PROMPTS_PATH == expected_plugin_path


def test_config_preferred_url_scheme():
    """Test that Config correctly reads PREFERRED_URL_SCHEME from the environment."""
    # Set the environment variable for the test
    os.environ["PREFERRED_URL_SCHEME"] = "http"

    config = Config()  # Reinitialise to capture new env var
    assert config.PREFERRED_URL_SCHEME == "http"

    # Clean up by removing the environment variable
    del os.environ["PREFERRED_URL_SCHEME"]

    # Reinitialise Config to check default value
    config = Config()
    assert config.PREFERRED_URL_SCHEME == "https"  # Default value
