# tests/unit/logging/test_logging_config.py
"""Tests for debug logging configuration."""

import sys
import pytest


def reload_config(monkeypatch, **env):
    """Helper to reload config module with environment variables."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # Remove all related modules from sys.modules to force fresh import
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith("app.logging")]
    for module in modules_to_remove:
        del sys.modules[module]

    # Import using full module path
    import app.logging._config

    return app.logging._config


def test_default_config_values():
    """Test that default configuration values are set correctly."""
    # Import the module directly
    import app.logging._config as config

    assert config.MAX_FIELD_LENGTH == 8000
    assert config.BLOB_DIR.name == "blobs"
    assert config.BLOB_DIR.parent.name == "logs"
    assert config.INCLUDE_STACK_TRACES is True


def test_env_overrides(monkeypatch, tmp_path):
    """Test that environment variables override default values."""
    # Set environment variables and reload
    custom_blob_dir = tmp_path / "custom_blobs"
    config = reload_config(monkeypatch, KS_MAX_FIELD_LEN="1234", KS_BLOB_DIR=str(custom_blob_dir), KS_STACK_TRACES="0")

    # Verify overrides
    assert config.MAX_FIELD_LENGTH == 1234
    assert config.BLOB_DIR == custom_blob_dir.absolute()
    assert config.INCLUDE_STACK_TRACES is False

    # Verify blob directory was created
    assert custom_blob_dir.exists()
    assert custom_blob_dir.is_dir()


def test_blob_dir_creation(tmp_path, monkeypatch):
    """Test that blob directory is created automatically."""
    # Set a non-existent directory
    new_blob_dir = tmp_path / "new" / "nested" / "blobs"
    assert not new_blob_dir.exists()

    # Reload with new directory
    _ = reload_config(monkeypatch, KS_BLOB_DIR=str(new_blob_dir))

    # Directory should now exist
    assert new_blob_dir.exists()
    assert new_blob_dir.is_dir()


def test_invalid_env_values(monkeypatch):
    """Test handling of invalid environment variable values."""
    # Non-numeric MAX_FIELD_LEN should raise ValueError
    monkeypatch.setenv("KS_MAX_FIELD_LEN", "not-a-number")

    # Remove all related modules
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith("app.logging")]
    for module in modules_to_remove:
        del sys.modules[module]

    # The ValueError will be raised when the module tries to convert to int
    with pytest.raises(ValueError, match="invalid literal for int"):
        import app.logging._config  # noqa: F401


def test_stack_traces_env_parsing(monkeypatch):
    """Test various values for KS_STACK_TRACES environment variable."""
    test_cases = [
        ("1", True),
        ("0", False),
        ("true", False),  # Only "1" is True
        ("false", False),
        ("", False),
        ("yes", False),
    ]

    for env_value, expected in test_cases:
        config = reload_config(monkeypatch, KS_STACK_TRACES=env_value)
        assert config.INCLUDE_STACK_TRACES is expected, f"Failed for value: {env_value}"
