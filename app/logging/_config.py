# app/logging/_config.py
"""Configuration for debug logging system."""

from pathlib import Path
import os

# Maximum length for field values before they're stored as blobs
MAX_FIELD_LENGTH = int(os.getenv("KS_MAX_FIELD_LEN", 8000))

# Directory for storing large field values
BLOB_DIR = Path(os.getenv("KS_BLOB_DIR", "logs/blobs")).absolute()
BLOB_DIR.mkdir(parents=True, exist_ok=True)

# Whether to include stack traces in error logs (can be disabled for cleaner output)
INCLUDE_STACK_TRACES = os.getenv("KS_STACK_TRACES", "1") == "1"
