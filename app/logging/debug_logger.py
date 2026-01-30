# app/logging/debug_logger.py
"""
Structured debug logging with event types, context binding, and blob storage.

Provides a lightweight event-based logging system with:
- Fixed vocabulary of events for consistent grepping
- Context variable binding for correlation IDs
- Automatic blob storage for large fields
- Daily log file rotation
"""

import logging
import logging.handlers
import contextvars
import uuid
import json
import threading
import os
import hashlib
from pathlib import Path
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
from typing import Any

from ._config import BLOB_DIR, INCLUDE_STACK_TRACES, MAX_FIELD_LENGTH

# ==================== Event Types ====================


class EventType(Enum):
    """Fixed vocabulary of events for consistent grepping and diffing."""

    SCHEDULE_START = "SCHEDULE_START"
    SCHEDULE_END = "SCHEDULE_END"
    AMENDMENT_IDENTIFIED = "AMENDMENT_IDENTIFIED"
    AMENDMENT_APPLYING = "AMENDMENT_APPLYING"
    AMENDMENT_APPLIED = "AMENDMENT_APPLIED"
    AMENDMENT_FAILED = "AMENDMENT_FAILED"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_RETRY = "LLM_RETRY"
    LLM_SETTINGS = "LLM_SETTINGS"
    XML_PARSE_ERROR = "XML_PARSE_ERROR"
    XML_VALIDATION_ERROR = "XML_VALIDATION_ERROR"
    EDITORIAL_NOTE_INSERTED = "EDITORIAL_NOTE_INSERTED"
    EDITORIAL_NOTE_REF_INSERTED = "EDITORIAL_NOTE_REF_INSERTED"
    ERROR_COMMENT_INSERTED = "ERROR_COMMENT_INSERTED"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    CANDIDATE_SKIPPED = "CANDIDATE_SKIPPED"
    CANDIDATE_IDENTIFIED = "CANDIDATE_IDENTIFIED"
    IDENTIFICATION_SUMMARY = "IDENTIFICATION_SUMMARY"
    PATTERN_EXTRACTION_START = "PATTERN_EXTRACTION_START"
    PATTERN_EXTRACTION_SUCCESS = "PATTERN_EXTRACTION_SUCCESS"
    PATTERN_EXTRACTION_FAILED = "PATTERN_EXTRACTION_FAILED"
    EACH_PLACE_APPLICATION = "EACH_PLACE_APPLICATION"

    def __str__(self) -> str:
        """Return the raw event keyword."""
        return self.value


# ==================== Public Interface ====================


def configure_root_logging() -> None:
    """
    Configure root logger with daily rotation and console output.

    Sets up:
    - Daily log files in logs/debug/YYYY-MM-DD.log format
    - Console output with same formatting
    - Context injection and truncation

    This is idempotent - if handlers already exist, does nothing.
    """
    root = logging.getLogger()

    # Skip if already configured
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    # Create formatter and filter
    formatter = TruncatingFormatter()
    context_filter = ContextFilter()

    # Create log directory
    log_dir = Path("logs/debug")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create custom rotating handler for daily files
    file_handler = DailyFileHandler(log_dir)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)

    # Attach handlers to the root logger
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Reduce verbosity of chatty third-party libraries
    # These libraries produce excessive DEBUG/INFO logs that clutter output.
    # We set them to WARNING level to suppress noise while preserving important errors.
    # Set DEBUG_THIRD_PARTY=1 to re-enable detailed logging when troubleshooting.
    noisy_libs = (
        "openai",  # Azure / OpenAI client
        "httpx",  # HTTP client used by openai & others
        "httpcore",  # httpx transport layer
        "urllib3",  # requests / botocore internals
        "botocore",  # AWS low-level SDK
        "semantic_kernel",  # Microsoft Semantic Kernel framework
    )

    # Check debug mode from environment
    debug_mode = os.getenv("DEBUG", "1") == "1"

    # Force ERROR level for noisy libraries
    for name in noisy_libs:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Werkzeug: Keep startup banner, hide per-request logs in production
    werkzeug_level = logging.INFO if debug_mode else logging.WARNING
    logging.getLogger("werkzeug").setLevel(werkzeug_level)

    # Configure app verbosity based on debug mode
    # DEBUG mode: Full verbosity for deep debugging
    # Production: Only high-level events and warnings
    app_level = logging.DEBUG if debug_mode else logging.INFO
    services_level = logging.DEBUG if debug_mode else logging.WARNING

    logging.getLogger("app").setLevel(app_level)
    logging.getLogger("app.services").setLevel(services_level)
    logging.getLogger("app.kernel").setLevel(services_level)
    logging.getLogger("app.benchmarking").setLevel(services_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def event(logger: logging.Logger, ev: EventType, msg: str = "", /, **kw) -> None:
    """
    Log an event with optional message and key-value pairs.

    Args:
        logger: Logger instance to use
        ev: Event type from the fixed vocabulary
        msg: Optional message (defaults to event name)
        **kw: Additional key-value pairs to log
    """
    logger.info(msg or str(ev), extra={"event": str(ev), **kw})


@contextmanager
def bind(**kwargs):
    """
    Bind context variables for the duration of this block.

    Currently supports:
    - schedule_id: Schedule identifier
    - amendment_id: Amendment identifier
    - prompt_id: Prompt identifier
    - candidate_eid: Candidate provision identifier

    Note: Context variables are thread-local. When using ThreadPoolExecutor,
    wrap submissions with contextvars.copy_context().run(fn, *args) to preserve bindings.

    Args:
        **kwargs: Context variables to bind

    Yields:
        None
    """
    tokens = []
    try:
        # Set context variables
        if "schedule_id" in kwargs:
            tokens.append(ctx_schedule_id.set(kwargs["schedule_id"]))
        if "amendment_id" in kwargs:
            tokens.append(ctx_amendment_id.set(kwargs["amendment_id"]))
        if "prompt_id" in kwargs:
            tokens.append(ctx_prompt_id.set(kwargs["prompt_id"]))
        if "candidate_eid" in kwargs:
            tokens.append(ctx_candidate_eid.set(kwargs["candidate_eid"]))
        yield
    finally:
        # Reset context variables
        for token in tokens:
            token.var.reset(token)


# ==================== Context Management ====================

# Context variables for correlation IDs
ctx_schedule_id = contextvars.ContextVar("schedule_id", default=None)
ctx_amendment_id = contextvars.ContextVar("amendment_id", default=None)
ctx_prompt_id = contextvars.ContextVar("prompt_id", default=None)
ctx_candidate_eid = contextvars.ContextVar("candidate_eid", default=None)


class ContextFilter(logging.Filter):
    """Inject context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context variables to record if set.

        Args:
            record: Log record to augment

        Returns:
            Always True (never filters out records)
        """
        if schedule_id := ctx_schedule_id.get():
            record.schedule_id = schedule_id
        if amendment_id := ctx_amendment_id.get():
            record.amendment_id = amendment_id
        if prompt_id := ctx_prompt_id.get():
            record.prompt_id = prompt_id
        if candidate_eid := ctx_candidate_eid.get():
            record.candidate_eid = candidate_eid
        return True


# ==================== Formatting and Truncation ====================


class TruncatingFormatter(logging.Formatter):
    """Format logs with truncation and blob storage for large fields."""

    # Built-in attributes to skip when formatting extras
    SKIP_ATTRS = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
        "taskName",
    }

    # Keys that should always be stored as blobs
    BLOB_KEYS = {"assistant_text", "prompt_body"}

    def format(self, record: logging.LogRecord) -> str:
        """
        Format record with timestamp, level, logger, event, and key=value pairs.

        Format: <timestamp> <level> <logger> <event> key1=value1 key2=value2

        Large values are stored as blobs and replaced with key=blob:<path>

        Args:
            record: Log record to format

        Returns:
            Formatted log line with optional stack trace
        """
        # Format timestamp with timezone
        ts = datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")

        # Base format: timestamp level logger event_or_msg
        level = record.levelname[:5]
        logger = record.name[:35] if len(record.name) > 35 else record.name

        # Get event or use message
        event = getattr(record, "event", record.getMessage())
        base = f"{ts} {level:<5} {logger:<35} {event}"

        # Collect extra attributes
        extras = []
        for key, value in sorted(record.__dict__.items()):
            if key in self.SKIP_ATTRS or key == "event":
                continue

            # Handle different value types
            formatted_value = self._format_value(key, value)
            extras.append(f"{key}={formatted_value}")

        # Build final message
        if extras:
            base += " " + " ".join(extras)

        # Handle stack traces
        if record.exc_info and INCLUDE_STACK_TRACES:
            import traceback

            base += "\n" + "".join(traceback.format_exception(*record.exc_info))

        return base

    def _format_value(self, key: str, value: Any) -> str:
        """
        Format a single value, storing large content as blob.

        Args:
            key: The field key
            value: Value to format

        Returns:
            Formatted value or blob reference
        """
        # Check if this key should be blob-stored
        if key in self.BLOB_KEYS and isinstance(value, str):
            sha = hashlib.sha256(value.encode()).hexdigest()
            blob_dir = BLOB_DIR / datetime.now().strftime("%Y-%m-%d")
            blob_dir.mkdir(parents=True, exist_ok=True)
            (blob_dir / f"{sha}.txt").write_text(value, "utf-8")
            return f"[[ STORED_AS_BLOB {sha}.txt LENGTH={len(value)} ]]"

        # Convert dicts/lists to JSON
        if isinstance(value, (dict, list)):
            try:
                value = json.dumps(value, separators=(",", ":"))
            except (TypeError, ValueError):
                value = str(value)

        # Convert bytes to string
        elif isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = repr(value)

        # Convert everything else to string
        else:
            value = str(value)

        # Check length and store as blob if needed
        if len(value) > MAX_FIELD_LENGTH:
            blob_path = _store_blob(value)
            return f"blob:{blob_path}"

        return value


class DailyFileHandler(logging.Handler):
    """
    Thread-safe handler that writes to daily log files.

    Creates files named YYYY-MM-DD.log in the specified directory.
    """

    def __init__(self, log_dir: Path):
        """
        Initialise handler.

        Args:
            log_dir: Directory to store log files
        """
        super().__init__()
        self.log_dir = log_dir
        self._current_date = None
        self._current_file = None
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record to the appropriate daily file.

        Thread-safe implementation that handles date rollover.

        Args:
            record: Log record to emit
        """
        try:
            with self._lock:
                # Get current date
                record_date = datetime.fromtimestamp(record.created).date()

                # Open new file if date changed
                if record_date != self._current_date:
                    if self._current_file:
                        self._current_file.close()

                    filename = self.log_dir / f"{record_date.strftime('%Y-%m-%d')}.log"
                    self._current_file = open(filename, "a", encoding="utf-8")
                    self._current_date = record_date

                # Format and write record
                msg = self.format(record)
                self._current_file.write(msg + "\n")
                self._current_file.flush()

                # Force OS-level flush for durability
                os.fsync(self._current_file.fileno())

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the current file if open."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None
        super().close()


# ==================== Internal Utilities ====================


def _store_blob(text: str) -> str:
    """
    Store oversize text in blob file.

    Args:
        text: Text content to store

    Returns:
        Relative path to blob file
    """
    blob_id = uuid.uuid4()
    blob_path = BLOB_DIR / f"{blob_id}.txt"
    blob_path.write_text(text, encoding="utf-8")

    # Return path relative to current directory
    try:
        return blob_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        # If not relative to cwd, return absolute path
        return blob_path.as_posix()
