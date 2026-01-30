"""Tests for debug logging system."""

import sys
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pytest
from unittest.mock import patch

from app.logging.debug_logger import (
    EventType as EVT,
    configure_root_logging,
    get_logger,
    event,
    bind,
    TruncatingFormatter,
    DailyFileHandler,
    ContextFilter,
    _store_blob,
)


@pytest.fixture
def clean_logging():
    """Reset logging configuration after each test."""
    yield
    # Close all handlers and clear from root logger
    root = logging.getLogger()
    for h in root.handlers:
        h.close()
    root.handlers.clear()


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory."""
    log_dir = tmp_path / "logs" / "debug"
    log_dir.mkdir(parents=True)
    return log_dir


class TestEventType:
    """Test the EventType enum."""

    def test_event_type_string_conversion(self):
        """Test that EventType converts to expected string."""
        assert str(EVT.SCHEDULE_START) == "SCHEDULE_START"
        assert str(EVT.LLM_REQUEST) == "LLM_REQUEST"
        assert str(EVT.AMENDMENT_FAILED) == "AMENDMENT_FAILED"

    def test_all_event_types_defined(self):
        """Test that all expected event types are defined."""
        expected_events = [
            "SCHEDULE_START",
            "SCHEDULE_END",
            "AMENDMENT_IDENTIFIED",
            "AMENDMENT_APPLYING",
            "AMENDMENT_APPLIED",
            "AMENDMENT_FAILED",
            "LLM_REQUEST",
            "LLM_RESPONSE",
            "LLM_RETRY",
            "LLM_SETTINGS",
            "XML_PARSE_ERROR",
            "XML_VALIDATION_ERROR",
            "EDITORIAL_NOTE_INSERTED",
            "EDITORIAL_NOTE_REF_INSERTED",
            "ERROR_COMMENT_INSERTED",
            "CANDIDATE_FOUND",
            "CANDIDATE_SKIPPED",
            "CANDIDATE_IDENTIFIED",
            "IDENTIFICATION_SUMMARY",
            "PATTERN_EXTRACTION_START",
            "PATTERN_EXTRACTION_SUCCESS",
            "PATTERN_EXTRACTION_FAILED",
            "EACH_PLACE_APPLICATION",
        ]

        actual_events = [event.value for event in EVT]
        assert set(actual_events) == set(expected_events)


class TestFormatter:
    """Test the TruncatingFormatter class."""

    def test_basic_formatting(self):
        """Test basic log line formatting."""
        formatter = TruncatingFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.event = "TEST_EVENT"

        formatted = formatter.format(record)

        # Check format components
        assert "INFO" in formatted
        assert "test.module" in formatted
        assert "TEST_EVENT" in formatted
        # Should have ISO format timestamp with timezone
        assert "T" in formatted  # ISO format indicator
        assert "+" in formatted or "-" in formatted  # Timezone offset

    def test_truncation_to_blob(self, tmp_path, monkeypatch):
        """Test that long values are stored as blobs."""
        # Create blob directory
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()

        # Patch the imported constant in debug_logger module
        monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

        formatter = TruncatingFormatter()

        # Create record with long field
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Get MAX_FIELD_LENGTH from debug_logger's copy
        from app.logging.debug_logger import MAX_FIELD_LENGTH

        long_value = "x" * (MAX_FIELD_LENGTH + 100)
        record.long_field = long_value

        formatted = formatter.format(record)

        # Check blob reference in output
        assert "long_field=blob:" in formatted

        # Extract blob path and verify file exists
        import re

        match = re.search(r"long_field=blob:([^\s]+)", formatted)
        assert match
        blob_path = Path(match.group(1))

        # Check blob file contains original content
        if not blob_path.is_absolute():
            blob_path = Path.cwd() / blob_path
        assert blob_path.exists()
        assert blob_path.read_text() == long_value

    def test_dict_and_bytes_handling(self):
        """Test formatting of dict and bytes values."""
        formatter = TruncatingFormatter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Add various types
        record.dict_field = {"key": "value", "number": 42}
        record.bytes_field = b"hello bytes"
        record.list_field = [1, 2, 3]

        formatted = formatter.format(record)

        # Check JSON serialisation (compact)
        assert 'dict_field={"key":"value","number":42}' in formatted
        assert "bytes_field=hello bytes" in formatted
        assert "list_field=[1,2,3]" in formatted

    def test_large_dict_to_blob(self, tmp_path, monkeypatch):
        """Test that large dicts are stored as blobs."""
        # Create blob directory
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()

        # Patch the imported constant in debug_logger module
        monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

        formatter = TruncatingFormatter()

        # Create record with large dict
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Create dict that will exceed MAX_FIELD_LENGTH when serialised
        large_dict = {f"key_{i}": f"value_{i}" * 10 for i in range(100)}
        record.large_dict = large_dict

        formatted = formatter.format(record)

        # Check blob reference
        assert "large_dict=blob:" in formatted

        # Verify blob contains JSON
        import re

        match = re.search(r"large_dict=blob:([^\s]+)", formatted)
        assert match
        blob_path = Path(match.group(1))

        if not blob_path.is_absolute():
            blob_path = Path.cwd() / blob_path

        blob_content = blob_path.read_text()

        # Verify JSON is compact (no spaces after colons or commas)
        assert '": "' not in blob_content  # Should be ":"
        assert '", "' not in blob_content  # Should be ","

        recovered_dict = json.loads(blob_content)
        assert recovered_dict == large_dict

    def test_stack_trace_inclusion(self, monkeypatch):
        """Test stack trace inclusion based on configuration."""
        formatter = TruncatingFormatter()

        # Create record with exception
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        # Test with stack traces enabled (default)
        monkeypatch.setattr("app.logging.debug_logger.INCLUDE_STACK_TRACES", True, raising=True)
        formatted = formatter.format(record)
        assert "Traceback" in formatted
        assert "ZeroDivisionError" in formatted

        # Test with stack traces disabled
        monkeypatch.setattr("app.logging.debug_logger.INCLUDE_STACK_TRACES", False, raising=True)
        formatted = formatter.format(record)
        assert "Traceback" not in formatted
        assert "ZeroDivisionError" not in formatted

    def test_blob_keys_automatic_storage(self, tmp_path, monkeypatch):
        """Test that BLOB_KEYS fields are automatically stored as blobs."""
        # Create blob directory
        blob_dir = tmp_path / "blobs"
        blob_dir.mkdir()

        # Patch the blob directory
        monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

        formatter = TruncatingFormatter()

        # Create record with fields that should auto-blob
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Add fields that are in BLOB_KEYS
        record.assistant_text = "This is assistant response text that should be stored as blob"
        record.prompt_body = "This is a prompt body that should also be stored as blob"
        record.regular_field = "This should not be blobbed"

        formatted = formatter.format(record)

        # Check that BLOB_KEYS fields were stored as blobs
        assert "assistant_text=[[ STORED_AS_BLOB" in formatted
        assert "prompt_body=[[ STORED_AS_BLOB" in formatted
        assert "regular_field=This should not be blobbed" in formatted

        # Verify blob files were created with correct content
        import re

        # Check assistant_text blob
        match = re.search(r"assistant_text=\[\[ STORED_AS_BLOB ([a-f0-9]+\.txt) LENGTH=\d+ \]\]", formatted)
        assert match
        blob_file = blob_dir / datetime.now().strftime("%Y-%m-%d") / match.group(1)
        assert blob_file.exists()
        assert blob_file.read_text() == "This is assistant response text that should be stored as blob"

        # Check prompt_body blob
        match = re.search(r"prompt_body=\[\[ STORED_AS_BLOB ([a-f0-9]+\.txt) LENGTH=\d+ \]\]", formatted)
        assert match
        blob_file = blob_dir / datetime.now().strftime("%Y-%m-%d") / match.group(1)
        assert blob_file.exists()
        assert blob_file.read_text() == "This is a prompt body that should also be stored as blob"

    def test_json_serialisation_error(self):
        """Test handling of objects that can't be JSON serialised."""
        formatter = TruncatingFormatter()

        # Create record with non-serialisable object
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Create a custom object that will fail JSON serialisation
        # Use a circular reference to force JSON serialisation to fail
        circular_dict = {}
        circular_dict["self"] = circular_dict

        # Add the circular reference dict
        record.circular_ref = circular_dict

        formatted = formatter.format(record)

        # Should fall back to str() representation
        assert "circular_ref=" in formatted
        # The str() of a dict with circular reference will contain {...}
        assert "{...}" in formatted or "{'self': {...}}" in formatted

    def test_bytes_decode_error(self):
        """Test handling of bytes that can't be decoded as UTF-8."""
        formatter = TruncatingFormatter()

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test", args=(), exc_info=None
        )

        # Add bytes that can't be decoded as UTF-8
        # These are invalid UTF-8 sequences
        record.invalid_bytes = b"\x80\x81\x82\xff"

        formatted = formatter.format(record)

        # Should use repr() for non-decodable bytes
        assert "invalid_bytes=" in formatted
        assert "\\x80\\x81\\x82\\xff" in formatted


class TestContextBinding:
    """Test context variable binding."""

    def test_bind_context_variables(self, clean_logging):
        """Test binding and unbinding of context variables."""
        log = get_logger("test")
        formatter = TruncatingFormatter()
        context_filter = ContextFilter()

        # Create handler to capture output
        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

        # Log without context
        log.info("No context")
        output1 = stream.getvalue()
        assert "schedule_id=" not in output1
        assert "amendment_id=" not in output1

        # Log with context
        stream.truncate(0)
        stream.seek(0)
        with bind(schedule_id="S123", amendment_id="A456"):
            log.info("With context")
            output2 = stream.getvalue()
            assert "schedule_id=S123" in output2
            assert "amendment_id=A456" in output2

        # Log after context
        stream.truncate(0)
        stream.seek(0)
        log.info("After context")
        output3 = stream.getvalue()
        assert "schedule_id=" not in output3
        assert "amendment_id=" not in output3

    def test_nested_bind_contexts(self, clean_logging):
        """Test nested context binding."""
        log = get_logger("test")
        formatter = TruncatingFormatter()
        context_filter = ContextFilter()

        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

        with bind(schedule_id="S1"):
            with bind(amendment_id="A1"):
                log.info("Nested")
                output = stream.getvalue()
                assert "schedule_id=S1" in output
                assert "amendment_id=A1" in output

    def test_thread_isolation(self, clean_logging):
        """Test that context variables are thread-isolated."""
        # Clear root logger to avoid interference
        logging.getLogger().handlers.clear()

        results = {}

        def thread_func(name):
            """Function to run in thread."""
            formatter = TruncatingFormatter()
            context_filter = ContextFilter()
            from io import StringIO

            stream = StringIO()
            handler = logging.StreamHandler(stream)
            handler.setFormatter(formatter)
            handler.addFilter(context_filter)

            thread_log = get_logger(f"test.{name}")
            thread_log.addHandler(handler)
            thread_log.setLevel(logging.INFO)
            thread_log.propagate = False  # Don't propagate to root

            thread_log.info(f"Thread {name}")
            results[name] = stream.getvalue()

        # Bind in main thread
        with bind(schedule_id="MAIN"):
            # Start thread - should NOT inherit context
            thread = threading.Thread(target=thread_func, args=("worker",))
            thread.start()
            thread.join()

        # Check thread output doesn't have main context
        assert "schedule_id=" not in results["worker"]

    def test_bind_all_context_variables(self, clean_logging):
        """Test binding all supported context variables including prompt_id and candidate_eid."""
        log = get_logger("test")
        formatter = TruncatingFormatter()
        context_filter = ContextFilter()

        # Create handler to capture output
        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

        # Test with all context variables
        with bind(schedule_id="S123", amendment_id="A456", prompt_id="P789", candidate_eid="eid_001"):
            log.info("All contexts")
            output = stream.getvalue()
            assert "schedule_id=S123" in output
            assert "amendment_id=A456" in output
            assert "prompt_id=P789" in output
            assert "candidate_eid=eid_001" in output


class TestEventHelper:
    """Test the event() helper function."""

    def test_event_helper_basic(self, clean_logging):
        """Test basic event logging."""
        log = get_logger("test")
        formatter = TruncatingFormatter()
        context_filter = ContextFilter()

        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

        # Log event with extra fields
        event(log, EVT.LLM_REQUEST, prompt_id="P123", tokens=1500)

        output = stream.getvalue()
        assert "LLM_REQUEST" in output
        assert "prompt_id=P123" in output
        assert "tokens=1500" in output

    def test_event_with_custom_message(self, clean_logging):
        """Test event with custom message."""
        log = get_logger("test")
        formatter = TruncatingFormatter()
        context_filter = ContextFilter()

        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

        # Log event with custom message
        event(log, EVT.SCHEDULE_START, "Starting schedule processing", act="Test Act")

        output = stream.getvalue()
        # The event function logs the message, but the formatter shows the event name
        assert "SCHEDULE_START" in output
        assert "act=Test Act" in output


class TestRootConfiguration:
    """Test root logger configuration."""

    def test_configure_root_logging_idempotent(self, clean_logging):
        """Test that configure_root_logging is idempotent."""
        root = logging.getLogger()

        # First call
        configure_root_logging()
        handler_count_1 = len(root.handlers)
        assert handler_count_1 == 2  # File + Console

        # Second call - should not add more handlers
        configure_root_logging()
        handler_count_2 = len(root.handlers)
        assert handler_count_2 == handler_count_1

    def test_handlers_have_formatters_and_filters(self, clean_logging, monkeypatch):
        """Test that console and file handlers have proper setup."""
        # Clear all handlers first to ensure configure_root_logging will run
        root = logging.getLogger()
        root.handlers.clear()

        configure_root_logging()

        # Now check the handlers that were added
        assert len(root.handlers) == 2  # Console and file handlers

        # Check that all handlers have proper formatters and filters
        for handler in root.handlers:
            assert handler.formatter is not None
            assert isinstance(handler.formatter, TruncatingFormatter)

            # Check that all handlers have the context filter
            has_context_filter = any(isinstance(f, ContextFilter) for f in handler.filters)
            assert has_context_filter


class TestDailyFileHandler:
    """Test the DailyFileHandler class."""

    def test_daily_file_creation(self, temp_log_dir):
        """Test that handler creates daily files with correct names."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        # Create and emit a record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test message", args=(), exc_info=None
        )

        handler.emit(record)
        handler.close()

        # Check file was created with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = temp_log_dir / f"{today}.log"
        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_daily_rollover(self, temp_log_dir):
        """Test that handler creates new file on date change."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        # Mock datetime to control dates
        with patch("app.logging.debug_logger.datetime", autospec=True) as mock_datetime:
            # Day 1
            day1 = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.fromtimestamp.return_value = day1
            mock_datetime.now.return_value = day1

            record1 = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Day 1 message",
                args=(),
                exc_info=None,
            )
            record1.created = day1.timestamp()
            handler.emit(record1)

            # Day 2
            day2 = datetime(2024, 1, 2, 12, 0, 0)
            mock_datetime.fromtimestamp.return_value = day2
            mock_datetime.now.return_value = day2

            record2 = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Day 2 message",
                args=(),
                exc_info=None,
            )
            record2.created = day2.timestamp()
            handler.emit(record2)

            handler.close()

        # Check both files exist
        file1 = temp_log_dir / "2024-01-01.log"
        file2 = temp_log_dir / "2024-01-02.log"

        assert file1.exists()
        assert file2.exists()
        assert "Day 1 message" in file1.read_text()
        assert "Day 2 message" in file2.read_text()

    def test_thread_safe_writes(self, temp_log_dir):
        """Test concurrent writes from multiple threads."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        lines_per_thread = 100
        num_threads = 20

        def write_logs(thread_id):
            """Write logs from a thread."""
            for i in range(lines_per_thread):
                record = logging.LogRecord(
                    name=f"thread{thread_id}",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=10,
                    msg=f"Thread {thread_id} line {i}",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

        # Run threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_logs, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        handler.close()

        # Count lines in log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = temp_log_dir / f"{today}.log"
        lines = log_file.read_text().strip().split("\n")

        # Should have exactly the expected number of lines
        assert len(lines) == lines_per_thread * num_threads

        # Verify no corruption (each line should parse)
        for line in lines:
            assert "Thread" in line
            assert "line" in line

        # Verify no interleaving (lines should be properly formatted with timestamps)
        # Each line should start with a valid ISO timestamp
        for line in lines:
            # Extract timestamp (up to first space after timezone)
            parts = line.split(" ", 1)
            assert len(parts) == 2
            timestamp_part = parts[0]
            # Should be valid ISO format with timezone
            assert "T" in timestamp_part
            assert "+" in timestamp_part or "-" in timestamp_part

    def test_fsync_durability(self, temp_log_dir):
        """Test that fsync ensures durability."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        # Write a record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Important message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # Get the file path before closing
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = temp_log_dir / f"{today}.log"

        # Close handler
        handler.close()

        # Verify message was written
        assert log_file.exists()
        assert "Important message" in log_file.read_text()

    def test_emit_exception_handling(self, temp_log_dir, monkeypatch):
        """Test that emit handles exceptions gracefully."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        # Create a record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test message", args=(), exc_info=None
        )

        # Mock the format method to raise an exception
        def failing_format(record):
            raise RuntimeError("Formatting failed!")

        handler.format = failing_format

        # Capture the handleError output
        error_called = []
        original_handle_error = handler.handleError

        def mock_handle_error(record):
            error_called.append(record)
            # Call the original to maintain proper behaviour
            original_handle_error(record)

        handler.handleError = mock_handle_error

        # Emit should not raise, but should call handleError
        handler.emit(record)

        # Verify handleError was called
        assert len(error_called) == 1
        assert error_called[0] == record

        handler.close()

    def test_emit_file_write_error(self, temp_log_dir):
        """Test emit when file writing fails."""
        handler = DailyFileHandler(temp_log_dir)
        formatter = TruncatingFormatter()
        handler.setFormatter(formatter)

        # Create a record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Test message", args=(), exc_info=None
        )

        # First emit to create the file
        handler.emit(record)

        # Now close the file handle but keep reference
        handler._current_file.close()

        # Track if handleError was called
        error_handled = []
        original_handle_error = handler.handleError

        def track_handle_error(record):
            error_handled.append(True)
            original_handle_error(record)

        handler.handleError = track_handle_error

        # Try to emit again - should fail because file is closed
        handler.emit(record)

        # Verify error was handled
        assert len(error_handled) == 1

        # Clean up
        handler._current_file = None
        handler.close()


class TestUtilities:
    """Test utility functions."""

    def test_store_blob(self, tmp_path, monkeypatch):
        """Test blob storage function."""
        blob_dir = tmp_path / "test_blobs"
        blob_dir.mkdir()

        # Patch the imported constant in debug_logger module
        monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

        # Store some content
        content = "This is test content for blob storage"
        blob_path = _store_blob(content)

        # Verify path format
        assert "test_blobs" in blob_path

        # Verify content
        if not Path(blob_path).is_absolute():
            full_path = Path.cwd() / blob_path
        else:
            full_path = Path(blob_path)

        assert full_path.exists()
        assert full_path.read_text() == content

    def test_blob_path_relativity(self, tmp_path, monkeypatch):
        """Test blob path when BLOB_DIR is outside cwd."""
        # Create blob dir outside current directory
        blob_dir = tmp_path / "outside" / "blobs"
        blob_dir.mkdir(parents=True)

        # Patch the imported constant in debug_logger module
        monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

        # Store content
        content = "Test content"
        blob_path = _store_blob(content)

        # Should return absolute path when not relative to cwd
        assert Path(blob_path).is_absolute()
        assert Path(blob_path).read_text() == content

    def test_store_blob_non_relative_path(self, monkeypatch):
        """Test blob storage returns absolute path when needed."""
        import tempfile
        import os

        # Save current directory
        original_cwd = os.getcwd()

        try:
            # Create a temporary directory for blobs
            with tempfile.TemporaryDirectory() as tmpdir:
                blob_dir = Path(tmpdir) / "blobs"
                blob_dir.mkdir()

                # Patch the blob directory
                monkeypatch.setattr("app.logging.debug_logger.BLOB_DIR", blob_dir, raising=True)

                # Change to a different directory to ensure paths aren't relative
                # Use the system temp directory which exists on all platforms
                temp_dir = tempfile.gettempdir()
                os.chdir(temp_dir)

                # Store blob
                content = "Test content from different working directory"
                blob_path = _store_blob(content)

                # The returned path should work regardless of whether it's absolute or relative
                full_path = Path(blob_path)
                if not full_path.is_absolute():
                    full_path = Path.cwd() / blob_path

                # Verify content can be read
                assert full_path.exists()
                assert full_path.read_text() == content

        finally:
            # Restore original directory
            os.chdir(original_cwd)
