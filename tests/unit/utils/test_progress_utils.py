# tests/unit/test_progress_utils.py

from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from app.utils.progress_utils import (
    get_progress_message,
    _read_file_backwards,
    _get_latest_event_for_schedule,
    _extract_amendment_details,
    _extract_event_data,
    _generate_message_for_event,
    STAGE_MESSAGES,
)


class TestGetProgressMessage:
    """Test cases for the main get_progress_message function."""

    @patch("app.utils.progress_utils._get_latest_event_for_schedule")
    @patch("app.utils.progress_utils._generate_message_for_event")
    def test_get_progress_message_with_event(self, mock_generate, mock_get_latest):
        """Test progress message when an event is found."""
        # Mock latest event
        mock_event = {"event": "CANDIDATE_IDENTIFIED", "candidate_eid": "sec_40"}
        mock_get_latest.return_value = mock_event
        mock_generate.return_value = "Identifying amendments in section 40..."

        result = get_progress_message("test-schedule-123")

        assert result == "Identifying amendments in section 40..."
        mock_get_latest.assert_called_once_with("test-schedule-123")
        mock_generate.assert_called_once_with(mock_event)

    @patch("app.utils.progress_utils._get_latest_event_for_schedule")
    def test_get_progress_message_no_event(self, mock_get_latest):
        """Test progress message when no event is found."""
        mock_get_latest.return_value = None

        result = get_progress_message("test-schedule-123")

        assert result == STAGE_MESSAGES["STARTING"]
        mock_get_latest.assert_called_once_with("test-schedule-123")

    @patch("app.utils.progress_utils._get_latest_event_for_schedule")
    def test_get_progress_message_exception(self, mock_get_latest):
        """Test fallback message when an exception occurs."""
        mock_get_latest.side_effect = Exception("Test error")

        result = get_progress_message("test-schedule-123")

        assert result == "Processing schedule updates..."


class TestReadFileBackwards:
    """Test cases for _read_file_backwards function."""

    def test_read_file_backwards_simple(self):
        """Test reading a simple file backwards."""
        content = b"line1\nline2\nline3"
        mock_file = mock_open(read_data=content)()
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=len(content))
        mock_file.read = MagicMock(return_value=content)

        with patch("builtins.open", return_value=mock_file):
            lines = list(_read_file_backwards(Path("test.log")))

        # Should get lines in reverse order
        assert lines == ["line3", "line2", "line1"]

    def test_read_file_backwards_empty(self):
        """Test reading an empty file."""
        mock_file = mock_open(read_data=b"")()
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=0)

        with patch("builtins.open", return_value=mock_file):
            lines = list(_read_file_backwards(Path("test.log")))

        assert lines == []


class TestGetLatestEventForSchedule:
    """Test cases for _get_latest_event_for_schedule function."""

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_simple(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test getting the latest event for a schedule."""
        # Mock datetime for consistent log filename
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Mock log lines
        log_lines = [
            "2025-07-01T10:00:00.000+00:00 INFO SCHEDULE_END schedule_id=test-123",
            "2025-07-01T09:00:00.000+00:00 INFO CANDIDATE_IDENTIFIED schedule_id=test-123 candidate_eid=sec_40",
            "2025-07-01T08:00:00.000+00:00 INFO SCHEDULE_START schedule_id=test-123",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "SCHEDULE_END"
        assert result["timestamp"] == "2025-07-01T10:00:00.000+00:00"
        assert result["schedule_id"] == "test-123"

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    def test_get_latest_event_no_log_file(self, mock_exists, mock_datetime):
        """Test when log file doesn't exist."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = False

        result = _get_latest_event_for_schedule("test-123")

        assert result is None

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_with_amendment_details(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test getting event with amendment details for LLM_RESPONSE."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Mock log lines including amendment details
        log_lines = [
            "2025-07-01T10:00:00.000+00:00 INFO LLM_RESPONSE schedule_id=test-123 "
            "prompt_name=ApplyInsertionAmendment amendment_id=amend-1",
            "2025-07-01T09:00:00.000+00:00 INFO AMENDMENT_IDENTIFIED schedule_id=test-123 "
            "amendment_id=amend-1 source_eid=sec_40 affected_provision=sec_50 amendment_type=insertion",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "LLM_RESPONSE"
        assert result["phase"] == "application"
        assert result["amendment_id"] == "amend-1"
        # Should have amendment details merged in
        assert result["source_eid"] == "sec_40"
        assert result["affected_provision"] == "sec_50"

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_skip_different_schedule_id(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test that lines with different schedule_id are skipped."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Mock log lines with different schedule IDs
        log_lines = [
            "2025-07-01T10:00:00.000+00:00 INFO SCHEDULE_END schedule_id=other-schedule",
            "2025-07-01T09:00:00.000+00:00 INFO CANDIDATE_IDENTIFIED schedule_id=test-123 candidate_eid=sec_40",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "CANDIDATE_IDENTIFIED"
        assert result["schedule_id"] == "test-123"

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_skip_non_event_lines(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test that lines without relevant events are skipped."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Mock log lines with non-event and already found latest event
        log_lines = [
            "2025-07-01T10:00:00.000+00:00 INFO SCHEDULE_END schedule_id=test-123",
            "2025-07-01T09:00:00.000+00:00 INFO SOME_OTHER_EVENT schedule_id=test-123",
            "2025-07-01T08:00:00.000+00:00 INFO SCHEDULE_START schedule_id=test-123",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "SCHEDULE_END"

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_skip_lines_without_timestamp(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test that lines without valid timestamp are skipped."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Mock log lines with invalid timestamp format
        log_lines = [
            "Invalid timestamp format INFO SCHEDULE_END schedule_id=test-123",
            "2025-07-01T09:00:00.000+00:00 INFO CANDIDATE_IDENTIFIED schedule_id=test-123 candidate_eid=sec_40",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "CANDIDATE_IDENTIFIED"

    @patch("app.utils.progress_utils.datetime")
    @patch("app.utils.progress_utils.Path.exists")
    @patch("app.utils.progress_utils._read_file_backwards")
    def test_get_latest_event_skip_non_event_with_latest_event(self, mock_read_backwards, mock_exists, mock_datetime):
        """Test that non-event lines are skipped when latest_event is already set."""
        mock_datetime.now.return_value.strftime.return_value = "2025-07-01"
        mock_exists.return_value = True

        # Reading backwards, find LLM_RESPONSE first, then need to skip non-event lines
        # while searching for AMENDMENT_IDENTIFIED details
        log_lines = [
            "2025-07-01T10:00:00.000+00:00 INFO LLM_RESPONSE schedule_id=test-123 "
            "prompt_name=ApplyInsertionAmendment amendment_id=amend-1",
            "2025-07-01T09:30:00.000+00:00 DEBUG Random debug message schedule_id=test-123",  # This should be skipped
            "2025-07-01T09:00:00.000+00:00 INFO AMENDMENT_IDENTIFIED schedule_id=test-123 "
            "amendment_id=amend-1 source_eid=sec_40 affected_provision=sec_50 amendment_type=insertion",
        ]
        mock_read_backwards.return_value = log_lines

        result = _get_latest_event_for_schedule("test-123")

        assert result is not None
        assert result["event"] == "LLM_RESPONSE"


class TestExtractAmendmentDetails:
    """Test cases for _extract_amendment_details function."""

    @patch("app.utils.progress_utils.eid_to_source")
    def test_extract_amendment_details_complete(self, mock_eid_to_source):
        """Test extracting complete amendment details."""
        mock_eid_to_source.return_value = "s. 40"

        line = (
            "2025-07-01T10:00:00.000+00:00 INFO AMENDMENT_IDENTIFIED amendment_id=amend-1 "
            "source_eid=sec_40 affected_provision=sec_50 location=AFTER amendment_type=insertion"
        )
        amendment_details = {}

        _extract_amendment_details(line, amendment_details)

        assert "amend-1" in amendment_details
        details = amendment_details["amend-1"]
        assert details["source_eid"] == "sec_40"
        assert details["affected_provision"] == "sec_50"
        assert details["location"] == "AFTER"
        assert details["amendment_type"] == "insertion"
        assert details["source"] == "s. 40"

    def test_extract_amendment_details_no_id(self):
        """Test when amendment_id is missing."""
        line = "2025-07-01T10:00:00.000+00:00 INFO AMENDMENT_IDENTIFIED source_eid=sec_40"
        amendment_details = {}

        _extract_amendment_details(line, amendment_details)

        assert len(amendment_details) == 0


class TestExtractEventData:
    """Test cases for _extract_event_data function."""

    def test_extract_candidate_identified(self):
        """Test extracting CANDIDATE_IDENTIFIED event data."""
        line = "2025-07-01T10:00:00.000+00:00 INFO CANDIDATE_IDENTIFIED candidate_eid=sec_40"
        event_data = {}
        amendment_details = {}

        _extract_event_data("CANDIDATE_IDENTIFIED", line, event_data, amendment_details)

        assert event_data["candidate_eid"] == "sec_40"

    def test_extract_llm_response_identification(self):
        """Test extracting LLM_RESPONSE for identification phase."""
        line = "2025-07-01T10:00:00.000+00:00 INFO LLM_RESPONSE prompt_name=TableOfAmendments candidate_eid=sec_40"
        event_data = {}
        amendment_details = {}

        _extract_event_data("LLM_RESPONSE", line, event_data, amendment_details)

        assert event_data["phase"] == "identification"
        assert event_data["candidate_eid"] == "sec_40"

    def test_extract_llm_response_application(self):
        """Test extracting LLM_RESPONSE for application phase."""
        line = (
            "2025-07-01T10:00:00.000+00:00 INFO LLM_RESPONSE "
            "prompt_name=ApplyInsertionAmendment amendment_id=amend-1"
        )
        event_data = {}
        amendment_details = {}

        _extract_event_data("LLM_RESPONSE", line, event_data, amendment_details)

        assert event_data["phase"] == "application"
        assert event_data["amendment_id"] == "amend-1"

    @patch("app.utils.progress_utils.eid_to_source")
    def test_extract_amendment_applied(self, mock_eid_to_source):
        """Test extracting AMENDMENT_APPLIED event data."""
        mock_eid_to_source.return_value = "s. 50"

        line = "2025-07-01T10:00:00.000+00:00 INFO AMENDMENT_APPLIED affected_provision=sec_50"
        event_data = {}
        amendment_details = {}

        _extract_event_data("AMENDMENT_APPLIED", line, event_data, amendment_details)

        assert event_data["source"] == "s. 50"

    def test_extract_amendment_applying(self):
        """Test extracting AMENDMENT_APPLYING event data."""
        line = "2025-07-01T10:00:00.000+00:00 INFO AMENDMENT_APPLYING affected_provision=sec_50"
        event_data = {}
        amendment_details = {}

        _extract_event_data("AMENDMENT_APPLYING", line, event_data, amendment_details)

        assert event_data["affected_provision"] == "sec_50"


class TestGenerateMessageForEvent:
    """Test cases for _generate_message_for_event function."""

    def test_generate_message_schedule_start(self):
        """Test message for SCHEDULE_START event."""
        event_data = {"event": "SCHEDULE_START"}
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["STARTING"]

    def test_generate_message_candidate_found(self):
        """Test message for CANDIDATE_FOUND event."""
        event_data = {"event": "CANDIDATE_FOUND"}
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["CANDIDATES"]

    @patch("app.utils.progress_utils.eid_to_source")
    def test_generate_message_candidate_identified(self, mock_eid_to_source):
        """Test message for CANDIDATE_IDENTIFIED event."""
        mock_eid_to_source.return_value = "s. 40"

        event_data = {"event": "CANDIDATE_IDENTIFIED", "candidate_eid": "sec_40"}
        result = _generate_message_for_event(event_data)
        assert result == "Identifying amendments in s. 40..."

    def test_generate_message_candidate_identified_no_eid(self):
        """Test message for CANDIDATE_IDENTIFIED without eid."""
        event_data = {"event": "CANDIDATE_IDENTIFIED"}
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["IDENTIFICATION"]

    @patch("app.utils.progress_utils.eid_to_source")
    def test_generate_message_llm_response_identification(self, mock_eid_to_source):
        """Test message for LLM_RESPONSE in identification phase."""
        mock_eid_to_source.return_value = "s. 40"

        event_data = {"event": "LLM_RESPONSE", "phase": "identification", "candidate_eid": "sec_40"}
        result = _generate_message_for_event(event_data)
        assert result == "Identifying amendments in s. 40..."

    @patch("app.utils.progress_utils.eid_to_source")
    def test_generate_message_llm_response_application(self, mock_eid_to_source):
        """Test message for LLM_RESPONSE in application phase."""
        mock_eid_to_source.return_value = "s. 50"

        event_data = {
            "event": "LLM_RESPONSE",
            "phase": "application",
            "affected_provision": "sec_50",
            "amendment_type": "INSERTION",
        }
        result = _generate_message_for_event(event_data)
        assert result == "Applied insertion to s. 50..."

    @patch("app.utils.progress_utils.eid_to_source")
    def test_generate_message_amendment_applying(self, mock_eid_to_source):
        """Test message for AMENDMENT_APPLYING event."""
        mock_eid_to_source.return_value = "s. 50"

        event_data = {"event": "AMENDMENT_APPLYING", "affected_provision": "sec_50"}
        result = _generate_message_for_event(event_data)
        assert result == "Merging amendment into s. 50..."

    def test_generate_message_amendment_applied(self):
        """Test message for AMENDMENT_APPLIED event."""
        event_data = {"event": "AMENDMENT_APPLIED", "source": "s. 50"}
        result = _generate_message_for_event(event_data)
        assert result == "Merged amendment into s. 50..."

    def test_generate_message_schedule_end(self):
        """Test message for SCHEDULE_END event."""
        event_data = {"event": "SCHEDULE_END"}
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["FINALISING"]

    def test_generate_message_unknown_event(self):
        """Test fallback message for unknown event."""
        event_data = {"event": "UNKNOWN_EVENT"}
        result = _generate_message_for_event(event_data)
        assert result == "Processing schedule updates..."

    def test_generate_message_llm_response_identification_no_eid(self):
        """Test LLM_RESPONSE identification phase without candidate_eid."""
        event_data = {
            "event": "LLM_RESPONSE",
            "phase": "identification",
            # No candidate_eid
        }
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["IDENTIFICATION"]

    def test_generate_message_llm_response_application_no_provision(self):
        """Test LLM_RESPONSE application phase without affected_provision."""
        event_data = {
            "event": "LLM_RESPONSE",
            "phase": "application",
            # No affected_provision
        }
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["APPLICATION"]

    def test_generate_message_amendment_applying_no_provision(self):
        """Test AMENDMENT_APPLYING without affected_provision."""
        event_data = {
            "event": "AMENDMENT_APPLYING"
            # No affected_provision
        }
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["APPLICATION"]

    def test_generate_message_amendment_applied_no_source(self):
        """Test AMENDMENT_APPLIED without source."""
        event_data = {
            "event": "AMENDMENT_APPLIED"
            # No source field
        }
        result = _generate_message_for_event(event_data)
        assert result == STAGE_MESSAGES["APPLICATION"]
