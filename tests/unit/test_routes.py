# tests/unit/test_routes.py

import pytest
from unittest.mock import patch, MagicMock, Mock
import sys
from app.models.amendments import Amendment, AmendmentType, AmendmentLocation

# This ensures Python uses the standard library logging module
import logging

sys.modules["app.logging"] = logging


@pytest.fixture
def sample_xml():
    """Fixture providing sample XML content for tests."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
        <act>
            <body>
                <section eId="sec_1">
                    <num>1</num>
                    <heading>Test Section</heading>
                </section>
            </body>
        </act>
    </akomaNtoso>"""


def test_home_route(client):
    """Test that the home route returns the expected response."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Keeling Schedule Flask App is running!" in response.data


def test_health_check(client):
    """Test that health check returns 200 and valid JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


@patch("tempfile.NamedTemporaryFile")
@patch("os.unlink")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.routes._get_keeling_service")
@patch("lxml.etree.fromstring", return_value=MagicMock())
def test_keeling_schedule_success(
    mock_fromstring, mock_get_service, mock_open, mock_unlink, mock_temp_file, client, sample_xml
):
    """Test successful generation of Keeling schedule."""

    # Mock temporary file handling
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "test.xml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Ensure file read operation works
    mock_file = MagicMock()
    mock_file.read.return_value = sample_xml
    mock_open.return_value.__enter__.return_value = mock_file

    # Create mock amendment
    mock_amendment = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    # Setup mock service
    mock_keeling_service = MagicMock()
    mock_keeling_service.process_amending_bill.return_value = [mock_amendment]
    mock_keeling_service.apply_amendments.return_value = True
    mock_get_service.return_value = mock_keeling_service

    # Make request
    payload = {
        "bill_xml": sample_xml,
        "act_name": "Test Act 2020",
        "act_xml": sample_xml,
        "include_table_of_amendments": True,
    }
    response = client.post("/api/keeling-schedule/generate", json=payload)

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert "amended_act" in response.json
    assert "table_of_amendments" in response.json
    amendments = response.json["table_of_amendments"]
    assert len(amendments) == 1
    assert amendments[0]["source"] == "Test Source"


def test_keeling_schedule_missing_fields(client):
    """Test missing fields handling."""
    payload = {"act_name": "Test Act 2020"}
    response = client.post("/api/keeling-schedule/generate", json=payload)
    assert response.status_code == 400
    assert "Missing required fields" in response.json["message"]


@patch("lxml.etree.fromstring")
def test_keeling_schedule_invalid_xml(mock_fromstring, client):
    """Test invalid XML handling."""

    # Mock XML parsing to raise a specific etree.ParseError
    from lxml.etree import ParseError

    mock_fromstring.side_effect = ParseError("XML parse error", 1, 0, 0)

    payload = {
        "bill_xml": "<invalid>xml<invalid>",
        "act_name": "Test Act 2020",
        "act_xml": "<invalid>xml<invalid>",
        "include_table_of_amendments": True,
    }
    response = client.post("/api/keeling-schedule/generate", json=payload)
    assert response.status_code == 400
    assert "Invalid XML provided" in response.json["message"]


@patch("tempfile.NamedTemporaryFile")
@patch("os.unlink")
@patch("os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.routes._get_keeling_service")
@patch("lxml.etree.fromstring", return_value=MagicMock())
def test_keeling_schedule_cleanup_error(
    mock_fromstring, mock_get_service, mock_open, mock_exists, mock_unlink, mock_temp_file, client, sample_xml
):
    """Test that cleanup errors are handled gracefully."""

    # Mock temporary file handling
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "test.xml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Ensure file read operation works
    mock_file = MagicMock()
    mock_file.read.return_value = sample_xml
    mock_open.return_value.__enter__.return_value = mock_file

    # Mock file existence check to return True
    mock_exists.return_value = True

    # Mock unlink to raise an exception
    mock_unlink.side_effect = Exception("Failed to delete file")

    # Create mock amendment and service
    mock_amendment = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    # Setup the mock service properly
    mock_keeling_service = MagicMock()
    mock_keeling_service.process_amending_bill.return_value = [mock_amendment]
    mock_keeling_service.apply_amendments.return_value = True
    mock_get_service.return_value = mock_keeling_service

    # Make request
    payload = {
        "bill_xml": sample_xml,
        "act_name": "Test Act 2020",
        "act_xml": sample_xml,
        "include_table_of_amendments": True,
    }
    response = client.post("/api/keeling-schedule/generate", json=payload)

    # The request should still succeed even if cleanup fails
    assert response.status_code == 200
    assert response.json["status"] == "success"

    # Verify cleanup was attempted
    mock_unlink.assert_called()
    mock_exists.assert_called()


def test_handle_proxy_headers(client):
    """Test that proxy headers are properly handled for HTTPS."""
    # Test with HTTPS header
    response = client.get("/", headers={"X-Forwarded-Proto": "https"})
    assert response.status_code == 200
    # Test without HTTPS header (HTTP)
    response = client.get("/")
    assert response.status_code == 200


@patch("app.routes.get_kernel")
def test_get_keeling_service_direct(mock_get_kernel):
    """
    Test coverage for _get_keeling_service() by calling it directly.
    """
    from app.routes import _get_keeling_service
    from app.services.keeling_service import KeelingService

    # Mock the kernel
    mock_kernel = Mock()
    mock_get_kernel.return_value = mock_kernel

    # Mock KeelingService to avoid its actual initialisation
    with patch("app.routes.KeelingService") as mock_keeling_class:
        mock_keeling_instance = Mock(spec=KeelingService)
        mock_keeling_class.return_value = mock_keeling_instance

        service = _get_keeling_service()

        assert service == mock_keeling_instance
        mock_keeling_class.assert_called_once_with(mock_kernel)


@patch("tempfile.NamedTemporaryFile")
@patch("os.path.exists")
@patch("os.unlink")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.routes._get_keeling_service")
@patch("lxml.etree.fromstring", return_value=MagicMock())
def test_generate_keeling_schedule_exception(
    mock_fromstring, mock_get_service, mock_open, mock_unlink, mock_exists, mock_temp_file, client, sample_xml
):
    """
    Force an exception in generate_keeling_schedule.
    """

    # Mock temporary file handling
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "test.xml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Setup mock to ensure files appear to exist for cleanup
    mock_exists.return_value = True

    # Ensure file read operation works
    mock_file = MagicMock()
    mock_file.read.return_value = sample_xml
    mock_open.return_value.__enter__.return_value = mock_file

    # Setup the mock service to raise the specific error
    mock_keeling_service = MagicMock()
    mock_keeling_service.process_amending_bill.side_effect = RuntimeError("Test error")
    mock_get_service.return_value = mock_keeling_service

    # Make request
    payload = {
        "bill_xml": sample_xml,
        "act_name": "Test Act 2020",
        "act_xml": sample_xml,
        "include_table_of_amendments": True,
    }
    response = client.post("/api/keeling-schedule/generate", json=payload)

    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "failure"
    # The act_xml is returned unmodified
    assert data["amended_act"] == sample_xml
    assert data["table_of_amendments"] == []
    assert "Test error" in data["message"]


def test_keeling_schedule_empty_json_data(client):
    """Test handling when empty JSON object is provided."""
    # Send an empty JSON object - valid JSON but no data
    response = client.post("/api/keeling-schedule/generate", json={})

    assert response.status_code == 400
    assert response.json["status"] == "failure"
    assert response.json["message"] == "No JSON data provided"


@patch("app.routes.get_progress_message")
def test_get_progress_success(mock_get_progress_message, client):
    """Test successful retrieval of progress information."""
    # Mock the progress message
    mock_get_progress_message.return_value = "Identifying amendments in section 40(2)(b)..."

    # Make request
    schedule_id = "test-schedule-123"
    response = client.get(f"/api/keeling-schedule/progress/{schedule_id}")

    # Verify response
    assert response.status_code == 200
    assert response.json["message"] == "Identifying amendments in section 40(2)(b)..."

    # Verify the progress message function was called with correct ID
    mock_get_progress_message.assert_called_once_with(schedule_id)


@patch("app.routes.datetime")
@patch("app.routes.get_progress_message")
def test_get_progress_exception(mock_get_progress_message, mock_datetime, client):
    """Test fallback response when get_progress_message raises an exception."""
    # Mock the progress message to raise an exception
    mock_get_progress_message.side_effect = Exception("Test error")

    # Mock datetime to return a fixed time
    mock_utcnow = MagicMock()
    mock_utcnow.strftime.return_value = "2025-07-01 12:00:00"
    mock_datetime.utcnow.return_value = mock_utcnow

    # Make request
    schedule_id = "test-schedule-123"
    response = client.get(f"/api/keeling-schedule/progress/{schedule_id}")

    # Verify fallback response
    assert response.status_code == 200
    assert response.json["message"] == "Processing... (Last checked: 2025-07-01 12:00:00)"

    # Verify the progress message function was called
    mock_get_progress_message.assert_called_once_with(schedule_id)


@patch("tempfile.NamedTemporaryFile")
@patch("os.unlink")
@patch("builtins.open", new_callable=MagicMock)
@patch("app.routes._get_keeling_service")
@patch("lxml.etree.fromstring", return_value=MagicMock())
def test_keeling_schedule_success_prevent_include_table_of_amendments(
    mock_fromstring, mock_get_service, mock_open, mock_unlink, mock_temp_file, client, sample_xml
):
    """Test successful generation of Keeling schedule."""

    # Mock temporary file handling
    mock_temp_file_instance = MagicMock()
    mock_temp_file_instance.name = "test.xml"
    mock_temp_file.return_value = mock_temp_file_instance

    # Ensure file read operation works
    mock_file = MagicMock()
    mock_file.read.return_value = sample_xml
    mock_open.return_value.__enter__.return_value = mock_file

    # Create mock amendment
    mock_amendment = Amendment(
        source="Test Source",
        source_eid="sec_1",
        affected_document="Test Act",
        affected_provision="sec_2",
        location=AmendmentLocation.AFTER,
        amendment_type=AmendmentType.INSERTION,
        whole_provision=True,
    )

    # Setup mock service
    mock_keeling_service = MagicMock()
    mock_keeling_service.process_amending_bill.return_value = [mock_amendment]
    mock_keeling_service.apply_amendments.return_value = True
    mock_get_service.return_value = mock_keeling_service

    # Make request
    payload = {
        "bill_xml": sample_xml,
        "act_name": "Test Act 2020",
        "act_xml": sample_xml,
        "include_table_of_amendments": False,
    }
    response = client.post("/api/keeling-schedule/generate", json=payload)

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert "amended_act" in response.json
    assert "table_of_amendments" not in response.json
