import pytest
import sqlite3
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from semantic_kernel.functions import FunctionResult, KernelFunctionFromPrompt
from app.benchmarking.metrics_logger import MetricsLogger


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_file = tmp_path / "test_keeling_logs.db"
    return str(db_file)


@pytest.fixture
def metrics_logger(test_db_path):
    """Fixture for creating a default MetricsLogger instance for testing."""
    return MetricsLogger(
        enable_aws_bedrock=True,
        bedrock_service_id="bedrock-claude",
        bedrock_model_id="anthropic.claude-3-sonnet-20240229",
        enable_azure_openai=True,
        azure_model_deployment_name="gpt-4o",
        db_path=test_db_path,
    )


def test_init():
    """Test the initialisation of MetricsLogger with various configurations."""
    # Test with only AWS Bedrock enabled
    logger = MetricsLogger(
        enable_aws_bedrock=True,
        bedrock_service_id="bedrock-claude",
        bedrock_model_id="model-id",
        enable_azure_openai=False,
    )
    assert logger.enable_aws_bedrock is True
    assert logger.bedrock_service_id == "bedrock-claude"
    assert logger.bedrock_model_id == "model-id"
    assert logger.enable_azure_openai is False
    assert logger.azure_model_deployment_name is None

    # Test with only Azure OpenAI enabled
    logger = MetricsLogger(
        enable_aws_bedrock=False,
        enable_azure_openai=True,
        azure_model_deployment_name="gpt-4o",
    )
    assert logger.enable_aws_bedrock is False
    assert logger.bedrock_service_id is None
    assert logger.bedrock_model_id is None
    assert logger.enable_azure_openai is True
    assert logger.azure_model_deployment_name == "gpt-4o"

    # Test with both enabled
    logger = MetricsLogger(
        enable_aws_bedrock=True,
        bedrock_service_id="bedrock-claude",
        bedrock_model_id="model-id",
        enable_azure_openai=True,
        azure_model_deployment_name="gpt-4o",
    )
    assert logger.enable_aws_bedrock is True
    assert logger.bedrock_service_id == "bedrock-claude"
    assert logger.bedrock_model_id == "model-id"
    assert logger.enable_azure_openai is True
    assert logger.azure_model_deployment_name == "gpt-4o"


def test_initialise_database(test_db_path):
    """Test database initialisation creates the expected tables."""
    # Create a logger with a test database path
    # The logger variable is used to initialise the database
    MetricsLogger(db_path=test_db_path)

    # Connect to the database and check if tables exist
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Check if our tables exist
    assert "schedules" in tables
    assert "amendments" in tables
    assert "prompts" in tables

    # Check schema of schedules table
    cursor.execute("PRAGMA table_info(schedules);")
    columns = [row[1] for row in cursor.fetchall()]
    expected_schedule_columns = [
        "schedule_id",
        "act_name",
        "model_id",
        "service_id",
        "start_timestamp",
        "end_timestamp",
        "total_duration_seconds",
        "bill_xml_size",
        "act_xml_size",
        "total_amendments_found",
        "total_amendments_applied",
        "application_rate",
        "total_prompts_executed",
        "total_token_usage",
        "total_cost_usd",
    ]
    for col in expected_schedule_columns:
        assert col in columns

    conn.close()


def test_initialise_database_exception():
    """Test exception handling in _initialise_database method."""
    # Create a logger with a mocked _get_db_connection method that raises an exception
    logger = MetricsLogger()

    with (
        patch.object(logger, "_get_db_connection") as mock_get_conn,
        patch("app.benchmarking.metrics_logger.logger.error") as mock_logger,
    ):
        mock_get_conn.side_effect = Exception("Test database initialisation exception")

        # Call _initialise_database directly to force exception
        logger._initialise_database()

        # Verify that logging.error was called with the expected message
        mock_logger.assert_called_once()
        assert "Failed to initialise database" in mock_logger.call_args[0][0]


def test_calculate_cost():
    """Test the _calculate_cost method with various scenarios."""
    logger = MetricsLogger(
        enable_aws_bedrock=True,
        bedrock_service_id="bedrock-claude",
        enable_azure_openai=True,
        azure_model_deployment_name="gpt-4o",
    )

    # Test with non-numeric inputs
    assert logger._calculate_cost("not-a-number", 100) == 0.0
    assert logger._calculate_cost(100, "not-a-number") == 0.0
    assert logger._calculate_cost(None, 100) == 0.0
    assert logger._calculate_cost(100, None) == 0.0

    # Test with Azure OpenAI (default) rates
    cost = logger._calculate_cost(1000, 500)
    expected_cost = (1000 / 1000) * 0.005 + (500 / 1000) * 0.015
    assert abs(cost - expected_cost) < 0.0001

    # Test with AWS Bedrock rates
    cost = logger._calculate_cost(1000, 500, "bedrock-claude")
    expected_cost = (1000 / 1000) * (3.00 / 1000) + (500 / 1000) * (15.00 / 1000)
    assert abs(cost - expected_cost) < 0.0001


def test_log_schedule_start_and_end(metrics_logger, test_db_path):
    """Test logging the start and end of a schedule."""
    schedule_id = str(uuid.uuid4())

    # Log the start of a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Test Act",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
        bill_xml_size=1000,
        act_xml_size=2000,
    )

    # Verify the schedule was logged in the database
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT act_name, model_id, service_id FROM schedules WHERE schedule_id = ?", (schedule_id,))
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == "Test Act"
    assert result[1] == "test-model"
    assert result[2] == "test-service"

    # Log the end of the schedule
    metrics_logger.log_schedule_end(schedule_id=schedule_id, total_amendments_found=10, total_amendments_applied=8)

    # Verify the schedule was updated
    cursor.execute(
        "SELECT total_amendments_found, total_amendments_applied, application_rate "
        "FROM schedules WHERE schedule_id = ?",
        (schedule_id,),
    )
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == 10
    assert result[1] == 8
    assert result[2] == 80.0  # 8/10 * 100

    conn.close()


def test_log_amendment(metrics_logger, test_db_path):
    """Test logging an amendment."""
    schedule_id = str(uuid.uuid4())
    amendment_id = str(uuid.uuid4())

    # First, create a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Test Act",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    # Log an amendment
    metrics_logger.log_amendment(
        schedule_id=schedule_id,
        amendment_id=amendment_id,
        source="Section 1",
        source_eid="s1",
        affected_provision="Section 2",
        location="AFTER",
        amendment_type="INSERTION",
        whole_provision=True,
        identification_time_seconds=1.5,
    )

    # Verify the amendment was logged
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT source, affected_provision, location, amendment_type, whole_provision "
        "FROM amendments WHERE amendment_id = ?",
        (amendment_id,),
    )
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == "Section 1"
    assert result[1] == "Section 2"
    assert result[2] == "AFTER"
    assert result[3] == "INSERTION"
    assert result[4] == 1  # True

    conn.close()


def test_update_amendment_application(metrics_logger, test_db_path):
    """Test updating an amendment's application status."""
    schedule_id = str(uuid.uuid4())
    amendment_id = str(uuid.uuid4())

    # Create a schedule and an amendment
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Test Act",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    metrics_logger.log_amendment(
        schedule_id=schedule_id,
        amendment_id=amendment_id,
        source="Section 1",
        source_eid="s1",
        affected_provision="Section 2",
        location="AFTER",
        amendment_type="INSERTION",
        whole_provision=True,
        identification_time_seconds=1.5,
    )

    # Update the amendment's application status
    metrics_logger.update_amendment_application(
        amendment_id=amendment_id, application_time_seconds=2.0, application_status=True
    )

    # Verify the amendment was updated
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT application_time_seconds, total_processing_time_seconds, application_status "
        "FROM amendments WHERE amendment_id = ?",
        (amendment_id,),
    )
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == 2.0
    assert result[1] == 3.5  # 1.5 (identification) + 2.0 (application)
    assert result[2] == 1  # True

    conn.close()


@patch("builtins.open", new_callable=MagicMock)
def test_log_prompt(mock_open, metrics_logger, test_db_path):
    """Test logging a prompt execution."""
    schedule_id = str(uuid.uuid4())
    amendment_id = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())

    # Create a schedule and an amendment
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Test Act",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    metrics_logger.log_amendment(
        schedule_id=schedule_id,
        amendment_id=amendment_id,
        source="Section 1",
        source_eid="s1",
        affected_provision="Section 2",
        location="AFTER",
        amendment_type="INSERTION",
        whole_provision=True,
    )

    # Create mock prompt and output
    mock_prompt = MagicMock(spec=KernelFunctionFromPrompt)
    mock_prompt.name = "TestPrompt"

    mock_output = MagicMock(spec=FunctionResult)

    # Set up timestamp values
    start_ts = datetime.utcnow().isoformat()
    end_ts = (datetime.utcnow() + timedelta(seconds=2)).isoformat()

    # Log a prompt execution
    metrics_logger.log_prompt(
        prompt_id=prompt_id,
        prompt=mock_prompt,
        prompt_output=mock_output,
        input_parameters={
            "extracted_prompt_tokens": 1000,
            "extracted_completion_tokens": 500,
            "extracted_total_tokens": 1500,
            "extracted_model_id": "anthropic.claude-3-sonnet",
        },
        schedule_id=schedule_id,
        prompt_start_ts=start_ts,
        prompt_end_ts=end_ts,
        amendment_id=amendment_id,
        prompt_category="APPLY_AMENDMENT",
    )

    # Verify the prompt was logged
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT prompt_name, prompt_tokens, completion_tokens, total_tokens, model_id "
        "FROM prompts WHERE schedule_id = ?",
        (schedule_id,),
    )
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == "TestPrompt"
    assert result[1] == 1000
    assert result[2] == 500
    assert result[3] == 1500
    assert result[4] == "anthropic.claude-3-sonnet"

    conn.close()


def test_determine_service_id(metrics_logger):
    """Test the _determine_service_id method."""
    # Test with only AWS Bedrock enabled
    metrics_logger.enable_azure_openai = False
    metrics_logger.enable_aws_bedrock = True

    mock_prompt = MagicMock()
    service_id = metrics_logger._determine_service_id(mock_prompt)
    assert service_id == "bedrock-claude"

    # Test with only Azure OpenAI enabled
    metrics_logger.enable_azure_openai = True
    metrics_logger.enable_aws_bedrock = False

    service_id = metrics_logger._determine_service_id(mock_prompt)
    assert service_id == "gpt-4o"

    # Test with both enabled, with Azure settings
    metrics_logger.enable_azure_openai = True
    metrics_logger.enable_aws_bedrock = True

    mock_prompt = MagicMock()
    mock_azure_settings = MagicMock()
    mock_prompt.prompt_execution_settings = {"gpt-4o": mock_azure_settings}

    service_id = metrics_logger._determine_service_id(mock_prompt)
    assert service_id == "gpt-4o"

    # Test with both enabled, with Bedrock settings
    mock_prompt = MagicMock()
    mock_bedrock_settings = MagicMock()
    mock_prompt.prompt_execution_settings = {"bedrock-claude": mock_bedrock_settings}

    service_id = metrics_logger._determine_service_id(mock_prompt)
    assert service_id == "bedrock-claude"


def test_extract_metadata(metrics_logger):
    """Test the _extract_metadata method."""
    # Test with pre-extracted token info
    mock_prompt = MagicMock()
    mock_output = MagicMock()

    input_parameters = {
        "extracted_prompt_tokens": 1000,
        "extracted_completion_tokens": 500,
        "extracted_total_tokens": 1500,
        "extracted_model_id": "test-model-id",
    }

    metadata = metrics_logger._extract_metadata(mock_prompt, mock_output, input_parameters)

    assert metadata["prompt_tokens"] == 1000
    assert metadata["completion_tokens"] == 500
    assert metadata["total_tokens"] == 1500
    assert metadata["model_id"] == "test-model-id"

    # Test cost calculation
    assert metadata["cost_usd"] > 0

    # Test with no pre-extracted info, falling back to defaults
    input_parameters = {}

    # Mock for Bedrock service determination
    mock_prompt = MagicMock()
    metrics_logger.enable_azure_openai = False
    metrics_logger.enable_aws_bedrock = True

    metadata = metrics_logger._extract_metadata(mock_prompt, mock_output, input_parameters)

    assert metadata["prompt_tokens"] is None
    assert metadata["completion_tokens"] is None
    assert metadata["total_tokens"] is None
    assert metadata["model_id"] == "anthropic.claude-3-sonnet-20240229"
    assert metadata["cost_usd"] == 0.0


def test_log_schedule_start_exception(metrics_logger):
    """Test exception handling in log_schedule_start method."""
    schedule_id = str(uuid.uuid4())

    # Create a patch to _get_db_connection that raises an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Test exception")

        # Call the method, which should catch the exception
        metrics_logger.log_schedule_start(
            schedule_id=schedule_id,
            act_name="Test Act",
            model_id="test-model",
            service_id="test-service",
            max_worker_threads=256,
        )
        # Verify the connection was attempted
        assert mock_get_conn.called


def test_log_schedule_end_exception(metrics_logger):
    """Test exception handling in log_schedule_end method."""
    schedule_id = str(uuid.uuid4())

    # Create a patch to _get_db_connection that raises an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Test exception")

        # Call the method, which should catch the exception
        metrics_logger.log_schedule_end(schedule_id=schedule_id, total_amendments_found=10, total_amendments_applied=8)
        # Verify the connection was attempted
        assert mock_get_conn.called


def test_log_amendment_exception(metrics_logger):
    """Test exception handling in log_amendment method."""
    schedule_id = str(uuid.uuid4())
    amendment_id = str(uuid.uuid4())

    # Create a patch to _get_db_connection that raises an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Test exception")

        # Call the method, which should catch the exception
        metrics_logger.log_amendment(
            schedule_id=schedule_id,
            amendment_id=amendment_id,
            source="Section 1",
            source_eid="s1",
            affected_provision="Section 2",
            location="AFTER",
            amendment_type="INSERTION",
            whole_provision=True,
        )
        # Verify the connection was attempted
        assert mock_get_conn.called


def test_update_amendment_application_not_found(metrics_logger, test_db_path):
    """Test update_amendment_application when amendment not found."""
    # Create a non-existent amendment ID
    amendment_id = str(uuid.uuid4())

    # Patch logging.error to check it's called
    with patch("app.benchmarking.metrics_logger.logger.error") as mock_logger:
        metrics_logger.update_amendment_application(
            amendment_id=amendment_id, application_time_seconds=1.0, application_status=True
        )

        # Verify logging.error was called with expected message
        mock_logger.assert_called_once()
        assert f"Amendment {amendment_id} not found" in mock_logger.call_args[0][0]


def test_update_amendment_application_exception(metrics_logger):
    """Test exception handling in update_amendment_application method."""
    amendment_id = str(uuid.uuid4())

    # Create a patch to _get_db_connection that raises an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Test exception")

        # Call the method, which should catch the exception
        metrics_logger.update_amendment_application(
            amendment_id=amendment_id, application_time_seconds=1.0, application_status=True
        )
        # Verify the connection was attempted
        assert mock_get_conn.called


def test_log_prompt_timestamp_error(metrics_logger):
    """Test error handling for timestamp conversion in log_prompt."""
    schedule_id = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())

    # Create mock prompt and output
    mock_prompt = MagicMock(spec=KernelFunctionFromPrompt)
    mock_prompt.name = "TestPrompt"
    mock_output = MagicMock(spec=FunctionResult)

    # Deliberately provide invalid timestamps
    with patch("app.benchmarking.metrics_logger.logger.error") as mock_logger:
        metrics_logger.log_prompt(
            prompt_id=prompt_id,
            prompt=mock_prompt,
            prompt_output=mock_output,
            input_parameters={},
            schedule_id=schedule_id,
            prompt_start_ts="invalid-timestamp",  # This will cause ValueError
            prompt_end_ts="also-invalid",
            amendment_id=None,
            prompt_category="TEST",
        )

        # Verify logging.error was called
        mock_logger.assert_called_once()
        assert "Error calculating inference time" in mock_logger.call_args[0][0]


def test_log_prompt_exception(metrics_logger):
    """Test exception handling in log_prompt method."""
    schedule_id = str(uuid.uuid4())
    prompt_id = str(uuid.uuid4())

    # Create mock objects
    mock_prompt = MagicMock(spec=KernelFunctionFromPrompt)
    mock_prompt.name = "TestPrompt"
    mock_output = MagicMock(spec=FunctionResult)

    # Set up timestamps
    start_ts = datetime.utcnow().isoformat()
    end_ts = (datetime.utcnow() + timedelta(seconds=2)).isoformat()

    # Create a patch to _get_db_connection that raises an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = Exception("Test exception")

        # Also patch logger to capture error
        with (
            patch("app.benchmarking.metrics_logger.logger.error") as mock_logger,
            patch("app.benchmarking.metrics_logger.logger.exception") as mock_exception,
        ):
            metrics_logger.log_prompt(
                prompt_id=prompt_id,
                prompt=mock_prompt,
                prompt_output=mock_output,
                input_parameters={"extracted_prompt_tokens": 100},
                schedule_id=schedule_id,
                prompt_start_ts=start_ts,
                prompt_end_ts=end_ts,
                amendment_id=None,
                prompt_category=None,
            )

            # Verify error and exception logging were called
            assert mock_logger.called
            assert mock_exception.called
            assert "Failed to log prompt" in mock_logger.call_args[0][0]


def test_extract_metadata_different_service(metrics_logger):
    """Test _extract_metadata with azure service."""
    # Configure logger to use Azure
    metrics_logger.enable_aws_bedrock = False
    metrics_logger.enable_azure_openai = True

    # Create mock objects
    mock_prompt = MagicMock()
    mock_output = MagicMock()

    # Test with empty input parameters (missing tokens)
    input_parameters = {}

    metadata = metrics_logger._extract_metadata(mock_prompt, mock_output, input_parameters)

    # Check that Azure model ID is used when Azure is enabled
    assert metadata["model_id"] == "gpt-4o"
    assert metadata["cost_usd"] == 0.0  # No tokens, so cost should be zero


def test_calculate_duration_no_start_timestamp(metrics_logger):
    """Test _calculate_duration when start_timestamp is missing."""
    schedule_id = str(uuid.uuid4())
    end_ts = datetime.utcnow().isoformat()

    # Add schedule to _current_operations without start_timestamp
    metrics_logger._current_operations[schedule_id] = {"amendments": {}}  # Missing 'start_timestamp' key

    result = metrics_logger._calculate_duration(schedule_id, end_ts)
    assert result is None


def test_calculate_duration_invalid_timestamps(metrics_logger):
    """Test _calculate_duration with invalid timestamp formats."""
    schedule_id = str(uuid.uuid4())

    # Add schedule with invalid start timestamp
    metrics_logger._current_operations[schedule_id] = {"start_timestamp": "invalid-timestamp-format", "amendments": {}}

    end_ts = datetime.utcnow().isoformat()

    result = metrics_logger._calculate_duration(schedule_id, end_ts)
    assert result is None


def test_calculate_duration_type_error(metrics_logger):
    """Test _calculate_duration when timestamps cause TypeError."""
    schedule_id = str(uuid.uuid4())

    # Add schedule with None as start timestamp
    metrics_logger._current_operations[schedule_id] = {"start_timestamp": None, "amendments": {}}

    end_ts = datetime.utcnow().isoformat()

    result = metrics_logger._calculate_duration(schedule_id, end_ts)
    assert result is None


def test_calculate_application_rate_zero_found(metrics_logger):
    """Test _calculate_application_rate when total_found is zero."""
    result = metrics_logger._calculate_application_rate(0, 0)
    assert result is None

    # Also test with non-zero applied but zero found (edge case)
    result = metrics_logger._calculate_application_rate(0, 5)
    assert result is None


def test_calculate_inference_time_none_timestamps(metrics_logger):
    """Test _calculate_inference_time with None timestamps."""
    # Test with None start_ts
    result = metrics_logger._calculate_inference_time(None, datetime.utcnow().isoformat())
    assert result is None

    # Test with None end_ts
    result = metrics_logger._calculate_inference_time(datetime.utcnow().isoformat(), None)
    assert result is None

    # Test with both None
    result = metrics_logger._calculate_inference_time(None, None)
    assert result is None


def test_calculate_inference_time_empty_timestamps(metrics_logger):
    """Test _calculate_inference_time with empty string timestamps."""
    # Test with empty start_ts
    result = metrics_logger._calculate_inference_time("", datetime.utcnow().isoformat())
    assert result is None

    # Test with empty end_ts
    result = metrics_logger._calculate_inference_time(datetime.utcnow().isoformat(), "")
    assert result is None

    # Test with both empty
    result = metrics_logger._calculate_inference_time("", "")
    assert result is None


def test_calculate_duration_missing_schedule(metrics_logger):
    """Test _calculate_duration when schedule_id is not in _current_operations."""
    schedule_id = str(uuid.uuid4())
    end_ts = datetime.utcnow().isoformat()

    # Don't add the schedule to _current_operations
    result = metrics_logger._calculate_duration(schedule_id, end_ts)
    assert result is None


def test_calculate_application_rate_edge_cases(metrics_logger):
    """Test _calculate_application_rate with various edge cases."""
    # Normal case - 80% success rate
    result = metrics_logger._calculate_application_rate(10, 8)
    assert result == 80.0

    # 100% success rate
    result = metrics_logger._calculate_application_rate(5, 5)
    assert result == 100.0

    # 0% success rate (but with non-zero found)
    result = metrics_logger._calculate_application_rate(5, 0)
    assert result == 0.0

    # Fractional success rate
    result = metrics_logger._calculate_application_rate(3, 2)
    assert abs(result - 66.66666666666667) < 0.0001


def test_prepare_token_value(metrics_logger):
    """Test _prepare_token_value with various inputs."""
    # Test with integer
    assert metrics_logger._prepare_token_value(100) == 100

    # Test with float
    assert metrics_logger._prepare_token_value(100.5) == 100

    # Test with string number
    assert metrics_logger._prepare_token_value("100") == 100

    # Test with None
    assert metrics_logger._prepare_token_value(None) is None

    # Test with zero
    assert metrics_logger._prepare_token_value(0) == 0


def test_clear_amendment_from_memory(metrics_logger):
    """Test _clear_amendment_from_memory method."""
    schedule_id = str(uuid.uuid4())
    amendment_id = str(uuid.uuid4())

    # Add schedule and amendment to _current_operations
    metrics_logger._current_operations[schedule_id] = {
        "start_timestamp": datetime.utcnow().isoformat(),
        "amendments": {amendment_id: {"start_time": datetime.utcnow()}},
    }

    # Clear the amendment
    metrics_logger._clear_amendment_from_memory(schedule_id, amendment_id)

    # Verify it's removed
    assert amendment_id not in metrics_logger._current_operations[schedule_id]["amendments"]

    # Test clearing non-existent amendment (should not raise error)
    metrics_logger._clear_amendment_from_memory(schedule_id, "non-existent-id")

    # Test clearing from non-existent schedule (should not raise error)
    metrics_logger._clear_amendment_from_memory("non-existent-schedule", amendment_id)


def test_update_schedule_act_size(metrics_logger, test_db_path):
    """Test updating the act XML size for a schedule."""
    schedule_id = str(uuid.uuid4())

    # First create a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Test Act",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
        bill_xml_size=1000,
        act_xml_size=None,  # Initially None
    )

    # Update the act size
    metrics_logger.update_schedule_act_size(schedule_id, 2500)

    # Verify it was updated in the database
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT act_xml_size FROM schedules WHERE schedule_id = ?", (schedule_id,))
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == 2500
    conn.close()


def test_update_schedule_act_size_exception(metrics_logger):
    """Test exception handling in update_schedule_act_size method."""
    schedule_id = str(uuid.uuid4())

    # Mock _get_db_connection to raise an exception
    with patch.object(metrics_logger, "_get_db_connection") as mock_get_conn:
        mock_get_conn.side_effect = sqlite3.Error("Database is locked")

        # Should not raise exception
        metrics_logger.update_schedule_act_size(schedule_id, 2500)

        # Verify connection was attempted
        mock_get_conn.assert_called_once()


def test_create_ground_truth_table(test_db_path):
    """Test that ground_truth table is created with correct schema."""
    # Create a logger which will initialise the database
    MetricsLogger(db_path=test_db_path)

    # Connect and verify table exists
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()

    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ground_truth';")
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == "ground_truth"

    conn.close()


def test_load_ground_truth_if_needed(metrics_logger, test_db_path, tmp_path):
    """Test loading ground truth CSV into database."""
    # Create a test CSV file
    csv_content = """source,source_eid,type_of_amendment,affected_provision,location,whole_provision
                    s. 28(2)(a),sec_28__subsec_2__para_a,substitution,sec_212__subsec_1,Replace,FALSE
                    s. 28(5),sec_28__subsec_5,substitution,sec_215,Replace,TRUE"""

    csv_file = tmp_path / "test_ground_truth.csv"
    csv_file.write_text(csv_content)

    # Load the ground truth
    metrics_logger.load_ground_truth_if_needed("test_dataset", str(csv_file))

    # Verify data was loaded
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?", ("test_dataset",))
    count = cursor.fetchone()[0]
    assert count == 2

    conn.close()


def test_load_ground_truth_already_loaded(metrics_logger, test_db_path, tmp_path):
    """Test that ground truth is not reloaded if already present."""
    # Create a test CSV file
    csv_content = """source,source_eid,type_of_amendment,affected_provision,location,whole_provision
                    s. 101(2),sec_101__subsec_2,insertion,sec_1__subsec_3,After,FALSE"""

    csv_file = tmp_path / "test_ground_truth.csv"
    csv_file.write_text(csv_content)

    # Load once
    metrics_logger.load_ground_truth_if_needed("test_dataset2", str(csv_file))

    # Try to load again - should skip
    metrics_logger.load_ground_truth_if_needed("test_dataset2", str(csv_file))

    # Verify only loaded once (1 row, not 2)
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ground_truth WHERE dataset_name = ?", ("test_dataset2",))
    count = cursor.fetchone()[0]
    assert count == 1

    conn.close()


def test_auto_load_ground_truth_with_error(test_db_path, tmp_path):
    """Test auto-load handles CSV loading errors gracefully."""
    # Create ground truth directory with invalid CSV
    ground_truth_dir = tmp_path / "ground_truth"
    ground_truth_dir.mkdir()

    # Create invalid CSV file (missing required columns)
    bad_csv = ground_truth_dir / "bad_dataset.csv"
    bad_csv.write_text(
        """invalid,columns
                        1,2
                        3,4"""
    )

    # Create logger - should log error for bad file
    db_path = tmp_path / "keeling_metrics.db"

    with patch("app.benchmarking.metrics_logger.logger.error") as mock_logger:
        MetricsLogger(db_path=str(db_path))

        # Check error was logged for bad file
        assert mock_logger.called
        error_call = mock_logger.call_args[0][0]
        assert "Failed to load ground truth bad_dataset.csv" in error_call


def test_get_dataset_name_from_act(metrics_logger):
    """Test matching act names to dataset names."""
    # Mock some ground truth datasets in the database
    conn = sqlite3.connect(metrics_logger.db_path)
    cursor = conn.cursor()

    # Insert mock dataset names
    cursor.execute(
        """
        INSERT INTO ground_truth (ground_truth_id, dataset_name, source, source_eid,
                                  type_of_amendment, affected_provision, location, whole_provision)
        VALUES (?, ?, 'test', 'test', 'INSERTION', 'test', 'BEFORE', 0)
    """,
        (str(uuid.uuid4()), "renters_rights_housing_2004"),
    )

    cursor.execute(
        """
        INSERT INTO ground_truth (ground_truth_id, dataset_name, source, source_eid,
                                  type_of_amendment, affected_provision, location, whole_provision)
        VALUES (?, ?, 'test', 'test', 'INSERTION', 'test', 'BEFORE', 0)
    """,
        (str(uuid.uuid4()), "land_reform_scotland_agricultural_holdings_scotland_2003"),
    )

    conn.commit()
    conn.close()

    # Test successful matches
    assert metrics_logger._get_dataset_name_from_act("Housing Act 2004") == "renters_rights_housing_2004"
    assert (
        metrics_logger._get_dataset_name_from_act("Agricultural Holdings (Scotland) Act 2003")
        == "land_reform_scotland_agricultural_holdings_scotland_2003"
    )

    # Test no match
    assert metrics_logger._get_dataset_name_from_act("Some Other Act 2005") is None


def test_evaluate_schedule_accuracy_no_ground_truth(metrics_logger):
    """Test evaluation when no ground truth is available."""
    schedule_id = str(uuid.uuid4())

    # Log a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Unknown Act 2005",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    # Evaluate - should return empty dict as no ground truth matches
    metrics = metrics_logger.evaluate_schedule_accuracy(schedule_id, "Unknown Act 2005")
    assert metrics == {}


def test_evaluate_schedule_accuracy_with_ground_truth(metrics_logger):
    """Test evaluation with matching ground truth."""
    schedule_id = str(uuid.uuid4())

    # Insert ground truth data
    conn = sqlite3.connect(metrics_logger.db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO ground_truth (ground_truth_id, dataset_name, source, source_eid,
                                  type_of_amendment, affected_provision, location, whole_provision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (str(uuid.uuid4()), "renters_rights_housing_2004", "Section 1", "s1", "INSERTION", "section 12", "BEFORE", 1),
    )

    cursor.execute(
        """
        INSERT INTO ground_truth (ground_truth_id, dataset_name, source, source_eid,
                                  type_of_amendment, affected_provision, location, whole_provision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (str(uuid.uuid4()), "renters_rights_housing_2004", "Section 2", "s2", "DELETION", "section 13", "REPLACE", 0),
    )

    conn.commit()
    conn.close()

    # Log a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Housing Act 2004",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    # Log amendments - one correct, one incorrect
    metrics_logger.log_amendment(
        schedule_id=schedule_id,
        amendment_id=str(uuid.uuid4()),
        source="Section 1",
        source_eid="s1",
        affected_provision="section 12",
        location="BEFORE",
        amendment_type="INSERTION",
        whole_provision=True,
    )

    metrics_logger.log_amendment(
        schedule_id=schedule_id,
        amendment_id=str(uuid.uuid4()),
        source="Section 3",
        source_eid="s3",
        affected_provision="section 14",
        location="AFTER",
        amendment_type="SUBSTITUTION",
        whole_provision=False,
    )

    # Update first amendment as successfully applied
    conn = sqlite3.connect(metrics_logger.db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE amendments SET application_status = 1
        WHERE schedule_id = ? AND source_eid = 's1'
    """,
        (schedule_id,),
    )
    conn.commit()
    conn.close()

    # Evaluate
    metrics = metrics_logger.evaluate_schedule_accuracy(schedule_id, "Housing Act 2004")

    # Check metrics exist and are reasonable
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "location_accuracy" in metrics
    assert "whole_provision_accuracy" in metrics
    assert "geometric_mean_application" in metrics

    # Check specific values
    assert metrics["true_positives"] == 1  # One correct match
    assert metrics["false_positives"] == 1  # One incorrect
    assert metrics["false_negatives"] == 1  # One missed from ground truth
    assert metrics["precision"] == 0.5  # 1/(1+1)
    assert metrics["recall"] == 0.5  # 1/(1+1)

    # Check database was updated
    conn = sqlite3.connect(metrics_logger.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_name, identification_precision FROM schedules WHERE schedule_id = ?", (schedule_id,))
    result = cursor.fetchone()
    assert result[0] == "renters_rights_housing_2004"
    assert result[1] == 0.5
    conn.close()


def test_calculate_identification_metrics(metrics_logger):
    """Test calculation of identification metrics."""
    schedule_id = str(uuid.uuid4())
    dataset_name = "test_dataset"

    # Setup test data
    conn = sqlite3.connect(metrics_logger.db_path)
    cursor = conn.cursor()

    # Add schedule
    cursor.execute(
        """
        INSERT INTO schedules (schedule_id, act_name, model_id, service_id, start_timestamp)
        VALUES (?, 'Test Act', 'test-model', 'test-service', ?)
    """,
        (schedule_id, datetime.utcnow().isoformat()),
    )

    # Add ground truth
    cursor.execute(
        """
        INSERT INTO ground_truth (ground_truth_id, dataset_name, source, source_eid,
                                  type_of_amendment, affected_provision, location, whole_provision)
        VALUES (?, ?, 's1', 'eid1', 'INSERTION', 'section 1', 'BEFORE', 1)
    """,
        (str(uuid.uuid4()), dataset_name),
    )

    # Add matching amendment (true positive)
    cursor.execute(
        """
        INSERT INTO amendments (amendment_id, schedule_id, source, source_eid,
                                affected_provision, location, amendment_type, whole_provision)
        VALUES (?, ?, 's1', 'eid1', 'section 1', 'BEFORE', 'INSERTION', 1)
    """,
        (str(uuid.uuid4()), schedule_id),
    )

    # Add non-matching amendment (false positive)
    cursor.execute(
        """
        INSERT INTO amendments (amendment_id, schedule_id, source, source_eid,
                                affected_provision, location, amendment_type, whole_provision)
        VALUES (?, ?, 's2', 'eid2', 'section 2', 'AFTER', 'DELETION', 0)
    """,
        (str(uuid.uuid4()), schedule_id),
    )

    conn.commit()

    # Calculate metrics
    metrics = metrics_logger._calculate_identification_metrics(cursor, schedule_id, dataset_name)

    conn.close()

    # Verify
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1"] > 0.6  # Should be 2/3


def test_evaluate_schedule_accuracy_dataset_name_not_in_table(metrics_logger):
    """Test evaluation when dataset name is matched but not in ground_truth table."""
    schedule_id = str(uuid.uuid4())

    # Log a schedule
    metrics_logger.log_schedule_start(
        schedule_id=schedule_id,
        act_name="Housing Act 2004",
        model_id="test-model",
        service_id="test-service",
        max_worker_threads=256,
    )

    # Override _get_dataset_name_from_act to return a dataset that doesn't exist
    metrics_logger._get_dataset_name_from_act = lambda x: "non_existent_dataset"

    # Evaluate - should return empty dict as dataset doesn't exist in ground_truth
    metrics = metrics_logger.evaluate_schedule_accuracy(schedule_id, "Housing Act 2004")
    assert metrics == {}
