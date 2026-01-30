# tests/unit/kernel/test_llm_retry_handler.py
"""
Unit tests for the LLMRetryHandler class.
"""

from unittest.mock import Mock, patch, AsyncMock
import pytest
import uuid
from semantic_kernel.exceptions import KernelInvokeException
from app.kernel.llm_retry_handler import LLMRetryHandler


class TestLLMRetryHandler:
    """Tests for LLMRetryHandler class."""

    def test_init(self):
        """Test that LLMRetryHandler initialises with correct values."""
        max_retries = 3
        backoff_sequence = [1, 2, 4, 8]

        handler = LLMRetryHandler(max_retries, backoff_sequence)

        assert handler.max_retries == max_retries
        assert handler.backoff_sequence == backoff_sequence

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        # Mock successful async function
        mock_func = AsyncMock(return_value="success")
        prompt_id = str(uuid.uuid4())

        result = await handler.execute_with_retry(
            mock_func, prompt_name="test_prompt", request_count=1, prompt_id=prompt_id
        )

        assert result == "success"
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retries(self):
        """Test successful execution after rate limit retries."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[0.1, 0.2, 0.4])

        # Mock function that fails twice with rate limit, then succeeds
        mock_func = AsyncMock()
        rate_limit_error = KernelInvokeException("Rate limit error: 429")
        mock_func.side_effect = [rate_limit_error, rate_limit_error, "success"]
        prompt_id = str(uuid.uuid4())

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await handler.execute_with_retry(
                mock_func, prompt_name="test_prompt", request_count=1, prompt_id=prompt_id
            )

        assert result == "success"
        assert mock_func.call_count == 3
        # Should have slept twice
        assert mock_sleep.call_count == 2
        # Check sleep durations
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_exhausted(self):
        """Test that retries are exhausted and error is returned."""
        handler = LLMRetryHandler(max_retries=2, backoff_sequence=[0.1, 0.2])

        # Mock function that always fails with rate limit
        mock_func = AsyncMock()
        rate_limit_error = KernelInvokeException("RateLimitError: Too many requests")
        mock_func.side_effect = rate_limit_error
        prompt_id = str(uuid.uuid4())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await handler.execute_with_retry(
                mock_func, prompt_name="test_prompt", request_count=1, prompt_id=prompt_id
            )

        assert "Rate limit exceeded after 2 retries:" in result
        assert "RateLimitError" in result
        # Initial attempt + 2 retries = 3 total
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_rate_limit_error(self):
        """Test that non-rate-limit errors don't trigger retries."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[0.1, 0.2, 0.4])

        # Mock function that fails with non-rate-limit error
        mock_func = AsyncMock()
        other_error = KernelInvokeException("Invalid API key")
        mock_func.side_effect = other_error
        prompt_id = str(uuid.uuid4())

        result = await handler.execute_with_retry(
            mock_func, prompt_name="test_prompt", request_count=1, prompt_id=prompt_id
        )

        assert "Kernel failed to invoke prompt 'test_prompt'" in result
        assert "Invalid API key" in result
        # Should only try once
        mock_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_retry_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[0.1, 0.2, 0.4])

        # Mock function that raises unexpected exception
        mock_func = AsyncMock()
        mock_func.side_effect = ValueError("Unexpected error")
        prompt_id = str(uuid.uuid4())

        with patch("app.kernel.llm_retry_handler.logger") as mock_logger:
            result = await handler.execute_with_retry(
                mock_func, prompt_name="test_prompt", request_count=1, prompt_id=prompt_id
            )

        assert "Error processing request:" in result
        assert "Unexpected error" in result
        mock_func.assert_called_once()
        mock_logger.exception.assert_called_once_with("Unexpected error in retry handler")

    def test_is_rate_limit_error_with_429(self):
        """Test rate limit detection with 429 status code."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = KernelInvokeException("Error 429: Too Many Requests")
        assert handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_with_rate_limit_text(self):
        """Test rate limit detection with RateLimitError text."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = KernelInvokeException("RateLimitError: API rate limit exceeded")
        assert handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_with_cause(self):
        """Test rate limit detection in exception cause."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        # Create exception with rate limit in __cause__
        cause = Exception("429 Too Many Requests")
        error = KernelInvokeException("Wrapper error")
        error.__cause__ = cause

        assert handler._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_non_rate_limit(self):
        """Test that non-rate-limit errors are identified correctly."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = KernelInvokeException("Authentication failed")
        assert handler._is_rate_limit_error(error) is False

    def test_calculate_wait_time_with_retry_after_header(self):
        """Test wait time calculation with Retry-After header."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        # Mock exception with response headers
        error = Exception("Rate limited")
        error.response = Mock()
        error.response.headers = {"Retry-After": "5"}

        wait_time = handler._calculate_wait_time(error, retry_count=1)

        # Should use max of Retry-After (5) and backoff (1)
        assert wait_time == 5

    def test_calculate_wait_time_with_retry_after_in_message(self):
        """Test wait time extraction from error message."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception('Rate limited. Retry-After: "3" seconds')

        wait_time = handler._calculate_wait_time(error, retry_count=2)

        # Should use max of Retry-After (3) and backoff (2)
        assert wait_time == 3

    def test_calculate_wait_time_exponential_backoff_only(self):
        """Test wait time with only exponential backoff."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception("Rate limited")

        # Test different retry counts
        assert handler._calculate_wait_time(error, retry_count=1) == 1
        assert handler._calculate_wait_time(error, retry_count=2) == 2
        assert handler._calculate_wait_time(error, retry_count=3) == 4

    def test_calculate_wait_time_beyond_backoff_sequence(self):
        """Test wait time when retry count exceeds backoff sequence length."""
        handler = LLMRetryHandler(max_retries=5, backoff_sequence=[1, 2])

        error = Exception("Rate limited")

        # Should use last value in sequence
        assert handler._calculate_wait_time(error, retry_count=4) == 2
        assert handler._calculate_wait_time(error, retry_count=5) == 2

    def test_extract_retry_after_seconds_from_headers(self):
        """Test extracting Retry-After from response headers."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception("Rate limited")
        error.response = Mock()
        error.response.headers = {"Retry-After": "10"}

        retry_after = handler._extract_retry_after_seconds(error)
        assert retry_after == 10

    def test_extract_retry_after_seconds_non_digit_header(self):
        """Test that non-digit Retry-After header is ignored."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception("Rate limited")
        error.response = Mock()
        error.response.headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}

        retry_after = handler._extract_retry_after_seconds(error)
        assert retry_after is None

    def test_extract_retry_after_seconds_from_message(self):
        """Test extracting Retry-After from error message."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        # Test various message formats
        test_cases = [
            ("Retry-After: 5", 5),
            ("Retry-After:10", 10),
            ('Please wait. Retry-After: "15" seconds', 15),
            ("Retry-After:'20'", 20),
        ]

        for message, expected in test_cases:
            error = Exception(message)
            retry_after = handler._extract_retry_after_seconds(error)
            assert retry_after == expected

    def test_extract_retry_after_seconds_not_found(self):
        """Test that None is returned when Retry-After not found."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception("Generic error message")
        retry_after = handler._extract_retry_after_seconds(error)
        assert retry_after is None

    def test_format_error_rate_limit(self):
        """Test error formatting for rate limit errors."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = Exception("429 Too Many Requests")
        formatted = handler._format_error(error, "test_prompt")

        assert formatted == "Rate limit exceeded after 3 retries: 429 Too Many Requests"

    def test_format_error_kernel_invoke(self):
        """Test error formatting for KernelInvokeException."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = KernelInvokeException("Invalid parameters")
        formatted = handler._format_error(error, "test_prompt")

        assert formatted == "Kernel failed to invoke prompt 'test_prompt'. Error: Invalid parameters"

    def test_format_error_generic(self):
        """Test error formatting for generic exceptions."""
        handler = LLMRetryHandler(max_retries=3, backoff_sequence=[1, 2, 4])

        error = ValueError("Unexpected value")
        formatted = handler._format_error(error, "test_prompt")

        assert formatted == "Error processing request: Unexpected value"

    @pytest.mark.asyncio
    async def test_logging_during_retry(self):
        """Test that appropriate logging occurs during retry attempts."""
        handler = LLMRetryHandler(max_retries=2, backoff_sequence=[0.1, 0.2])

        # Mock function that fails once then succeeds
        mock_func = AsyncMock()
        rate_limit_error = KernelInvokeException("429 Rate limited")
        mock_func.side_effect = [rate_limit_error, "success"]
        prompt_id = str(uuid.uuid4())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with patch("app.kernel.llm_retry_handler.logger") as mock_logger:
                # Also patch the event function since it's imported
                with patch("app.kernel.llm_retry_handler.event") as mock_event:
                    result = await handler.execute_with_retry(
                        mock_func, prompt_name="test_prompt", request_count=5, prompt_id=prompt_id
                    )

        assert result == "success"

        # Check logging calls
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert "Attempt 1/3 to invoke prompt 'test_prompt'" in info_calls
        assert "Attempt 2/3 to invoke prompt 'test_prompt'" in info_calls
        assert "Request #5 succeeded" in info_calls

        # Check event() was called for retry event
        # event() is called with: logger, EventType.LLM_RETRY, message, **kwargs
        assert mock_event.called

        # Find the LLM_RETRY event call
        retry_event_calls = [
            call
            for call in mock_event.call_args_list
            if len(call[0]) >= 2 and hasattr(call[0][1], "value") and call[0][1].value == "LLM_RETRY"
        ]

        # Should have one retry event
        assert len(retry_event_calls) == 1

        # Check the retry event details
        retry_call = retry_event_calls[0]
        assert "Rate limit hit" in retry_call[0][2]  # message is 3rd positional arg

        # Check kwargs
        kwargs = retry_call[1]
        assert kwargs["retry_count"] == 1
        assert kwargs["max_retries"] == 2
        assert kwargs["wait_seconds"] == 0.1
