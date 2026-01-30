# app/kernel/llm_retry_handler.py
"""
Retry logic handler for LLM API calls with exponential backoff.

Provides intelligent retry mechanisms for handling rate limits and transient errors
when calling LLM services. Implements exponential backoff with configurable retry
sequences and respects API-provided retry-after headers.
"""

import asyncio
import re
from typing import Optional, Callable, Any

from semantic_kernel.exceptions import KernelInvokeException
from ..logging.debug_logger import get_logger, event, bind, EventType as EVT

logger = get_logger(__name__)


class LLMRetryHandler:
    """
    Handles retry logic for LLM calls with exponential backoff.

    This class provides a robust retry mechanism that:
    - Detects rate limit errors (HTTP 429) and retries automatically
    - Implements exponential backoff to avoid overwhelming the API
    - Respects Retry-After headers from the API when present
    - Distinguishes between retryable and non-retryable errors

    The handler is designed to be used with async functions and integrates
    seamlessly with the semantic kernel's exception handling.
    """

    def __init__(self, max_retries: int, backoff_sequence: list):
        """
        Initialise the retry handler with configuration.

        Args:
            max_retries: Maximum number of retry attempts before giving up
            backoff_sequence: List of wait times (in seconds) for each retry attempt
                             e.g., [1, 2, 4, 8, 16] for exponential backoff
        """
        self.max_retries = max_retries
        self.backoff_sequence = backoff_sequence

    # ==================== Public Interface Methods ====================

    async def execute_with_retry(
        self, func: Callable, prompt_name: str, request_count: int, prompt_id: str, candidate_eid: Optional[str] = None
    ) -> Any:
        """
        Execute an async function with automatic retry on rate limits.

        This is the main entry point for the retry handler. It executes the provided
        function and automatically retries on rate limit errors with exponential backoff.

        Args:
            func: Async function to execute (typically an LLM API call)
            prompt_name: Name of the prompt being executed (for logging)
            request_count: Sequential request number (for debugging)
            prompt_id: Unique identifier for this prompt execution
            candidate_eid: Optional candidate element ID for context

        Returns:
            Result from the function on success, or formatted error string on failure

        Note:
            The function will retry up to max_retries times for rate limit errors.
            Other errors will cause immediate failure without retry.
        """
        retry_count = 0
        last_error = None

        # Re-bind the prompt_id and candidate_eid in case we're in a new async context
        with bind(prompt_id=prompt_id, candidate_eid=candidate_eid):
            while retry_count <= self.max_retries:
                try:
                    logger.info(
                        f"Attempt {retry_count + 1}/{self.max_retries + 1} " f"to invoke prompt '{prompt_name}'"
                    )

                    result = await func()
                    logger.info(f"Request #{request_count} succeeded")
                    return result

                except KernelInvokeException as e:
                    last_error = e
                    logger.debug(f"KernelInvokeException caught: {str(e)}")

                    # Check if this is a retryable error and we have retries left
                    if self._is_rate_limit_error(e) and retry_count < self.max_retries:
                        retry_count += 1
                        wait_time = self._calculate_wait_time(e, retry_count)

                        event(
                            logger,
                            EVT.LLM_RETRY,
                            f"Rate limit hit, retrying after {wait_time:.2f}s",
                            prompt_name=prompt_name,
                            retry_count=retry_count,
                            max_retries=self.max_retries,
                            wait_seconds=wait_time,
                        )

                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Non-rate-limit error or retries exhausted
                        break

                except Exception as e:
                    # Unexpected error - don't retry
                    last_error = e
                    logger.exception("Unexpected error in retry handler")
                    break

            # All retries exhausted or non-retryable error encountered
            return self._format_error(last_error, prompt_name)

    # ==================== Private Helper Methods ====================

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """
        Check if an exception represents a rate limit error.

        Examines both the exception message and any underlying cause
        for indicators of rate limiting (HTTP 429 or RateLimitError).

        Args:
            exception: The exception to check

        Returns:
            True if this is a rate limit error, False otherwise
        """
        # Check main exception
        exception_str = str(exception)
        if "429" in exception_str or "RateLimitError" in exception_str:
            return True

        # Check underlying cause (often where the actual HTTP error is)
        if hasattr(exception, "__cause__") and exception.__cause__:
            cause_str = str(exception.__cause__)
            if "429" in cause_str or "RateLimitError" in cause_str:
                return True

        return False

    def _calculate_wait_time(self, exception: Exception, retry_count: int) -> float:
        """
        Calculate the wait time before the next retry attempt.

        Uses the maximum of:
        1. API-provided Retry-After header (if present)
        2. Configured exponential backoff time

        This ensures we respect the API's rate limit requirements while
        also implementing our own backoff strategy.

        Args:
            exception: The exception that triggered the retry
            retry_count: Current retry attempt number (1-based)

        Returns:
            Wait time in seconds
        """
        # Extract API-provided retry time if available
        retry_after = self._extract_retry_after_seconds(exception)

        # Get our configured backoff time
        # Use retry_count - 1 as index (retry_count is 1-based)
        backoff_index = min(retry_count - 1, len(self.backoff_sequence) - 1)
        exponential_wait = self.backoff_sequence[backoff_index]

        # Use the longer of the two to ensure we wait long enough
        return max(retry_after or 0, exponential_wait)

    def _extract_retry_after_seconds(self, exception: Exception) -> Optional[int]:
        """
        Extract the Retry-After value from an exception if available.

        Attempts to find the retry delay from:
        1. HTTP response headers (preferred)
        2. Error message text (fallback)

        Args:
            exception: The exception to extract retry delay from

        Returns:
            Retry delay in seconds if found, None otherwise
        """
        # Try to get from response headers first (most reliable)
        if hasattr(exception, "response") and exception.response:
            headers = getattr(exception.response, "headers", {})
            retry_after = headers.get("Retry-After")

            if retry_after and retry_after.isdigit():
                return int(retry_after)

        # Fallback: try to extract from error message
        # Look for patterns like 'Retry-After: 60' or 'Retry-After "60"'
        match = re.search(r'Retry-After[:\s]*["\']?(\d+)', str(exception))
        if match:
            return int(match.group(1))

        return None

    def _format_error(self, error: Exception, prompt_name: str) -> str:
        """
        Format an error message for return to the caller.

        Provides context-aware error messages based on the error type
        to help with debugging and user feedback.

        Args:
            error: The final error after all retries
            prompt_name: Name of the prompt that failed

        Returns:
            Formatted error message string
        """
        error_str = str(error)

        if "429" in error_str or "RateLimitError" in error_str:
            return f"Rate limit exceeded after {self.max_retries} retries: {error}"
        elif isinstance(error, KernelInvokeException):
            return f"Kernel failed to invoke prompt '{prompt_name}'. Error: {error}"
        else:
            return f"Error processing request: {error}"
