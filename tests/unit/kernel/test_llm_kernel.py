# tests/unit/kernel/test_llm_kernel.py
"""
Unit tests for the LLMKernel class.
"""

import threading
from unittest.mock import Mock, patch, AsyncMock
import pytest

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage

from app.kernel.llm_kernel import LLMKernel, get_kernel


class TestLLMKernel:
    """Tests for LLMKernel class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config object."""
        config = Mock()
        config.PROMPTS_PATH = "/path/to/prompts"
        config.LOG_PROMPTS = True
        return config

    @pytest.fixture
    def mock_dependencies(self):
        """Setup mock dependencies for LLMKernel."""
        with (
            patch("app.kernel.llm_kernel.LLMConfig") as mock_llm_config_class,
            patch("app.kernel.llm_kernel.LLMRetryHandler") as mock_retry_handler_class,
            patch("app.kernel.llm_kernel.MetricsLogger") as mock_metrics_logger_class,
            patch("app.kernel.llm_kernel.Kernel") as mock_kernel_class,
        ):

            # Setup mock instances
            mock_llm_config = Mock()
            mock_llm_config.max_retries = 5
            mock_llm_config.retry_backoff = [1, 2, 4, 8, 16]
            mock_llm_config.enable_azure_openai = True
            mock_llm_config.enable_aws_bedrock = False
            mock_llm_config.get_azure_api_key.return_value = "test-api-key"
            mock_llm_config.get_active_service_id.return_value = "gpt-4o"
            mock_llm_config.azure_model_deployment_name = "gpt-4o"
            mock_llm_config.azure_endpoint = "https://test.openai.azure.com/"
            mock_llm_config.azure_api_version = "2024-08-01-preview"
            mock_llm_config.bedrock_service_id = "bedrock-claude"
            mock_llm_config.bedrock_model_id = "claude-model"

            mock_llm_config_class.return_value = mock_llm_config

            # Setup kernel mock
            mock_kernel = Mock(spec=Kernel)
            mock_plugin = Mock()
            mock_kernel.add_plugin.return_value = mock_plugin
            mock_kernel_class.return_value = mock_kernel

            yield {
                "llm_config_class": mock_llm_config_class,
                "llm_config": mock_llm_config,
                "retry_handler_class": mock_retry_handler_class,
                "metrics_logger_class": mock_metrics_logger_class,
                "kernel_class": mock_kernel_class,
                "kernel": mock_kernel,
                "plugin": mock_plugin,
            }

    def test_init_successful(self, mock_config, mock_dependencies):
        """Test successful initialisation of LLMKernel."""
        with patch("threading.Thread") as mock_thread:
            kernel = LLMKernel(mock_config)

            # Verify component setup
            assert kernel.config == mock_config
            assert kernel.request_count == 0

            # Verify plugin setup
            mock_dependencies["kernel"].add_plugin.assert_called_once_with(
                plugin_name="prompts", parent_directory="/path/to/prompts"
            )

            # Verify retry handler creation
            mock_dependencies["retry_handler_class"].assert_called_once_with(5, [1, 2, 4, 8, 16])

            # Verify thread started
            mock_thread.assert_called_once()
            thread_instance = mock_thread.return_value
            thread_instance.start.assert_called_once()

    def test_init_no_api_key_raises_error(self, mock_config, mock_dependencies):
        """Test that missing API key raises RuntimeError."""
        mock_dependencies["llm_config"].get_azure_api_key.return_value = None

        with pytest.raises(RuntimeError, match="Azure OpenAI API key could not be retrieved"):
            LLMKernel(mock_config)

    def test_setup_services_azure_enabled(self, mock_config, mock_dependencies):
        """Test Azure OpenAI service setup when enabled."""
        with patch("app.kernel.llm_kernel.AzureChatCompletion") as mock_azure:
            with patch("threading.Thread"):
                LLMKernel(mock_config)

                # Verify Azure service created
                mock_azure.assert_called_once_with(
                    service_id="gpt-4o",
                    api_key="test-api-key",
                    endpoint="https://test.openai.azure.com/",
                    deployment_name="gpt-4o",
                    api_version="2024-08-01-preview",
                )

                # Verify service added to kernel
                mock_dependencies["kernel"].add_service.assert_called_once()

    def test_setup_services_bedrock_enabled(self, mock_config, mock_dependencies):
        """Test AWS Bedrock service setup when enabled."""
        # Configure for Bedrock
        mock_dependencies["llm_config"].enable_azure_openai = False
        mock_dependencies["llm_config"].enable_aws_bedrock = True
        mock_runtime = Mock()
        mock_client = Mock()
        mock_dependencies["llm_config"].get_bedrock_clients.return_value = (mock_runtime, mock_client)

        with patch("app.kernel.llm_kernel.BedrockChatCompletion") as mock_bedrock:
            with patch("threading.Thread"):
                LLMKernel(mock_config)

                # Verify Bedrock service created
                mock_bedrock.assert_called_once_with(
                    service_id="bedrock-claude",
                    model_id="claude-model",
                    runtime_client=mock_runtime,
                    client=mock_client,
                )

                # Verify service added to kernel
                mock_dependencies["kernel"].add_service.assert_called_once()

    def test_setup_logging(self, mock_config, mock_dependencies):
        """Test logging setup."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Verify MetricsLogger created with correct params
            mock_dependencies["metrics_logger_class"].assert_called_once_with(
                enable_aws_bedrock=False,
                bedrock_service_id="bedrock-claude",
                bedrock_model_id="claude-model",
                enable_azure_openai=True,
                azure_model_deployment_name="gpt-4o",
            )

            assert kernel.log_prompts is True

    @pytest.mark.asyncio
    async def test_run_inference_async_success(self, mock_config, mock_dependencies):
        """Test successful async inference execution."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Setup mocks
            mock_prompt = Mock(spec=KernelFunction)
            mock_prompt.name = "test_prompt"

            mock_messages = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}]

            mock_result = ChatMessageContent(role="assistant", content="Hello there!")

            # Setup retry handler to return result
            mock_retry_instance = Mock()
            mock_retry_instance.execute_with_retry = AsyncMock(return_value=mock_result)
            kernel.retry_handler = mock_retry_instance

            with patch("app.kernel.llm_kernel.create_chat_messages", return_value=mock_messages):
                prompt_id = "test-prompt-id"
                result = await kernel.run_inference_async(mock_prompt, prompt_id, test_param="value")

            assert result == mock_result
            assert kernel.request_count == 1

            # Verify retry handler called correctly
            mock_retry_instance.execute_with_retry.assert_called_once()

            # Get the call arguments
            call = mock_retry_instance.execute_with_retry.call_args

            # First positional arg should be the _execute function
            assert len(call[0]) == 1  # Only one positional argument
            assert callable(call[0][0])  # It should be a function

            # Check keyword arguments
            kwargs = call[1]
            assert kwargs["prompt_name"] == "test_prompt"
            assert kwargs["request_count"] == 1
            assert kwargs["prompt_id"] == prompt_id
            assert kwargs["candidate_eid"] is None

    @pytest.mark.asyncio
    async def test_run_inference_async_no_service_enabled(self, mock_config, mock_dependencies):
        """Test error when no LLM service is enabled."""
        mock_dependencies["llm_config"].get_active_service_id.return_value = None

        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_prompt = Mock(spec=KernelFunction)
            mock_prompt.name = "test_prompt"

            # Setup the retry handler to properly handle the function call with correct signature
            async def mock_execute_with_retry(func, prompt_name, request_count, prompt_id, candidate_eid=None):
                # Call the function to trigger the RuntimeError
                try:
                    return await func()
                except RuntimeError as e:
                    # Return error string like the real retry handler would
                    return f"Error processing request: {str(e)}"

            mock_retry_instance = Mock()
            mock_retry_instance.execute_with_retry = mock_execute_with_retry
            kernel.retry_handler = mock_retry_instance

            with patch("app.kernel.llm_kernel.create_chat_messages", return_value=[]):
                prompt_id = "test-prompt-id"
                result = await kernel.run_inference_async(mock_prompt, prompt_id)

            # Verify error message is returned
            assert result == "Error processing request: No LLM service is enabled"

    def test_run_inference_prompt_not_found(self, mock_config, mock_dependencies):
        """Test handling of missing prompt."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Setup plugin to not contain the prompt
            kernel.plugin = {}

            result = kernel.run_inference("missing_prompt", "schedule_123")

            assert result == "Prompt name not recognised"

    def test_run_inference_success(self, mock_config, mock_dependencies):
        """Test successful synchronous inference."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Setup plugin
            mock_prompt = Mock(spec=KernelFunction)
            mock_prompt.name = "test_prompt"
            # Description attribute that will be used by create_chat_messages
            mock_prompt.description = (
                """<message role="system">You are helpful</message><message role="user">{{param1}}</message>"""
            )

            kernel.plugin = {"test_prompt": mock_prompt}

            # Setup async result
            mock_chat_result = ChatMessageContent(
                role="assistant",
                content="```python\nprint('hello')\n```",
                metadata={"usage": CompletionUsage(prompt_tokens=10, completion_tokens=20)},
            )

            # Mock the event loop and future
            mock_future = Mock()
            mock_future.result.return_value = mock_chat_result

            with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(kernel, "_log_interaction") as mock_log:
                    result = kernel.run_inference(
                        "test_prompt", "schedule_123", amendment_id="amend_456", param1="value1"
                    )

            # Check cleaned response (code blocks removed)
            assert result == "print('hello')"

            # Verify logging called
            mock_log.assert_called_once()
            # Check that all the required parameters are passed
            assert len(mock_log.call_args[0]) == 13  # 13 positional arguments

    @pytest.mark.asyncio
    async def test_run_inference_async_execute_function(self, mock_config, mock_dependencies):
        """Test that the inner _execute function is properly executed."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Setup mocks
            mock_prompt = Mock(spec=KernelFunction)
            mock_prompt.name = "test_prompt"

            mock_messages = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}]

            # Mock the chat service
            mock_chat_service = Mock()
            mock_chat_result = ChatMessageContent(role="assistant", content="Hello there!")
            mock_chat_service.get_chat_message_content = AsyncMock(return_value=mock_chat_result)

            # Make kernel return our mock service
            kernel.kernel.get_service = Mock(return_value=mock_chat_service)

            # Mock the retry handler to actually call the function
            async def mock_retry_execute(func, prompt_name, request_count, prompt_id, candidate_eid=None):
                return await func()

            kernel.retry_handler.execute_with_retry = mock_retry_execute

            with patch("app.kernel.llm_kernel.create_chat_messages", return_value=mock_messages):
                with patch("app.kernel.llm_kernel.ChatHistory") as mock_history_class:
                    mock_history = Mock()
                    mock_history_class.return_value = mock_history

                    prompt_id = "test-prompt-id"
                    result = await kernel.run_inference_async(mock_prompt, prompt_id, test_param="value")

            # Verify result
            assert result == mock_chat_result

            # Verify chat history was built correctly
            assert mock_history.add_message.call_count == 2
            mock_history.add_message.assert_any_call(ChatMessageContent(role="system", content="You are helpful"))
            mock_history.add_message.assert_any_call(ChatMessageContent(role="user", content="Hello"))

            # Verify chat service was called
            mock_chat_service.get_chat_message_content.assert_called_once()

    def test_log_interaction_unknown_prompt_category(self, mock_config, mock_dependencies):
        """Test logging with a prompt that doesn't match known categories."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_prompt = Mock()
            metadata = {}

            # Use a prompt name that doesn't match any category
            kernel._log_interaction(
                "CustomPrompt",  # Not TableOfAmendments or Apply*
                "schedule_123",
                None,
                "prompt-id-123",
                {"param": "value"},
                "2024-01-01T10:00:00",
                "2024-01-01T10:00:01",
                mock_prompt,
                "Response text",
                metadata,
                None,  # prompt_tokens
                None,  # completion_tokens
                None,  # total_tokens
            )

            # Verify log call with prompt_category=None
            kernel.metrics_logger.log_prompt.assert_called_once()
            call_args = kernel.metrics_logger.log_prompt.call_args[1]

            assert call_args["prompt_category"] is None

    def test_run_inference_error_response(self, mock_config, mock_dependencies):
        """Test handling of error response from retry handler."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Setup plugin with proper mock prompt
            mock_prompt = Mock(spec=KernelFunction)
            mock_prompt.name = "test_prompt"
            mock_prompt.description = "<message role='user'>Test</message>"
            kernel.plugin = {"test_prompt": mock_prompt}

            # Return error string
            error_message = "Rate limit reached: 429 Too Many Requests"

            mock_future = Mock()
            mock_future.result.return_value = error_message

            with patch("asyncio.run_coroutine_threadsafe", return_value=mock_future):
                result = kernel.run_inference("test_prompt", "schedule_123")

            assert result == error_message

    def test_log_interaction_identification_prompt(self, mock_config, mock_dependencies):
        """Test logging of identification prompt."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_prompt = Mock()
            metadata = {"usage": CompletionUsage(prompt_tokens=100, completion_tokens=200), "model": "gpt-4o"}

            kernel._log_interaction(
                "TableOfAmendments",
                "schedule_123",
                None,
                "prompt-id-456",
                {"act_name": "Test Act"},
                "2024-01-01T10:00:00",
                "2024-01-01T10:00:01",
                mock_prompt,
                "Response text",
                metadata,
                100,  # prompt_tokens
                200,  # completion_tokens
                300,  # total_tokens
            )

            # Verify log call
            kernel.metrics_logger.log_prompt.assert_called_once()
            call_args = kernel.metrics_logger.log_prompt.call_args[1]

            assert call_args["prompt_category"] == "identification"
            assert call_args["input_parameters"]["extracted_prompt_tokens"] == 100
            assert call_args["input_parameters"]["extracted_completion_tokens"] == 200
            assert call_args["input_parameters"]["extracted_total_tokens"] == 300

    def test_log_interaction_application_prompt(self, mock_config, mock_dependencies):
        """Test logging of application prompt types."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            for prompt_name in ["ApplyInsertionAmendment", "ApplyDeletionAmendment", "ApplySubstitutionAmendment"]:
                kernel._log_interaction(
                    prompt_name,
                    "schedule_123",
                    "amend_456",
                    "prompt-id-789",
                    {},
                    "2024-01-01T10:00:00",
                    "2024-01-01T10:00:01",
                    Mock(),
                    "Response",
                    {},
                    None,  # prompt_tokens
                    None,  # completion_tokens
                    None,  # total_tokens
                )

                call_args = kernel.metrics_logger.log_prompt.call_args[1]
                assert call_args["prompt_category"] == "application"

    def test_log_interaction_preprocessing_prompt(self, mock_config, mock_dependencies):
        """Test logging of preprocessing prompt (IdentifyAmendmentReferences)."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_prompt = Mock()
            metadata = {}

            kernel._log_interaction(
                "IdentifyAmendmentReferences",
                "schedule_123",
                None,
                "prompt-id-123",
                {"act_name": "Test Act"},
                "2024-01-01T10:00:00",
                "2024-01-01T10:00:01",
                mock_prompt,
                "provision_type,start_number,end_number\nregulation,19,55",
                metadata,
                None,  # prompt_tokens
                None,  # completion_tokens
                None,  # total_tokens
            )

            # Verify log call with prompt_category="preprocessing"
            kernel.metrics_logger.log_prompt.assert_called_once()
            call_args = kernel.metrics_logger.log_prompt.call_args[1]

            assert call_args["prompt_category"] == "preprocessing"

    def test_log_interaction_pattern_extraction_prompt(self, mock_config, mock_dependencies):
        """Test logging of pattern extraction prompt (ExtractEachPlacePattern)."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_prompt = Mock()
            metadata = {}

            kernel._log_interaction(
                "ExtractEachPlacePattern",
                "schedule_123",
                "amend_456",
                "prompt-id-789",
                {"amendment_xml": "<amendment>for 'company' substitute 'corporation' in each place</amendment>"},
                "2024-01-01T10:00:00",
                "2024-01-01T10:00:01",
                mock_prompt,
                "find_text,replace_text\ncompany,corporation",
                metadata,
                None,  # prompt_tokens
                None,  # completion_tokens
                None,  # total_tokens
            )

            # Verify log call with prompt_category="pattern_extraction"
            kernel.metrics_logger.log_prompt.assert_called_once()
            call_args = kernel.metrics_logger.log_prompt.call_args[1]

            assert call_args["prompt_category"] == "pattern_extraction"
            assert call_args["amendment_id"] == "amend_456"

    def test_remove_code_blocks(self, mock_config, mock_dependencies):
        """Test _remove_code_blocks removes code block markers from responses."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Test cases for responses that might be wrapped in code blocks
            test_cases = [
                # XML wrapped in code blocks with language tag
                ("```xml\n<root>content</root>\n```", "<root>content</root>"),
                # XML wrapped without language tag
                ("```\n<root>content</root>\n```", "\n<root>content</root>"),
                # Plain XML without code blocks
                ("<root>content</root>", "<root>content</root>"),
                # Multi-line XML with language tag
                ("```xml\n<root>\n  <child>value</child>\n</root>\n```", "<root>\n  <child>value</child>\n</root>"),
                # Empty response
                ("", ""),
                # Just code blocks - ending ``` remains because no \n before it
                ("```xml\n```", "```"),
                # Empty code block with newline before closing
                ("```xml\n\n```", ""),
                # Response with trailing newline and backticks
                ("<root>content</root>\n```", "<root>content</root>"),
                # Mixed content - only cleans wrapper, not content
                (
                    "```xml\n<example>Some text\n```\ncode\n```</example>\n```",
                    "<example>Some text\n```\ncode\n```</example>",
                ),
                # Code block at start only
                ("```xml\n<root>content</root>", "<root>content</root>"),
                # Code block at end only with newline
                ("<root>content</root>\n```", "<root>content</root>"),
            ]

            for input_text, expected in test_cases:
                result = kernel._remove_code_blocks(input_text)
                assert result == expected, f"Failed for input: {repr(input_text)}, got: {repr(result)}"

    def test_fix_escaped_newlines(self, mock_config, mock_dependencies):
        """Test _fix_escaped_newlines converts escaped newlines in CSV data."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Test with CSV data containing escaped newlines
            input_text = (
                "source_eid,source,type_of_amendment\\n" "sec_135,s. 135,insertion\\n" "sec_136,s. 136,deletion"
            )
            expected = "source_eid,source,type_of_amendment\n" "sec_135,s. 135,insertion\n" "sec_136,s. 136,deletion"

            result = kernel._fix_escaped_newlines(input_text)
            assert result == expected

    def test_clean_response_combined(self, mock_config, mock_dependencies):
        """Test _clean_response handles both code blocks and escaped newlines."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Test case 1: XML with code blocks (no escaped newlines)
            xml_input = "```xml\n<amendment>test</amendment>\n```"
            xml_expected = "<amendment>test</amendment>"
            xml_result = kernel._clean_response(xml_input)
            assert xml_result == xml_expected

            # Test case 2: CSV with escaped newlines (no code blocks)
            csv_input = "header1,header2\\nvalue1,value2\\nvalue3,value4"
            csv_expected = "header1,header2\nvalue1,value2\nvalue3,value4"
            csv_result = kernel._clean_response(csv_input)
            assert csv_result == csv_expected

            # Test case 3: Combined - CSV with both code blocks AND escaped newlines
            combined_input = (
                "```csv\n"
                "source_eid,source,type_of_amendment\\n"
                "sec_135,s. 135,insertion\\n"
                "sec_136,s. 136,deletion\n"
                "```"
            )
            combined_expected = (
                "source_eid,source,type_of_amendment\n" "sec_135,s. 135,insertion\n" "sec_136,s. 136,deletion"
            )
            combined_result = kernel._clean_response(combined_input)
            assert combined_result == combined_expected

    def test_event_loop_setup(self, mock_config, mock_dependencies):
        """Test that event loop is properly set up in background thread."""
        mock_loop = Mock()
        mock_thread = Mock()

        with patch("asyncio.new_event_loop", return_value=mock_loop):
            with patch("threading.Thread", return_value=mock_thread) as mock_thread_class:
                kernel = LLMKernel(mock_config)

                # Verify loop created
                assert kernel.loop == mock_loop

                # Verify thread created as daemon
                mock_thread_class.assert_called_once()
                call_kwargs = mock_thread_class.call_args[1]
                assert call_kwargs["daemon"] is True
                assert "target" in call_kwargs

                # Verify thread started
                mock_thread.start.assert_called_once()

    def test_run_loop_forever(self, mock_config, mock_dependencies):
        """Test the event loop runner function."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            mock_loop = Mock()
            kernel.loop = mock_loop

            with patch("asyncio.set_event_loop") as mock_set_loop:
                # Call the loop runner
                kernel._run_loop_forever()

                # Verify loop was set and run
                mock_set_loop.assert_called_once_with(mock_loop)
                mock_loop.run_forever.assert_called_once()

    def test_setup_services_none_enabled(self, mock_config, mock_dependencies):
        """Test that exception is raised when no LLM services are enabled."""
        # Disable both services
        mock_dependencies["llm_config"].enable_azure_openai = False
        mock_dependencies["llm_config"].enable_aws_bedrock = False

        with pytest.raises(RuntimeError, match="No LLM services are enabled in configuration"):
            with patch("threading.Thread"):
                LLMKernel(mock_config)

    def test_setup_bedrock_clients_failed_with_bedrock_enabled(self, mock_config, mock_dependencies):
        """Test that exception is raised when Bedrock is enabled but clients fail to initialise."""
        # Configure for Bedrock only
        mock_dependencies["llm_config"].enable_azure_openai = False
        mock_dependencies["llm_config"].enable_aws_bedrock = True

        # Make client initialization fail
        mock_dependencies["llm_config"].get_bedrock_clients.return_value = (None, None)

        with pytest.raises(RuntimeError, match="AWS Bedrock clients could not be initialised"):
            with patch("threading.Thread"):
                LLMKernel(mock_config)

    def test_setup_bedrock_clients_failed_with_bedrock_disabled(self, mock_config, mock_dependencies):
        """Test that only warning is logged when Bedrock clients fail but Azure is enabled."""
        # Configure with Azure enabled, Bedrock disabled
        mock_dependencies["llm_config"].enable_azure_openai = True
        mock_dependencies["llm_config"].enable_aws_bedrock = False

        # Make bedrock client initialization return None (simulating optional Bedrock setup)
        mock_dependencies["llm_config"].get_bedrock_clients.return_value = (None, None)

        with patch("threading.Thread"):
            with patch("app.kernel.llm_kernel.BedrockChatCompletion") as mock_bedrock:
                with patch("app.kernel.llm_kernel.logger") as mock_logger:
                    # This should succeed because Azure is enabled
                    LLMKernel(mock_config)

                    # Bedrock setup should not be called since it's disabled
                    mock_bedrock.assert_not_called()

                    # No warning should be logged since Bedrock wasn't being set up
                    mock_logger.warning.assert_not_called()

    def test_setup_aws_bedrock_with_failed_clients_but_not_enabled(self, mock_config, mock_dependencies):
        """Test the warning path when AWS Bedrock setup is attempted but fails while not being required."""
        # Configure for Azure only initially
        mock_dependencies["llm_config"].enable_azure_openai = True
        mock_dependencies["llm_config"].enable_aws_bedrock = False

        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Now manually trigger the Bedrock setup with failed clients
            mock_dependencies["llm_config"].get_bedrock_clients.return_value = (None, None)

            # Temporarily set enable_aws_bedrock to False to hit the else branch
            kernel.llm_config.enable_aws_bedrock = False

            with patch("app.kernel.llm_kernel.logger") as mock_logger:
                # Call the setup method directly
                kernel._setup_aws_bedrock()

                # Should log warning, not raise exception
                mock_logger.warning.assert_called_once_with("AWS Bedrock clients could not be initialised")

    def test_setup_both_services_enabled(self, mock_config, mock_dependencies):
        """Test successful setup when both services are enabled."""
        # Enable both services
        mock_dependencies["llm_config"].enable_azure_openai = True
        mock_dependencies["llm_config"].enable_aws_bedrock = True

        # Setup successful Bedrock clients
        mock_runtime = Mock()
        mock_client = Mock()
        mock_dependencies["llm_config"].get_bedrock_clients.return_value = (mock_runtime, mock_client)

        with patch("app.kernel.llm_kernel.AzureChatCompletion") as mock_azure:
            with patch("app.kernel.llm_kernel.BedrockChatCompletion") as mock_bedrock:
                with patch("threading.Thread"):
                    LLMKernel(mock_config)

                    # Both services should be configured
                    mock_azure.assert_called_once()
                    mock_bedrock.assert_called_once()

                    # Kernel should have both services added
                    assert mock_dependencies["kernel"].add_service.call_count == 2

    def test_calculate_duration_ms_valid_timestamps(self, mock_config, mock_dependencies):
        """Test _calculate_duration_ms with valid timestamps."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Test with valid ISO timestamps
            start_ts = "2024-01-01T10:00:00.000"
            end_ts = "2024-01-01T10:00:05.500"

            result = kernel._calculate_duration_ms(start_ts, end_ts)

            assert result == 5500  # 5.5 seconds = 5500 milliseconds

            # Test with microseconds
            start_ts = "2024-01-01T10:00:00.123456"
            end_ts = "2024-01-01T10:00:01.623456"

            result = kernel._calculate_duration_ms(start_ts, end_ts)

            assert result == 1500  # 1.5 seconds = 1500 milliseconds

    def test_calculate_duration_ms_invalid_timestamps(self, mock_config, mock_dependencies):
        """Test _calculate_duration_ms with invalid timestamp formats."""
        with patch("threading.Thread"):
            kernel = LLMKernel(mock_config)

            # Test various invalid timestamp formats
            test_cases = [
                ("invalid-timestamp", "2024-01-01T10:00:00"),  # Invalid start
                ("2024-01-01T10:00:00", "not-a-timestamp"),  # Invalid end
                ("", "2024-01-01T10:00:00"),  # Empty start
                ("2024-01-01T10:00:00", ""),  # Empty end
                ("malformed", "malformed"),  # Both invalid
                ("2024-13-01T10:00:00", "2024-01-01T10:00:00"),  # Invalid month
                ("2024-01-32T10:00:00", "2024-01-01T10:00:00"),  # Invalid day
            ]

            for start_ts, end_ts in test_cases:
                result = kernel._calculate_duration_ms(start_ts, end_ts)
                assert result is None, f"Expected None for timestamps: {start_ts}, {end_ts}"


class TestGetKernel:
    """Tests for the get_kernel singleton function."""

    def teardown_method(self):
        """Reset global state after each test."""
        global llm_kernel
        llm_kernel = None

    @patch("app.kernel.llm_kernel.LLMKernel")
    @patch("app.kernel.llm_kernel.Config")
    def test_get_kernel_creates_singleton(self, mock_config_class, mock_kernel_class):
        """Test that get_kernel creates a singleton instance."""
        mock_kernel_instance = Mock()
        mock_kernel_class.return_value = mock_kernel_instance

        # First call creates instance
        kernel1 = get_kernel()
        assert kernel1 == mock_kernel_instance
        mock_kernel_class.assert_called_once()

        # Second call returns same instance
        kernel2 = get_kernel()
        assert kernel2 == kernel1
        # Should still only be called once
        assert mock_kernel_class.call_count == 1

    @patch("app.kernel.llm_kernel.LLMKernel")
    @patch("app.kernel.llm_kernel.Config")
    def test_get_kernel_thread_safety(self, mock_config_class, mock_kernel_class):
        """Test that get_kernel is thread-safe."""
        import app.kernel.llm_kernel

        # Reset the global kernel state
        app.kernel.llm_kernel.llm_kernel = None

        creation_count = 0
        created_instance = Mock()
        created_instance.id = "singleton"

        def slow_creation(*args):
            nonlocal creation_count
            # Simulate slow initialisation
            import time

            time.sleep(0.01)
            creation_count += 1
            return created_instance

        mock_kernel_class.side_effect = slow_creation

        # Try to create kernel from multiple threads
        results = []
        threads = []

        def get_kernel_thread():
            # Import inside thread to ensure proper module state
            from app.kernel.llm_kernel import get_kernel

            results.append(get_kernel())

        for _ in range(5):
            thread = threading.Thread(target=get_kernel_thread)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should get the same instance
        assert len(results) == 5
        assert all(r == created_instance for r in results)
        # Should only create one instance
        assert creation_count == 1
