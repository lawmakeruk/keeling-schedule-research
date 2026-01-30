# app/kernel/llm_kernel.py
"""
An implementation of Microsoft's semantic kernel to handle requests to LLM providers.

Uses the config object to configure the kernel.
Handles all calls to LLM providers and uses yaml prompts as plugins to Kernel.
Logs the input prompt and separately logs the output information.
"""
import asyncio
import json
import re
import threading
import uuid
from datetime import datetime
from typing import Optional, Union

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.connectors.ai.bedrock import BedrockChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage

from .llm_config import LLMConfig
from .llm_retry_handler import LLMRetryHandler
from .prompt_parser import create_chat_messages, extract_execution_settings
from ..config import Config
from ..benchmarking.metrics_logger import MetricsLogger
from ..logging.debug_logger import get_logger, event, bind, EventType as EVT

logger = get_logger(__name__)


class LLMKernel:
    """
    Central handler for all LLM interactions in the Keeling Schedule Service.

    Manages connections to multiple LLM providers (Azure OpenAI, AWS Bedrock),
    handles retry logic, and provides comprehensive logging of all interactions.
    Uses a dedicated async event loop in a background thread for non-blocking operations.
    """

    def __init__(self, config: Config):
        """
        Initialise the Kernel with required dependencies.

        Args:
            config: Application configuration object
        """
        self.config = config
        self.llm_config = LLMConfig()
        self.kernel = Kernel()

        # Setup components
        self._setup_plugins()
        self._setup_services()
        self._setup_logging()
        self._setup_retry_handler()
        self._setup_event_loop()

        # Track request count for debugging
        self.request_count = 0

    # ==================== Public Interface Methods ====================

    def run_inference(
        self,
        prompt_name: str,
        schedule_id: str,
        amendment_id: Optional[str] = None,
        candidate_eid: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Synchronous wrapper for async inference with logging.

        This is the main entry point for LLM calls throughout the application.
        Handles prompt execution, retry logic, and comprehensive logging.

        Args:
            prompt_name: Name of the prompt to execute
            schedule_id: ID of the schedule being processed
            amendment_id: Optional amendment ID for logging
            candidate_eid: Optional candidate element ID for logging
            **kwargs: Template variables for the prompt

        Returns:
            Cleaned assistant response text
        """
        # Generate prompt_id at the start
        prompt_id = str(uuid.uuid4())

        with bind(
            schedule_id=schedule_id,
            amendment_id=amendment_id,
            prompt_id=prompt_id,
            candidate_eid=candidate_eid,
        ):
            # Validate prompt exists
            if prompt_name not in self.plugin:
                logger.error(f"Prompt {prompt_name} not found in plugin.")
                return "Prompt name not recognised"

            prompt = self.plugin[prompt_name]

            # Build serialised copy of chat messages
            messages = create_chat_messages(prompt, **kwargs)
            prompt_body = json.dumps(messages, separators=(",", ":"))

            start_ts = datetime.utcnow().isoformat()

            event(
                logger,
                EVT.LLM_REQUEST,
                "Starting LLM inference",
                prompt_name=prompt_name,
                request_count=self.request_count + 1,
                prompt_body=prompt_body,
            )

            # Run async operation on background loop
            coro = self.run_inference_async(prompt, prompt_id, candidate_eid, **kwargs)
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            result = future.result()  # Blocks until complete

            end_ts = datetime.utcnow().isoformat()

            # Process result
            if isinstance(result, ChatMessageContent):
                assistant_text = result.content
                metadata = result.metadata or {}

                # Extract token usage for event
                prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(metadata)

                event(
                    logger,
                    EVT.LLM_RESPONSE,
                    "LLM inference completed",
                    prompt_name=prompt_name,
                    response_length=len(assistant_text) if assistant_text else 0,
                    duration_ms=self._calculate_duration_ms(start_ts, end_ts),
                    assistant_text=assistant_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                # Log if enabled
                if self.log_prompts:
                    self._log_interaction(
                        prompt_name,
                        schedule_id,
                        amendment_id,
                        prompt_id,
                        kwargs,
                        start_ts,
                        end_ts,
                        prompt,
                        assistant_text,
                        metadata,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )

                # ABLATION: Response post-processing disabled
                # return self._clean_response(assistant_text)
                return assistant_text
            else:
                # Error message from retry handler
                event(logger, EVT.LLM_RESPONSE, "LLM inference failed", prompt_name=prompt_name, error=str(result))
                return str(result)

    async def run_inference_async(
        self, prompt: KernelFunction, prompt_id: str, candidate_eid: Optional[str] = None, **kwargs
    ) -> Union[ChatMessageContent, str]:
        """
        Execute prompt with retry handling.

        Args:
            prompt: The kernel function to execute
            prompt_id: Unique identifier for this prompt execution
            candidate_eid: Optional candidate element ID for context
            **kwargs: Template variables for the prompt

        Returns:
            ChatMessageContent on success, error string on failure
        """
        self.request_count += 1

        async def _execute():
            # Parse prompt to extract messages
            messages = create_chat_messages(prompt, **kwargs)

            # Get active service
            service_id = self.llm_config.get_active_service_id()
            if not service_id:
                raise RuntimeError("No LLM service is enabled")

            chat_service = self.kernel.get_service(service_id)

            # Build chat history
            history = ChatHistory()
            for m in messages:
                history.add_message(ChatMessageContent(role=m["role"], content=m["content"]))

            # Extract execution settings from the prompt
            settings_dict = extract_execution_settings(prompt, service_id)

            # Create PromptExecutionSettings with the extracted values
            settings = PromptExecutionSettings(
                service_id=service_id, **settings_dict  # This will include max_tokens, temperature, top_p, etc.
            )

            # Log the settings being used for debugging
            event(
                logger,
                EVT.LLM_SETTINGS,
                f"Using execution settings for {service_id}",
                service_id=service_id,
                max_tokens=settings_dict.get("max_tokens"),
                temperature=settings_dict.get("temperature"),
                top_p=settings_dict.get("top_p"),
            )

            # Execute chat completion with proper settings
            return await chat_service.get_chat_message_content(chat_history=history, settings=settings)

        # Execute with retry handling
        return await self.retry_handler.execute_with_retry(
            _execute,
            prompt_name=prompt.name,
            request_count=self.request_count,
            prompt_id=prompt_id,
            candidate_eid=candidate_eid,
        )

    # ==================== Setup Methods ====================

    def _setup_plugins(self) -> None:
        """Load YAML prompt templates as plugins."""
        self.plugin_path = self.config.PROMPTS_PATH
        self.plugin = self.kernel.add_plugin(plugin_name="prompts", parent_directory=self.plugin_path)
        logger.info(f"Loaded prompts from {self.plugin_path}")

    def _setup_services(self) -> None:
        """Initialise LLM services based on configuration."""
        # Setup Azure OpenAI if enabled
        if self.llm_config.enable_azure_openai:
            self._setup_azure_openai()

        # Setup AWS Bedrock if enabled
        if self.llm_config.enable_aws_bedrock:
            self._setup_aws_bedrock()

        # Verify at least one service is configured
        if not self.llm_config.enable_azure_openai and not self.llm_config.enable_aws_bedrock:
            raise RuntimeError("No LLM services are enabled in configuration")

    def _setup_azure_openai(self) -> None:
        """Configure Azure OpenAI service."""
        api_key = self.llm_config.get_azure_api_key()
        if not api_key:
            raise RuntimeError("Azure OpenAI API key could not be retrieved.")

        self.kernel.add_service(
            AzureChatCompletion(
                service_id=self.llm_config.azure_model_deployment_name,
                api_key=api_key,
                endpoint=self.llm_config.azure_endpoint,
                deployment_name=self.llm_config.azure_model_deployment_name,
                api_version=self.llm_config.azure_api_version,
            )
        )
        logger.info("Azure OpenAI service configured successfully")

    def _setup_aws_bedrock(self) -> None:
        """Configure AWS Bedrock service."""
        runtime_client, client = self.llm_config.get_bedrock_clients()
        if runtime_client and client:
            self.kernel.add_service(
                BedrockChatCompletion(
                    service_id=self.llm_config.bedrock_service_id,
                    model_id=self.llm_config.bedrock_model_id,
                    runtime_client=runtime_client,
                    client=client,
                )
            )
            logger.info("AWS Bedrock service configured successfully")
        else:
            if self.llm_config.enable_aws_bedrock:
                raise RuntimeError("AWS Bedrock clients could not be initialised")
            else:
                logger.warning("AWS Bedrock clients could not be initialised")

    def _setup_logging(self) -> None:
        """Initialise logging components."""
        self.log_prompts = self.config.LOG_PROMPTS
        self.metrics_logger = MetricsLogger(
            enable_aws_bedrock=self.llm_config.enable_aws_bedrock,
            bedrock_service_id=self.llm_config.bedrock_service_id,
            bedrock_model_id=self.llm_config.bedrock_model_id,
            enable_azure_openai=self.llm_config.enable_azure_openai,
            azure_model_deployment_name=self.llm_config.azure_model_deployment_name,
        )

    def _setup_retry_handler(self) -> None:
        """Initialise retry handler with exponential backoff."""
        self.retry_handler = LLMRetryHandler(self.llm_config.max_retries, self.llm_config.retry_backoff)

    def _setup_event_loop(self) -> None:
        """Create dedicated event loop in background thread."""
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop_forever, daemon=True, name="LLMKernel-EventLoop")
        self.loop_thread.start()
        logger.debug("Started async event loop in background thread")

    # ==================== Private Helper Methods ====================

    def _run_loop_forever(self) -> None:
        """Run the event loop forever in dedicated thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _log_interaction(
        self,
        prompt_name: str,
        schedule_id: str,
        amendment_id: Optional[str],
        prompt_id: str,
        input_params: dict,
        start_ts: str,
        end_ts: str,
        prompt: KernelFunction,
        assistant_text: str,
        metadata: dict,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        """
        Log comprehensive details about an LLM interaction.

        Extracts token usage, costs, and categorises prompts for analysis.
        """
        # Classify prompt type
        prompt_category = self._classify_prompt(prompt_name)

        # Extract model ID
        model_id = metadata.get("model") or metadata.get("id") or self.llm_config.get_active_service_id()

        # Log via MetricsLogger
        self.metrics_logger.log_prompt(
            prompt_id=prompt_id,
            prompt=prompt,
            prompt_output=assistant_text,
            input_parameters={
                **input_params,
                "extracted_prompt_tokens": prompt_tokens,
                "extracted_completion_tokens": completion_tokens,
                "extracted_total_tokens": total_tokens,
                "extracted_model_id": model_id,
            },
            schedule_id=schedule_id,
            prompt_start_ts=start_ts,
            prompt_end_ts=end_ts,
            amendment_id=amendment_id,
            prompt_category=prompt_category,
        )

    def _classify_prompt(self, prompt_name: str) -> Optional[str]:
        """Classify prompt into categories for analysis."""
        if "TableOfAmendments" in prompt_name:
            return "identification"
        elif any(x in prompt_name for x in ("ApplyInsertion", "ApplyDeletion", "ApplySubstitution")):
            return "application"
        elif "IdentifyAmendmentReferences" in prompt_name:
            return "preprocessing"
        elif "ExtractEachPlacePattern" in prompt_name:
            return "pattern_extraction"
        return None

    def _extract_token_usage(self, metadata: dict) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token usage information from response metadata."""
        usage_obj = metadata.get("usage")
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if isinstance(usage_obj, CompletionUsage):
            prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
            completion_tokens = getattr(usage_obj, "completion_tokens", None)

        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens

    def _clean_response(self, response: str) -> str:
        """
        Clean up LLM response by removing code block markers and fixing formatting issues.

        Args:
            response (str): Raw response text from the LLM that may contain formatting issues

        Returns:
            str: Cleaned response with code block markers removed and escaped newlines fixed
        """
        cleaned = self._remove_code_blocks(response)
        cleaned = self._fix_escaped_newlines(cleaned)
        return cleaned

    def _remove_code_blocks(self, response: str) -> str:
        """
        Remove markdown code block syntax that LLMs sometimes add.

        Args:
            response (str): Response text that may contain markdown code block markers

        Returns:
            str: Response text with code block markers (``` and language identifiers) removed
        """
        cleaned = re.sub(r"^```(?:[a-z]+\n)?", "", response)
        cleaned = re.sub(r"\n```$", "", cleaned)
        return cleaned

    def _fix_escaped_newlines(self, response: str) -> str:
        """
        Convert escaped newlines to actual newlines.

        Args:
            response (str): Response text that may contain literal '\n' strings instead of newlines

        Returns:
            str: Response text with escaped newlines (\n) converted to actual newline characters
        """
        return response.replace("\\n", "\n")

    def _calculate_duration_ms(self, start_ts: str, end_ts: str) -> Optional[int]:
        """
        Calculate duration in milliseconds between two ISO timestamps.

        Args:
            start_ts: Start timestamp in ISO format (e.g., '2024-01-15T10:30:00.123456')
            end_ts: End timestamp in ISO format (e.g., '2024-01-15T10:30:05.789012')

        Returns:
            Duration in milliseconds as an integer, or None if timestamps are invalid
        """
        try:
            start = datetime.fromisoformat(start_ts)
            end = datetime.fromisoformat(end_ts)
            return int((end - start).total_seconds() * 1000)
        except ValueError:
            return None


# ==================== Module-level Singleton ====================

# Thread-safe singleton implementation
llm_kernel: Optional[LLMKernel] = None
_kernel_lock = threading.Lock()


def get_kernel() -> LLMKernel:
    """
    Thread-safe singleton to ensure only one LLM Kernel instance.

    Returns:
        The single LLMKernel instance for the application
    """
    global llm_kernel

    if llm_kernel is not None:
        return llm_kernel

    with _kernel_lock:
        if llm_kernel is None:
            llm_kernel = LLMKernel(Config())
        return llm_kernel