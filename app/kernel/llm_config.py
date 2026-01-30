# app/kernel/llm_config.py
"""
Configuration and credentials management for LLM services.

Manages configuration for multiple LLM providers including Azure OpenAI and AWS Bedrock.
Handles credential retrieval from AWS Secrets Manager and provides a unified interface
for service configuration across the application.

NOTE:
Real service endpoints, AWS regions, profiles, and secret identifiers have been
sanitised in this repository. Placeholder values are used instead.
"""

import json
import logging
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ProfileNotFound
from botocore.config import Config as BotocoreConfig

logger = logging.getLogger(__name__)


class LLMConfig:
    """
    Configuration and credentials manager for LLM services.

    This class centralises all LLM service configuration including:
    - Service enablement flags
    - API endpoints and model configurations
    - Credential management via AWS Secrets Manager
    - HTTP timeout and retry configurations

    The class uses lazy initialisation for AWS sessions to avoid unnecessary
    connections when services are disabled.
    """

    def __init__(self):
        """
        Initialise LLM configuration with default values.

        Sets up configuration for both Azure OpenAI and AWS Bedrock services,
        though typically only one service is enabled at a time.
        """
        # Service enablement flags
        self.enable_azure_openai = True
        self.enable_aws_bedrock = False

        # Azure OpenAI configuration
        # NOTE: Endpoint value sanitised
        self.azure_endpoint = "https://example.openai.azure.com/"
        self.azure_model_deployment_name = "gpt-4o"
        self.azure_api_version = "2024-08-01-preview"

        # AWS general settings
        # NOTE: Region and profile values sanitised
        self.aws_region = "example-region"
        self.aws_profile = "example-profile"

        # Secrets Manager configuration
        self.secret_name = "azure-openai-api-key"

        # Bedrock specific configuration
        # NOTE: Model ID value sanitised
        self.bedrock_model_id = "anthropic.claude-3-5-sonnet"
        self.bedrock_service_id = "bedrock-claude"

        # HTTP timeout settings (in seconds)
        self.http_connect_timeout = 10
        self.http_read_timeout = 300  # 5 minutes for long LLM responses

        # Rate limit handling with exponential backoff
        self.max_retries = 5
        self.retry_backoff = [1, 2, 4, 8, 16]  # Seconds between retries

        # Model-specific completion token limits
        self.azure_openai_max_completion_tokens = 16384  # GPT-4o limit
        self.bedrock_max_completion_tokens = 8192  # Claude 3.5 Sonnet limit

        # Lazy-initialised AWS session
        self._session = None

    # ==================== Public Interface Methods ====================

    def get_active_service_id(self) -> Optional[str]:
        """
        Get the ID of the currently active LLM service.

        Returns the service ID for the enabled service, prioritising
        Azure OpenAI if both services are enabled.

        Returns:
            Service ID string if a service is enabled, None otherwise
        """
        if self.enable_azure_openai:
            return self.azure_model_deployment_name
        elif self.enable_aws_bedrock:
            return self.bedrock_service_id
        return None

    def get_azure_api_key(self) -> Optional[str]:
        """
        Fetch the Azure OpenAI API key from AWS Secrets Manager.

        Retrieves the API key only if Azure OpenAI is enabled and AWS session
        is available. Handles multiple key formats in the secret for flexibility.

        Returns:
            API key string if successful, None if service disabled or on error
        """
        if not self.enable_azure_openai:
            return None

        if not self.session:
            logger.warning("No AWS session available for retrieving Azure API key")
            return None

        try:
            secrets_client = self.session.client("secretsmanager", region_name=self.aws_region)
            response = secrets_client.get_secret_value(SecretId=self.secret_name)

            # Parse secret data
            secret_data = self._parse_secret_response(response)

            # Try multiple key formats for flexibility
            api_key = (
                secret_data.get("azure-openai-api-key-1")
                or secret_data.get("azure-openai-api-key-2")
                or secret_data.get("azure-openai-api-key")
            )

            if not api_key:
                logger.error("No valid API key found in secret data")

            return api_key

        except Exception as e:
            logger.error(f"Error retrieving Azure OpenAI API key: {e}")
            return None

    def get_bedrock_clients(self) -> Tuple[Optional[boto3.client], Optional[boto3.client]]:
        """
        Get configured Bedrock runtime and management clients.

        Creates boto3 clients for AWS Bedrock with appropriate timeout configurations.
        Only creates clients if Bedrock is enabled and AWS session is available.

        Returns:
            Tuple of (runtime_client, management_client) or (None, None) if unavailable
        """
        if not self.enable_aws_bedrock:
            return None, None

        if not self.session:
            logger.warning("No AWS session available for creating Bedrock clients")
            return None, None

        try:
            # Configure timeouts for long-running LLM operations
            config = BotocoreConfig(connect_timeout=self.http_connect_timeout, read_timeout=self.http_read_timeout)

            # Create clients
            runtime_client = self.session.client("bedrock-runtime", region_name=self.aws_region, config=config)

            management_client = self.session.client("bedrock", region_name=self.aws_region, config=config)

            return runtime_client, management_client

        except Exception as e:
            logger.error(f"Error creating Bedrock clients: {e}")
            return None, None

    def get_max_completion_tokens(self) -> int:
        """
        Get the maximum completion tokens for the currently active LLM service.

        Returns:
            Maximum number of completion tokens allowed by the active model
        """
        if self.enable_azure_openai:
            return self.azure_openai_max_completion_tokens
        elif self.enable_aws_bedrock:
            return self.bedrock_max_completion_tokens
        else:
            # Default to Azure OpenAI limit as fallback
            return self.azure_openai_max_completion_tokens

    # ==================== AWS Session Management ====================

    @property
    def session(self) -> Optional[boto3.Session]:
        """
        Lazy initialisation of boto3 session.

        Creates session on first access to avoid unnecessary AWS connections
        when services are disabled.

        Returns:
            Boto3 session if successful, None on error
        """
        if self._session is None:
            self._session = self._create_boto3_session()
        return self._session

    # ==================== Private Helper Methods ====================

    def _create_boto3_session(self) -> Optional[boto3.Session]:
        """
        Create a boto3 session with fallback for IAM role-based credentials.

        Attempts to use configured AWS profile first, falling back to IAM role
        credentials if profile is not available (e.g., in EC2/ECS environments).

        Returns:
            Boto3 session if successful, None on error
        """
        try:
            # Try profile-based authentication first
            logger.info(f"Attempting to create AWS session with profile: {self.aws_profile}")
            return boto3.Session(profile_name=self.aws_profile, region_name=self.aws_region)

        except ProfileNotFound:
            logger.info("Profile not found. Falling back to IAM role session.")
            try:
                # Use IAM role or environment credentials
                return boto3.Session(region_name=self.aws_region)

            except Exception as e:
                logger.error(f"Failed to create AWS session with IAM role: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error creating AWS session: {e}")
            return None

    def _parse_secret_response(self, response: dict) -> dict:
        """
        Parse the response from AWS Secrets Manager.

        Handles both string and binary secret formats.

        Args:
            response: Response dict from get_secret_value

        Returns:
            Parsed secret data as dictionary

        Raises:
            json.JSONDecodeError: If secret data is not valid JSON
        """
        if "SecretString" in response:
            return json.loads(response["SecretString"])
        else:
            # Handle binary secrets
            return json.loads(response["SecretBinary"].decode("utf-8"))
