# tests/unit/kernel/test_llm_config.py
"""
Unit tests for the LLMConfig class.

Note: environment-specific values (endpoints, regions, profiles) have been
sanitised for publication.
"""

import json
from unittest.mock import Mock, patch
from botocore.exceptions import ProfileNotFound
from botocore.config import Config as BotocoreConfig

from app.kernel.llm_config import LLMConfig


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_init_default_configuration(self):
        """Test that LLMConfig initialises with correct configuration structure."""
        config = LLMConfig()

        # Configuration flags - at least one should be enabled
        assert isinstance(config.enable_azure_openai, bool)
        assert isinstance(config.enable_aws_bedrock, bool)
        assert config.enable_azure_openai or config.enable_aws_bedrock  # At least one must be True

        # Azure OpenAI configuration should exist regardless of enablement
        assert config.azure_endpoint == "https://example.openai.azure.com/"
        assert config.azure_model_deployment_name == "gpt-4o"
        assert config.azure_api_version == "2024-08-01-preview"

        # AWS configuration
        assert config.aws_region == "example-region"
        assert config.aws_profile == "example-profile"

        # Secrets Manager configuration
        assert config.secret_name == "azure-openai-api-key"

        # Bedrock specific configuration
        assert config.bedrock_model_id == "anthropic.claude-3-5-sonnet"
        assert config.bedrock_service_id == "bedrock-claude"

        # HTTP timeout settings
        assert config.http_connect_timeout == 10
        assert config.http_read_timeout == 300

        # Retry configuration
        assert config.max_retries == 5
        assert config.retry_backoff == [1, 2, 4, 8, 16]

        # Model-specific token limits
        assert config.azure_openai_max_completion_tokens == 16384
        assert config.bedrock_max_completion_tokens == 8192

        # AWS session should not be initialised yet (lazy loading)
        assert config._session is None

    def test_session_lazy_initialisation(self):
        """Test that boto3 session is created lazily on first access."""
        config = LLMConfig()

        # Session not created yet
        assert config._session is None

        # Mock _create_boto3_session
        mock_session = Mock()
        with patch.object(config, "_create_boto3_session", return_value=mock_session):
            # Access session property
            session = config.session

            # Should create and return session
            assert session == mock_session
            assert config._session == mock_session

            # Second access should return same session without calling create again
            session2 = config.session
            assert session2 == mock_session
            config._create_boto3_session.assert_called_once()

    @patch("boto3.Session")
    def test_create_boto3_session_with_profile(self, mock_boto_session):
        """Test boto3 session creation with AWS profile."""
        config = LLMConfig()
        mock_session_instance = Mock()
        mock_boto_session.return_value = mock_session_instance

        session = config._create_boto3_session()

        # Should create session with profile
        mock_boto_session.assert_called_once_with(profile_name="example-profile", region_name="example-region")
        assert session == mock_session_instance

    @patch("boto3.Session")
    @patch("app.kernel.llm_config.logger")
    def test_create_boto3_session_fallback_to_iam(self, mock_logger, mock_boto_session):
        """Test boto3 session creation falls back to IAM role when profile not found."""
        config = LLMConfig()

        # First call raises ProfileNotFound
        mock_boto_session.side_effect = [ProfileNotFound(profile="example-profile"), Mock()]  # Second call succeeds

        config._create_boto3_session()

        # Should try profile first, then fallback
        assert mock_boto_session.call_count == 2
        mock_boto_session.assert_any_call(profile_name="example-profile", region_name="example-region")
        mock_boto_session.assert_any_call(region_name="example-region")

        # Should log fallback message
        mock_logger.info.assert_any_call("Profile not found. Falling back to IAM role session.")

    def test_get_bedrock_clients_when_disabled(self):
        """Test get_bedrock_clients returns None when Bedrock is disabled."""
        config = LLMConfig()
        config.enable_aws_bedrock = False

        runtime_client, client = config.get_bedrock_clients()

        assert runtime_client is None
        assert client is None

    def test_get_bedrock_clients_when_no_session(self):
        """Test get_bedrock_clients returns None when no session available."""
        config = LLMConfig()
        config.enable_aws_bedrock = True
        config._session = None

        with patch.object(config, "_create_boto3_session", return_value=None):
            runtime_client, client = config.get_bedrock_clients()

        assert runtime_client is None
        assert client is None

    def test_get_bedrock_clients_success(self):
        """Test successful creation of Bedrock clients."""
        config = LLMConfig()
        config.enable_aws_bedrock = True

        # Mock session and clients
        mock_session = Mock()
        mock_runtime_client = Mock()
        mock_client = Mock()

        mock_session.client.side_effect = [mock_runtime_client, mock_client]
        config._session = mock_session

        runtime_client, client = config.get_bedrock_clients()

        # Verify clients created with correct config
        expected_config = BotocoreConfig(connect_timeout=10, read_timeout=300)

        assert mock_session.client.call_count == 2

        # Check runtime client call
        runtime_call = mock_session.client.call_args_list[0]
        assert runtime_call[0][0] == "bedrock-runtime"
        assert runtime_call[1]["region_name"] == "example-region"
        assert runtime_call[1]["config"].connect_timeout == expected_config.connect_timeout
        assert runtime_call[1]["config"].read_timeout == expected_config.read_timeout

        # Check standard client call
        client_call = mock_session.client.call_args_list[1]
        assert client_call[0][0] == "bedrock"
        assert client_call[1]["region_name"] == "example-region"

        assert runtime_client == mock_runtime_client
        assert client == mock_client

    def test_get_azure_api_key_when_disabled(self):
        """Test get_azure_api_key returns None when Azure is disabled."""
        config = LLMConfig()
        config.enable_azure_openai = False

        api_key = config.get_azure_api_key()

        assert api_key is None

    def test_get_azure_api_key_when_no_session(self):
        """Test get_azure_api_key returns None when no session available."""
        config = LLMConfig()
        config.enable_azure_openai = True
        config._session = None

        with patch.object(config, "_create_boto3_session", return_value=None):
            api_key = config.get_azure_api_key()

        assert api_key is None

    def test_get_azure_api_key_success_with_key1(self):
        """Test successful retrieval of Azure API key using key1."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock get_secret_value response
        mock_secrets_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {"azure-openai-api-key-1": "test-api-key-1", "azure-openai-api-key-2": "test-api-key-2"}
            )
        }

        config._session = mock_session

        api_key = config.get_azure_api_key()

        # Verify correct calls
        mock_session.client.assert_called_once_with("secretsmanager", region_name="example-region")
        mock_secrets_client.get_secret_value.assert_called_once_with(SecretId="azure-openai-api-key")

        assert api_key == "test-api-key-1"

    def test_get_azure_api_key_fallback_to_key2(self):
        """Test Azure API key retrieval falls back to key2 when key1 not present."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock response with only key2
        mock_secrets_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"azure-openai-api-key-2": "test-api-key-2"})
        }

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key == "test-api-key-2"

    def test_get_azure_api_key_binary_secret(self):
        """Test Azure API key retrieval from binary secret."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock binary secret response
        secret_data = {"azure-openai-api-key-1": "test-api-key-binary"}
        mock_secrets_client.get_secret_value.return_value = {"SecretBinary": json.dumps(secret_data).encode("utf-8")}

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key == "test-api-key-binary"

    @patch("app.kernel.llm_config.logger")
    def test_get_azure_api_key_exception_handling(self, mock_logger):
        """Test Azure API key retrieval handles exceptions gracefully."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock exception
        mock_secrets_client.get_secret_value.side_effect = Exception("Test error")

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key is None
        mock_logger.error.assert_called_once_with("Error retrieving Azure OpenAI API key: Test error")

    def test_get_active_service_id_azure_enabled(self):
        """Test get_active_service_id returns Azure service when enabled."""
        config = LLMConfig()
        config.enable_azure_openai = True
        config.enable_aws_bedrock = False

        service_id = config.get_active_service_id()

        assert service_id == "gpt-4o"

    def test_get_active_service_id_bedrock_enabled(self):
        """Test get_active_service_id returns Bedrock service when enabled."""
        config = LLMConfig()
        config.enable_azure_openai = False
        config.enable_aws_bedrock = True

        service_id = config.get_active_service_id()

        assert service_id == "bedrock-claude"

    def test_get_active_service_id_both_enabled(self):
        """Test get_active_service_id prioritises Azure when both enabled."""
        config = LLMConfig()
        config.enable_azure_openai = True
        config.enable_aws_bedrock = True

        service_id = config.get_active_service_id()

        # Azure takes precedence
        assert service_id == "gpt-4o"

    def test_get_active_service_id_none_enabled(self):
        """Test get_active_service_id returns None when no service enabled."""
        config = LLMConfig()
        config.enable_azure_openai = False
        config.enable_aws_bedrock = False

        service_id = config.get_active_service_id()

        assert service_id is None

    @patch("app.kernel.llm_config.logger")
    def test_get_azure_api_key_no_valid_key_in_secret(self, mock_logger):
        """Test Azure API key retrieval when secret contains no valid keys."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock response with no recognised key formats
        mock_secrets_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"some-other-key": "value", "unrelated-data": "test"})
        }

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key is None
        mock_logger.error.assert_called_once_with("No valid API key found in secret data")

    def test_get_azure_api_key_fallback_to_generic_key(self):
        """Test Azure API key retrieval falls back to generic key format."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock response with only generic key format
        mock_secrets_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"azure-openai-api-key": "test-api-key-generic"})
        }

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key == "test-api-key-generic"

    @patch("app.kernel.llm_config.logger")
    def test_get_bedrock_clients_exception_handling(self, mock_logger):
        """Test get_bedrock_clients handles exceptions gracefully."""
        config = LLMConfig()
        config.enable_aws_bedrock = True

        # Mock session that raises exception when creating client
        mock_session = Mock()
        mock_session.client.side_effect = Exception("Failed to create client")

        config._session = mock_session

        runtime_client, client = config.get_bedrock_clients()

        assert runtime_client is None
        assert client is None
        mock_logger.error.assert_called_once_with("Error creating Bedrock clients: Failed to create client")

    @patch("boto3.Session")
    @patch("app.kernel.llm_config.logger")
    def test_create_boto3_session_iam_role_failure(self, mock_logger, mock_boto_session):
        """Test boto3 session creation when both profile and IAM role fail."""
        config = LLMConfig()

        # Both attempts fail
        mock_boto_session.side_effect = [ProfileNotFound(profile="example-profile"), Exception("IAM role error")]

        session = config._create_boto3_session()

        assert session is None
        assert mock_boto_session.call_count == 2

        # Check all expected log messages
        mock_logger.info.assert_any_call("Attempting to create AWS session with profile: example-profile")
        mock_logger.info.assert_any_call("Profile not found. Falling back to IAM role session.")
        mock_logger.error.assert_called_with("Failed to create AWS session with IAM role: IAM role error")

    @patch("boto3.Session")
    @patch("app.kernel.llm_config.logger")
    def test_create_boto3_session_unexpected_error(self, mock_logger, mock_boto_session):
        """Test boto3 session creation with unexpected error."""
        config = LLMConfig()

        # Unexpected error on first attempt
        mock_boto_session.side_effect = Exception("Unexpected error")

        session = config._create_boto3_session()

        assert session is None
        mock_boto_session.assert_called_once_with(profile_name="example-profile", region_name="example-region")

        # Check log messages
        mock_logger.info.assert_called_once_with("Attempting to create AWS session with profile: example-profile")
        mock_logger.error.assert_called_once_with("Unexpected error creating AWS session: Unexpected error")

    def test_parse_secret_response_with_secret_string(self):
        """Test parsing secret response with SecretString."""
        config = LLMConfig()

        response = {"SecretString": json.dumps({"key": "value"})}

        result = config._parse_secret_response(response)

        assert result == {"key": "value"}

    def test_parse_secret_response_with_secret_binary(self):
        """Test parsing secret response with SecretBinary."""
        config = LLMConfig()

        secret_data = {"key": "binary-value"}
        response = {"SecretBinary": json.dumps(secret_data).encode("utf-8")}

        result = config._parse_secret_response(response)

        assert result == secret_data

    @patch("app.kernel.llm_config.logger")
    def test_get_azure_api_key_json_decode_error(self, mock_logger):
        """Test Azure API key retrieval with invalid JSON in secret."""
        config = LLMConfig()
        config.enable_azure_openai = True

        # Mock session and secrets client
        mock_session = Mock()
        mock_secrets_client = Mock()
        mock_session.client.return_value = mock_secrets_client

        # Mock response with invalid JSON
        mock_secrets_client.get_secret_value.return_value = {"SecretString": "invalid-json{{"}

        config._session = mock_session

        api_key = config.get_azure_api_key()

        assert api_key is None
        # Should log error about JSON decode failure
        assert mock_logger.error.call_count == 1
        assert "Error retrieving Azure OpenAI API key" in mock_logger.error.call_args[0][0]

    def test_get_max_completion_tokens_azure_enabled(self):
        """Test get_max_completion_tokens returns Azure limit when Azure is enabled."""
        config = LLMConfig()
        config.enable_azure_openai = True
        config.enable_aws_bedrock = False

        max_tokens = config.get_max_completion_tokens()

        assert max_tokens == 16384

    def test_get_max_completion_tokens_bedrock_enabled(self):
        """Test get_max_completion_tokens returns Bedrock limit when Bedrock is enabled."""
        config = LLMConfig()
        config.enable_azure_openai = False
        config.enable_aws_bedrock = True

        max_tokens = config.get_max_completion_tokens()

        assert max_tokens == 8192

    def test_get_max_completion_tokens_both_enabled(self):
        """Test get_max_completion_tokens prioritises Azure when both services enabled."""
        config = LLMConfig()
        config.enable_azure_openai = True
        config.enable_aws_bedrock = True

        max_tokens = config.get_max_completion_tokens()

        # Should return Azure limit as it takes precedence
        assert max_tokens == 16384

    def test_get_max_completion_tokens_none_enabled(self):
        """Test get_max_completion_tokens returns Azure limit as fallback when no service enabled."""
        config = LLMConfig()
        config.enable_azure_openai = False
        config.enable_aws_bedrock = False

        max_tokens = config.get_max_completion_tokens()

        # Should return Azure limit as default fallback
        assert max_tokens == 16384

    def test_get_max_completion_tokens_custom_limits(self):
        """Test get_max_completion_tokens with custom token limits."""
        config = LLMConfig()

        # Test with custom Azure limit
        config.enable_azure_openai = True
        config.enable_aws_bedrock = False
        config.azure_openai_max_completion_tokens = 32000  # Custom limit

        max_tokens = config.get_max_completion_tokens()
        assert max_tokens == 32000

        # Test with custom Bedrock limit
        config.enable_azure_openai = False
        config.enable_aws_bedrock = True
        config.bedrock_max_completion_tokens = 8192  # Custom limit

        max_tokens = config.get_max_completion_tokens()
        assert max_tokens == 8192
