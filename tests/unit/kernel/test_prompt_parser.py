# tests/unit/kernel/test_prompt_parser.py

from unittest.mock import MagicMock, patch
from semantic_kernel.functions import KernelFunction
from app.kernel.prompt_parser import parse_message_tags, create_chat_messages, get_prompt_template


def test_parse_message_tags_basic():
    """Test parsing simple message tags with different roles."""
    prompt_content = """
    <message role="system">You are a helpful assistant</message>
    <message role="user">What is the capital of France?</message>
    """

    messages = parse_message_tags(prompt_content)

    assert len(messages) == 2
    assert messages[0][0] == "system"
    assert messages[0][1] == "You are a helpful assistant"
    assert messages[1][0] == "user"
    assert messages[1][1] == "What is the capital of France?"


def test_parse_message_tags_complex_content():
    """Test parsing message tags with complex content including XML."""
    prompt_content = """
    <message role="system">
    You are an expert in applying amendments to legislation represented in Akoma Ntoso XML.
    An amendment is an instruction to make a single change to another piece of legislation.
    Respond with XML only, no markdown.
    </message>
    <message role="user">
    Original provision XML:
    <level>
        <num>(a)</num>
        <content><p>Text content</p></content>
    </level>
    </message>
    """

    messages = parse_message_tags(prompt_content)

    assert len(messages) == 2
    assert messages[0][0] == "system"
    assert "expert in applying amendments" in messages[0][1]
    assert messages[1][0] == "user"
    assert "<level>" in messages[1][1]


def test_parse_message_tags_empty():
    """Test parsing with no message tags."""
    prompt_content = "Plain text with no message tags"

    messages = parse_message_tags(prompt_content)

    assert len(messages) == 0


def test_get_prompt_template():
    """Test getting the template content from a KernelFunction."""
    # Test with prompt_template attribute
    mock_function1 = MagicMock()
    mock_function1.prompt_template = "Template content"
    assert get_prompt_template(mock_function1) == "Template content"

    # Test with description attribute
    mock_function2 = MagicMock()
    mock_function2.prompt_template = None
    mock_function2.description = "Description content"
    assert get_prompt_template(mock_function2) == "Description content"

    # Test with no attributes
    mock_function3 = MagicMock()
    mock_function3.prompt_template = None
    mock_function3.description = None
    assert get_prompt_template(mock_function3) == ""


@patch("app.kernel.prompt_parser.get_prompt_template")
@patch("app.kernel.prompt_parser.parse_message_tags")
def test_create_chat_messages(mock_parse, mock_get_template):
    """Test creating chat messages from a kernel function prompt."""
    # Mock the get_prompt_template function
    mock_get_template.return_value = "Template with {{$variable1}}"

    # Mock the parse_message_tags function
    mock_parse.return_value = [("system", "System message"), ("user", "User message")]

    # Create a mock KernelFunction
    mock_function = MagicMock(spec=KernelFunction)

    # Call the function
    messages = create_chat_messages(mock_function, variable1="value1")

    # Verify get_prompt_template was called with the right function
    mock_get_template.assert_called_once_with(mock_function)

    # Verify parse_message_tags was called with a string containing the substituted value
    mock_parse.assert_called_once()
    assert "value1" in mock_parse.call_args[0][0]

    # Verify the messages are in the correct format
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System message"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User message"


def test_integration_parse_and_create():
    """Integration test for parsing and creating chat messages."""
    # Create a mock KernelFunction with a custom prompt_template
    mock_function = MagicMock(spec=KernelFunction)
    mock_function.prompt_template = """
    <message role="system">You are an expert in applying amendments to legislation</message>
    <message role="user">Apply the amendment: {{$amendment_xml}} to {{$original_xml}}</message>
    """

    # Call create_chat_messages
    messages = create_chat_messages(
        mock_function, amendment_xml="<amendment>Test</amendment>", original_xml="<original>Content</original>"
    )

    # Verify the chat messages are in the expected format
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "expert in applying amendments" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "<amendment>Test</amendment>" in messages[1]["content"]
    assert "<original>Content</original>" in messages[1]["content"]


def test_get_prompt_template_with_template_attribute():
    """Test getting the template from a prompt with a template attribute."""
    # Create a nested template object
    template_obj = MagicMock()
    template_obj.template = "Template with nested attribute"

    # Create a mock KernelFunction
    mock_function = MagicMock(spec=KernelFunction)
    mock_function.prompt_template = template_obj

    assert get_prompt_template(mock_function) == "Template with nested attribute"


def test_get_prompt_template_with_string_representation():
    """Test getting the template from a prompt using string representation."""

    # Create a template object with a custom string representation
    class CustomTemplate:
        def __str__(self):
            return "Template from string representation"

    template_obj = CustomTemplate()

    # Create a mock KernelFunction
    mock_function = MagicMock(spec=KernelFunction)
    mock_function.prompt_template = template_obj

    assert get_prompt_template(mock_function) == "Template from string representation"


def test_extract_execution_settings_success():
    """Test successful extraction of execution settings for a specific service."""
    from app.kernel.prompt_parser import extract_execution_settings

    # Create mock execution settings with extension_data
    mock_settings_gpt = MagicMock()
    mock_settings_gpt.extension_data = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16384}

    mock_settings_bedrock = MagicMock()
    mock_settings_bedrock.extension_data = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192}

    # Create mock prompt with execution settings
    mock_prompt = MagicMock()
    mock_prompt.name = "TestPrompt"
    mock_prompt.prompt_execution_settings = {"gpt-4o": mock_settings_gpt, "bedrock-claude": mock_settings_bedrock}

    # Test extracting settings for gpt-4o
    gpt_settings = extract_execution_settings(mock_prompt, "gpt-4o")
    assert gpt_settings == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16384}

    # Test extracting settings for bedrock-claude
    bedrock_settings = extract_execution_settings(mock_prompt, "bedrock-claude")
    assert bedrock_settings == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8192}


@patch("app.kernel.prompt_parser.logger")
def test_extract_execution_settings_failure(mock_logger):
    """Test extraction of execution settings when settings are missing or invalid."""
    from app.kernel.prompt_parser import extract_execution_settings

    # Test 1: No prompt_execution_settings attribute
    mock_prompt_no_attr = MagicMock()
    mock_prompt_no_attr.name = "NoSettingsPrompt"
    del mock_prompt_no_attr.prompt_execution_settings  # Remove the attribute

    result = extract_execution_settings(mock_prompt_no_attr, "gpt-4o")
    assert result == {}
    mock_logger.warning.assert_called_with("No execution settings found for NoSettingsPrompt and service gpt-4o")

    # Test 2: prompt_execution_settings is None
    mock_prompt_none = MagicMock()
    mock_prompt_none.name = "NoneSettingsPrompt"
    mock_prompt_none.prompt_execution_settings = None

    result = extract_execution_settings(mock_prompt_none, "gpt-4o")
    assert result == {}

    # Test 3: Service ID not in settings
    mock_prompt_no_service = MagicMock()
    mock_prompt_no_service.name = "MissingServicePrompt"
    mock_prompt_no_service.prompt_execution_settings = {"other-service": MagicMock()}

    result = extract_execution_settings(mock_prompt_no_service, "gpt-4o")
    assert result == {}

    # Test 4: Service settings exist but no extension_data
    mock_settings_no_ext = MagicMock()
    del mock_settings_no_ext.extension_data  # Remove the attribute

    mock_prompt_no_ext = MagicMock()
    mock_prompt_no_ext.name = "NoExtensionDataPrompt"
    mock_prompt_no_ext.prompt_execution_settings = {"gpt-4o": mock_settings_no_ext}

    result = extract_execution_settings(mock_prompt_no_ext, "gpt-4o")
    assert result == {}

    # Test 5: extension_data is None
    mock_settings_none_ext = MagicMock()
    mock_settings_none_ext.extension_data = None

    mock_prompt_none_ext = MagicMock()
    mock_prompt_none_ext.name = "NoneExtensionDataPrompt"
    mock_prompt_none_ext.prompt_execution_settings = {"gpt-4o": mock_settings_none_ext}

    result = extract_execution_settings(mock_prompt_none_ext, "gpt-4o")
    assert result == {}
