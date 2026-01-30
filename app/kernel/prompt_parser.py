# app/kernel/prompt_parser.py
"""
Parser for semantic kernel prompt templates with message tag support.

Handles extraction and parsing of prompt templates from KernelFunction objects,
supporting custom <message> tags for structured chat conversations. Provides
variable substitution and conversion to the format expected by LLM APIs.
"""
import re
from typing import Dict, List, Tuple

from semantic_kernel.functions import KernelFunction
from ..logging.debug_logger import get_logger

logger = get_logger(__name__)


# ==================== Public Interface Functions ====================


def create_chat_messages(prompt: KernelFunction, **variables) -> List[Dict[str, str]]:
    """
    Create a list of message dictionaries from a prompt template with message tags.

    This is the main entry point for parsing prompts. It extracts the template,
    substitutes variables, parses message tags, and returns structured messages
    ready for LLM consumption.

    Args:
        prompt: The KernelFunction containing the prompt template
        **variables: Variables to be substituted in the prompt using {{$variable}} syntax

    Returns:
        List of message dictionaries with 'role' and 'content' keys

    Example:
        Given a template with:
        ```
        <message role="system">You are a helpful assistant</message>
        <message role="user">{{$question}}</message>
        ```

        And variables: {"question": "What is Python?"}

        Returns:
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is Python?"}
        ]
    """
    # Extract the template content from the kernel function
    template = get_prompt_template(prompt)

    # Perform variable substitution
    rendered_prompt = _substitute_variables(template, variables)

    # Parse message tags and extract content
    message_tuples = parse_message_tags(rendered_prompt)

    # Convert to the expected dictionary format
    messages = [{"role": role, "content": content} for role, content in message_tuples]

    return messages


def get_prompt_template(prompt: KernelFunction) -> str:
    """
    Extract the prompt template content from a KernelFunction.

    Handles various ways the semantic kernel might store template content,
    providing a unified interface for template extraction.

    Args:
        prompt: The KernelFunction containing the prompt template

    Returns:
        The prompt template string, or empty string if not found

    Note:
        This function handles multiple storage patterns as the semantic
        kernel's internal structure may vary between versions.
    """
    # Primary method: direct prompt_template attribute
    if hasattr(prompt, "prompt_template") and prompt.prompt_template is not None:
        template_obj = prompt.prompt_template

        # Handle case where prompt_template is already a string
        if isinstance(template_obj, str):
            return template_obj

        # Handle case where prompt_template is an object with a 'template' attribute
        if hasattr(template_obj, "template"):
            return template_obj.template

        # Handle case where prompt_template has a string representation
        return str(template_obj)

    # Fallback method: check description attribute
    if hasattr(prompt, "description") and prompt.description:
        return prompt.description

    # Last resort: return empty string
    return ""


def parse_message_tags(prompt_content: str) -> List[Tuple[str, str]]:
    """
    Parse <message> tags from a prompt template into a list of (role, content) tuples.

    Extracts structured message elements from the template, supporting the
    custom message tag format used in semantic kernel prompts.

    Args:
        prompt_content: String content of the prompt template containing message tags

    Returns:
        List of tuples containing (role, content) pairs with whitespace trimmed

    Example:
        Input:
        ```
        <message role="system">
            You are an assistant.
        </message>
        <message role="user">Hello</message>
        ```

        Output: [("system", "You are an assistant."), ("user", "Hello")]
    """
    # Regex pattern to match message tags with role attribute
    # Captures: role value and inner content (including newlines)
    pattern = r'<message\s+role=["\']([^"\']+)["\']>(.*?)</message>'

    # Find all matches with DOTALL flag to include newlines in content
    raw_messages = re.findall(pattern, prompt_content, re.DOTALL)

    # Clean up whitespace from content while preserving role
    messages = [(role, content.strip()) for role, content in raw_messages]

    return messages


def extract_execution_settings(prompt: KernelFunction, service_id: str) -> dict:
    """
    Extract execution settings for a specific service from the prompt.

    Args:
        prompt: The KernelFunction containing the prompt template
        service_id: The service ID to get settings for (e.g., "gpt-4o", "bedrock-claude")

    Returns:
        Dictionary of execution settings for the specified service
    """
    # The settings are stored directly on the prompt as prompt_execution_settings
    if hasattr(prompt, "prompt_execution_settings") and prompt.prompt_execution_settings:
        settings = prompt.prompt_execution_settings

        # If it's a dict with service-specific PromptExecutionSettings objects
        if isinstance(settings, dict) and service_id in settings:
            service_settings = settings[service_id]

            # Extract values from PromptExecutionSettings object
            if hasattr(service_settings, "extension_data") and service_settings.extension_data:
                return service_settings.extension_data

    logger.warning(f"No execution settings found for {prompt.name} and service {service_id}")
    return {}


# ==================== Private Helper Functions ====================


def _substitute_variables(template: str, variables: Dict[str, any]) -> str:
    """
    Substitute template variables with their values.

    Replaces placeholders in the format {{$variable_name}} with the
    corresponding values from the variables dictionary.

    Args:
        template: Template string containing placeholders
        variables: Dictionary of variable names to values

    Returns:
        Template string with all placeholders replaced

    Note:
        All values are converted to strings for substitution.
        Missing variables are left as-is in the template.
    """
    rendered = template

    for key, value in variables.items():
        # Construct placeholder pattern: {{$variable_name}}
        placeholder = f"{{${key}}}"

        # Replace all occurrences with string representation of value
        rendered = rendered.replace(placeholder, str(value))

    return rendered
