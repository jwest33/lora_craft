"""Prompt template system for customizable instruction formatting with Unsloth integration."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict, field
import logging
from copy import deepcopy
from jinja2 import Template, Environment, meta
from jinja2.exceptions import TemplateSyntaxError, UndefinedError

try:
    from utils.validators import Validators
    from utils.logging_config import get_logger
except ImportError:
    # Fallback for testing or when utils not available
    import logging
    get_logger = logging.getLogger
    class Validators:
        @staticmethod
        def validate_template(template):
            return True, "OK"


logger = get_logger(__name__)


@dataclass
class TemplateConfig:
    """Configuration for a prompt template with GRPO support."""
    name: str
    description: str
    version: str = "1.0"

    # Template strings (now support Jinja2)
    system_prompt: Optional[str] = None
    instruction_template: str = "{{ instruction }}"
    response_template: str = "{{ response }}"
    chat_template: Optional[str] = None  # Jinja2 chat template

    # GRPO-specific markers
    reasoning_start_marker: str = "<start_working_out>"
    reasoning_end_marker: str = "<end_working_out>"
    solution_start_marker: str = "<SOLUTION>"
    solution_end_marker: str = "</SOLUTION>"

    # Additional GRPO settings
    prepend_reasoning_start: bool = False  # For add_generation_prompt
    include_eos_in_response: bool = True  # Include EOS token in responses

    # Format options
    add_bos_token: bool = True
    add_eos_token: bool = True
    strip_whitespace: bool = True
    preserve_formatting: bool = False

    # Model-specific settings
    model_type: str = "auto"  # 'qwen', 'llama', 'mistral', 'chatml', 'auto'
    tokenizer_chat_template: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None

    # Conversation settings
    conversation_extension: int = 1  # Number of turns to extend
    message_field_mapping: Dict[str, str] = field(default_factory=lambda: {"role": "role", "content": "content"})


class UnslothTemplateAdapter:
    """Adapter for Unsloth-compatible chat templates."""

    def __init__(self, tokenizer=None):
        """Initialize adapter.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
        self.jinja_env = Environment()

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        eos_token: Optional[str] = None,
        reasoning_start_marker: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[int]]:
        """Apply chat template to messages (Unsloth-compatible).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            chat_template: Jinja2 template string (uses tokenizer's if None)
            tokenize: Whether to return token IDs instead of string
            add_generation_prompt: Whether to add generation prompt
            eos_token: EOS token to use
            reasoning_start_marker: Marker to prepend when add_generation_prompt=True
            **kwargs: Additional template variables

        Returns:
            Formatted string or token IDs
        """
        # Use tokenizer's template if not provided
        if chat_template is None and self.tokenizer:
            chat_template = self.tokenizer.chat_template

        if not chat_template:
            raise ValueError("No chat template provided")

        # Prepare template variables
        template_vars = {
            'messages': messages,
            'add_generation_prompt': add_generation_prompt,
            'eos_token': eos_token or (self.tokenizer.eos_token if self.tokenizer else ''),
            'system_prompt': kwargs.get('system_prompt', ''),  # Ensure system_prompt is defined
            'reasoning_start': reasoning_start_marker or '',  # Ensure reasoning_start is defined
            **kwargs
        }

        # Compile and render template
        try:
            template = self.jinja_env.from_string(chat_template)
            result = template.render(**template_vars)
        except (TemplateSyntaxError, UndefinedError) as e:
            logger.error(f"Template rendering failed: {e}")
            raise

        # Add reasoning marker if requested
        if add_generation_prompt and reasoning_start_marker:
            result += reasoning_start_marker

        # Tokenize if requested
        if tokenize and self.tokenizer:
            return self.tokenizer.encode(result, add_special_tokens=False)

        return result

    def create_grpo_template(
        self,
        system_prompt: str,
        reasoning_start: str = "<start_working_out>",
        reasoning_end: str = "<end_working_out>",
        solution_start: str = "<SOLUTION>",
        solution_end: str = "</SOLUTION>",
        eos_token: str = ""
    ) -> str:
        """Create GRPO-compatible chat template.

        Args:
            system_prompt: System prompt text
            reasoning_start: Start marker for reasoning
            reasoning_end: End marker for reasoning
            solution_start: Start marker for solution
            solution_end: End marker for solution
            eos_token: EOS token

        Returns:
            Jinja2 chat template string
        """
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] }}" + eos_token +
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '" + system_prompt + "' }}" + eos_token +
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}" + eos_token +
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}"
            "{% endif %}"
        )

        return template


class PromptTemplate:
    """Manages prompt templates for training and inference."""

    # Default templates for common models (Jinja2 format matching Unsloth)
    MODEL_DEFAULTS = {
        'grpo': {  # Default GRPO template
            'chat_template': (
                "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] }}{{ eos_token }}"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}{{ eos_token }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<start_working_out>{% endif %}"
            ),
            'bos_token': '',
            'eos_token': '',
        },
        'qwen': {
            'chat_template': (
                "{% if messages[0]['role'] == 'system' %}"
                "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            ),
            'bos_token': '<|im_start|>',
            'eos_token': '<|im_end|>',
        },
        'llama': {
            'chat_template': (
                "{% if messages[0]['role'] == 'system' %}"
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ messages[0]['content'] }}<|eot_id|>"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "<|begin_of_text|>"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
            ),
            'bos_token': '<|begin_of_text|>',
            'eos_token': '<|eot_id|>',
        },
        'mistral': {
            'chat_template': (
                "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] }}\n\n"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}</s>"
                "{% endif %}"
                "{% endfor %}"
            ),
            'bos_token': '<s>',
            'eos_token': '</s>',
        },
        'chatml': {
            'chat_template': (
                "{% if messages[0]['role'] == 'system' %}"
                "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            ),
            'bos_token': '<|im_start|>',
            'eos_token': '<|im_end|>',
        }
    }

    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize prompt template.

        Args:
            config: Template configuration
        """
        self.config = config or self._get_default_config()
        self._jinja_env = Environment()
        self._compiled_template = None
        self._variables = set()
        self._validate_config()

    def _get_default_config(self) -> TemplateConfig:
        """Get default GRPO template configuration."""
        return TemplateConfig(
            name="grpo_default",
            description="Default GRPO template for reasoning models (Unsloth-compatible)",
            system_prompt=(
                "You are given a problem.\n"
                "Think about the problem and provide your working out.\n"
                "Place it between <start_working_out> and <end_working_out>.\n"
                "Then, provide your solution between <SOLUTION></SOLUTION>"
            ),
            chat_template=(
                "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] }}{{ eos_token }}"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{{ system_prompt }}{{ eos_token }}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] }}{{ eos_token }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<start_working_out>{% endif %}"
            ),
            reasoning_start_marker="<start_working_out>",
            reasoning_end_marker="<end_working_out>",
            solution_start_marker="<SOLUTION>",
            solution_end_marker="</SOLUTION>",
            prepend_reasoning_start=False,  # Already in template
            model_type="grpo"
        )

    def _validate_config(self):
        """Validate template configuration."""
        # Validate Jinja2 templates
        templates = [
            ('instruction', self.config.instruction_template),
            ('response', self.config.response_template)
        ]

        if self.config.system_prompt:
            templates.append(('system', self.config.system_prompt))

        if self.config.chat_template:
            templates.append(('chat', self.config.chat_template))

        for name, template in templates:
            if template:
                try:
                    # Try to compile as Jinja2 template
                    self._jinja_env.from_string(template)
                except TemplateSyntaxError as e:
                    logger.warning(f"Template '{name}' has syntax error: {e}")

        # Extract variables from templates
        self._extract_variables()

        # Compile chat template if present
        if self.config.chat_template:
            try:
                self._compiled_template = self._jinja_env.from_string(self.config.chat_template)
            except TemplateSyntaxError as e:
                logger.error(f"Failed to compile chat template: {e}")

    def _extract_variables(self):
        """Extract variable names from Jinja2 templates."""
        self._variables = set()

        templates = [
            self.config.instruction_template,
            self.config.response_template,
            self.config.system_prompt,
            self.config.chat_template
        ]

        for template in templates:
            if template:
                try:
                    # Parse Jinja2 template to extract variables
                    ast = self._jinja_env.parse(template)
                    variables = meta.find_undeclared_variables(ast)
                    self._variables.update(variables)
                except TemplateSyntaxError:
                    # Fallback to simple pattern matching
                    pattern = r'\{\{\s*([^}]+)\s*\}\}'
                    matches = re.findall(pattern, template)
                    self._variables.update(matches)

    def apply(self, sample: Union[Dict[str, Any], List[Dict[str, str]]], mode: str = 'training') -> str:
        """Apply template to a data sample.

        Args:
            sample: Data sample dictionary or list of messages
            mode: 'training' or 'inference'

        Returns:
            Formatted prompt string
        """
        # Handle message list format
        if isinstance(sample, list):
            return self.apply_chat_template(sample, add_generation_prompt=(mode == 'inference'))

        # Handle dictionary format
        if self.config.chat_template and self._compiled_template:
            # For chat templates, convert instruction/output to messages format
            messages = self._convert_to_messages(sample)
            return self.apply_chat_template(messages, add_generation_prompt=(mode == 'inference'))
        else:
            # Use legacy template formatting
            variables = self._prepare_variables(sample)
            # Build template from parts
            template = self._build_template(mode)
            # Simple string formatting for backward compatibility
            try:
                formatted = template.format(**variables)
            except KeyError as e:
                logger.error(f"Missing variable in template: {e}")
                formatted = str(e)

        # Post-process
        if self.config.strip_whitespace:
            formatted = self._strip_excess_whitespace(formatted)

        return formatted

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> Union[str, List[int]]:
        """Apply chat template to messages (tokenizer-compatible).

        Args:
            messages: List of message dictionaries
            tokenize: Whether to tokenize the result
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional template variables

        Returns:
            Formatted string or token IDs
        """
        if not self.config.chat_template:
            # Build from messages manually
            result = self._build_from_messages(messages, add_generation_prompt)
        else:
            # Use Jinja2 template
            template_vars = {
                'messages': messages,
                'add_generation_prompt': add_generation_prompt,
                'eos_token': self.config.eos_token or '',
                'system_prompt': self.config.system_prompt or '',  # Always include, even if empty
                **kwargs
            }

            # Add GRPO-specific variables
            if add_generation_prompt and self.config.prepend_reasoning_start:
                template_vars['reasoning_start'] = self.config.reasoning_start_marker
            else:
                template_vars['reasoning_start'] = ''  # Ensure it's defined even if not used

            try:
                result = self._compiled_template.render(**template_vars)
            except (TemplateSyntaxError, UndefinedError) as e:
                logger.error(f"Template rendering failed: {e}")
                result = str(e)

        # Add reasoning marker if configured
        if add_generation_prompt and self.config.prepend_reasoning_start:
            result += self.config.reasoning_start_marker

        if tokenize:
            logger.warning("Tokenization requested but no tokenizer available")
            return []

        return result

    def _build_from_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        """Build prompt from messages without template.

        Args:
            messages: List of message dictionaries
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatted string
        """
        parts = []

        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')

            if role == 'system':
                parts.append(f"System: {content}")
            elif role == 'user':
                parts.append(f"User: {content}")
            elif role == 'assistant':
                parts.append(f"Assistant: {content}")
                if self.config.include_eos_in_response and self.config.eos_token:
                    parts[-1] += self.config.eos_token

        if add_generation_prompt:
            parts.append("Assistant:")
            if self.config.prepend_reasoning_start:
                parts[-1] += " " + self.config.reasoning_start_marker

        return "\n\n".join(parts)

    def _convert_to_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert instruction/output format to messages format.

        Args:
            sample: Dictionary with instruction/output fields

        Returns:
            List of message dictionaries with role and content
        """
        messages = []

        # Add system message if configured
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # Get instruction field
        instruction = sample.get('instruction')
        if not instruction:
            # Try common field names
            for field in ['prompt', 'question', 'input', 'text']:
                if field in sample:
                    instruction = sample[field]
                    break

        # Get response field
        response = sample.get('response')
        if not response:
            # Try common field names
            for field in ['answer', 'output', 'completion', 'label']:
                if field in sample:
                    response = sample[field]
                    break

        # Add user message
        if instruction:
            messages.append({
                "role": "user",
                "content": str(instruction)
            })

        # Add assistant message (for training mode)
        if response:
            messages.append({
                "role": "assistant",
                "content": str(response)
            })

        return messages

    def _prepare_variables(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for template formatting."""
        variables = deepcopy(sample)

        # Add system prompt if not present
        if 'system' not in variables and self.config.system_prompt:
            variables['system'] = self.config.system_prompt

        # Handle instruction and response fields
        if 'instruction' not in variables:
            # Try common field names
            for field in ['prompt', 'question', 'input', 'text']:
                if field in variables:
                    variables['instruction'] = variables[field]
                    break

        if 'response' not in variables:
            # Try common field names
            for field in ['answer', 'output', 'completion', 'label']:
                if field in variables:
                    variables['response'] = variables[field]
                    break

        return variables

    def _build_template(self, mode: str) -> str:
        """Build template from components."""
        if mode == 'inference':
            # For inference, don't include response
            parts = []

            if self.config.system_prompt:
                parts.append(f"System: {self.config.system_prompt}")

            parts.append(f"User: {self.config.instruction_template}")
            parts.append("Assistant:")

            return "\n\n".join(parts)
        else:
            # For training, include full conversation
            parts = []

            if self.config.system_prompt:
                parts.append(f"System: {self.config.system_prompt}")

            parts.append(f"User: {self.config.instruction_template}")
            parts.append(f"Assistant: {self.config.response_template}")

            return "\n\n".join(parts)

    def _strip_excess_whitespace(self, text: str) -> str:
        """Strip excess whitespace while preserving formatting."""
        if self.config.preserve_formatting:
            # Only strip leading/trailing whitespace
            return text.strip()
        else:
            # Normalize whitespace
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                # Strip each line
                cleaned = line.strip()
                if cleaned:
                    cleaned_lines.append(cleaned)

            # Join with single newlines
            return '\n'.join(cleaned_lines)

    def add_reasoning_markers(self, text: str, reasoning: str) -> str:
        """Add reasoning markers to text.

        Args:
            text: Original text
            reasoning: Reasoning text to add

        Returns:
            Text with reasoning markers
        """
        marked_reasoning = (
            f"{self.config.reasoning_start_marker}\n"
            f"{reasoning}\n"
            f"{self.config.reasoning_end_marker}\n"
        )

        return marked_reasoning + text

    def add_solution_markers(self, text: str) -> str:
        """Add solution markers to text.

        Args:
            text: Solution text

        Returns:
            Text with solution markers
        """
        return (
            f"{self.config.solution_start_marker}\n"
            f"{text}\n"
            f"{self.config.solution_end_marker}"
        )

    def extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning from marked text.

        Args:
            text: Text with reasoning markers

        Returns:
            Extracted reasoning or None
        """
        pattern = re.escape(self.config.reasoning_start_marker) + r"(.*?)" + re.escape(self.config.reasoning_end_marker)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        return None

    def extract_solution(self, text: str) -> Optional[str]:
        """Extract solution from marked text.

        Args:
            text: Text with solution markers

        Returns:
            Extracted solution or None
        """
        pattern = re.escape(self.config.solution_start_marker) + r"(.*?)" + re.escape(self.config.solution_end_marker)
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        return None

    def preview(self, sample: Optional[Dict[str, Any]] = None) -> str:
        """Get preview of template with sample data.

        Args:
            sample: Optional sample data

        Returns:
            Preview string
        """
        if not sample:
            # Use example data
            sample = {
                'instruction': 'What is the capital of France?',
                'response': 'The capital of France is Paris.'
            }

        return self.apply(sample, mode='training')

    def save(self, path: str):
        """Save template to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = asdict(self.config)

        # Save as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Template saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'PromptTemplate':
        """Load template from file.

        Args:
            path: Template file path

        Returns:
            PromptTemplate instance
        """
        path = Path(path)

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = TemplateConfig(**config_dict)
        return cls(config)

    @classmethod
    def from_model_type(cls, model_type: str, grpo_mode: bool = True) -> 'PromptTemplate':
        """Create template for specific model type.

        Args:
            model_type: Model type ('qwen', 'llama', 'mistral', 'chatml')
            grpo_mode: Whether to create GRPO-optimized template (default: True)

        Returns:
            PromptTemplate instance
        """
        if model_type not in cls.MODEL_DEFAULTS:
            logger.warning(f"Unknown model type: {model_type}, using default grpo")
            model_type = 'grpo'

        defaults = cls.MODEL_DEFAULTS[model_type]

        if grpo_mode:
            # Create GRPO-optimized template
            system_prompt = (
                "You are given a problem.\n"
                "Think about the problem and provide your working out.\n"
                "Place it between <start_working_out> and <end_working_out>.\n"
                "Then, provide your solution between <SOLUTION></SOLUTION>"
            )

            # Modify template to include reasoning marker on generation
            chat_template = defaults['chat_template']
            if model_type == 'qwen':
                chat_template = chat_template.replace(
                    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                    "{% if add_generation_prompt %}<|im_start|>assistant\n<start_working_out>{% endif %}"
                )

            config = TemplateConfig(
                name=f"{model_type}_grpo_template",
                description=f"GRPO template for {model_type} models",
                system_prompt=system_prompt,
                chat_template=chat_template,
                reasoning_start_marker="<start_working_out>",
                reasoning_end_marker="<end_working_out>",
                solution_start_marker="<SOLUTION>",
                solution_end_marker="</SOLUTION>",
                bos_token=defaults.get('bos_token'),
                eos_token=defaults.get('eos_token'),
                prepend_reasoning_start=(model_type != 'qwen'),  # Qwen has it in template
                model_type=model_type
            )
        else:
            config = TemplateConfig(
                name=f"{model_type}_template",
                description=f"Template for {model_type} models",
                chat_template=defaults['chat_template'],
                bos_token=defaults.get('bos_token'),
                eos_token=defaults.get('eos_token'),
                model_type=model_type
            )

        return cls(config)

    def setup_for_unsloth(self, tokenizer):
        """Configure template for Unsloth integration.

        Args:
            tokenizer: HuggingFace tokenizer to configure

        Returns:
            Modified tokenizer
        """
        if not self.config.chat_template:
            logger.warning("No chat template defined for Unsloth setup")
            return tokenizer

        # Prepare the chat template with embedded system prompt
        # This avoids the undefined variable issue when TRL uses the template
        chat_template = self.config.chat_template

        # Get the actual values to embed
        system_prompt_text = self.config.system_prompt or ''
        reasoning_text = self.config.reasoning_start_marker or ''
        eos_token_placeholder = '{{ eos_token }}'  # Keep eos_token as variable since tokenizer provides it

        # Handle the expression {{ system_prompt + eos_token }}
        if '{{ system_prompt + eos_token }}' in chat_template:
            # Replace with concatenated result, keeping eos_token as variable
            replacement = system_prompt_text + eos_token_placeholder
            chat_template = chat_template.replace('{{ system_prompt + eos_token }}', replacement)

        # Handle {{ system_prompt }} standalone (if any remain)
        if '{{ system_prompt }}' in chat_template:
            chat_template = chat_template.replace('{{ system_prompt }}', system_prompt_text)

        # Handle the |default filter syntax if present
        if 'system_prompt|default' in chat_template:
            chat_template = chat_template.replace('{{ system_prompt|default(\'\') }}', system_prompt_text)
            chat_template = chat_template.replace('{{ system_prompt|default("") }}', system_prompt_text)

        # Handle reasoning_start variable
        if '{{ reasoning_start }}' in chat_template:
            chat_template = chat_template.replace('{{ reasoning_start }}', reasoning_text)

        # Use regex to catch any remaining system_prompt references in expressions
        import re
        # Match any {{ ... system_prompt ... }} pattern
        pattern = r'\{\{[^}]*system_prompt[^}]*\}\}'
        if re.search(pattern, chat_template):
            logger.warning("Found complex system_prompt expression in template, attempting to simplify")
            # For safety, replace any remaining system_prompt references with empty string
            chat_template = re.sub(r'\bsystem_prompt\b', '""', chat_template)

        # Set the processed template on tokenizer
        tokenizer.chat_template = chat_template

        # Set special tokens if configured
        if self.config.bos_token:
            tokenizer.bos_token = self.config.bos_token
        if self.config.eos_token:
            tokenizer.eos_token = self.config.eos_token

        logger.info(f"Configured tokenizer with {self.config.name} template (system_prompt embedded)")
        return tokenizer

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return asdict(self.config)

    def validate_sample(self, sample: Union[Dict[str, Any], List[Dict[str, str]]]) -> Tuple[bool, List[str]]:
        """Validate that sample has required fields.

        Args:
            sample: Sample to validate (dict or message list)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate message list format
        if isinstance(sample, list):
            if not sample:
                errors.append("Message list is empty")
                return False, errors

            for i, message in enumerate(sample):
                if not isinstance(message, dict):
                    errors.append(f"Message {i} is not a dictionary")
                    continue

                role_field = self.config.message_field_mapping.get('role', 'role')
                content_field = self.config.message_field_mapping.get('content', 'content')

                if role_field not in message:
                    errors.append(f"Message {i} missing '{role_field}' field")
                elif message[role_field] not in ['system', 'user', 'assistant']:
                    errors.append(f"Message {i} has invalid role: {message[role_field]}")

                if content_field not in message:
                    errors.append(f"Message {i} missing '{content_field}' field")

            return len(errors) == 0, errors

        # Validate dictionary format (legacy)
        missing = []
        required = {'instruction', 'response'} if self._variables else set()

        for var in required:
            if var in self._variables and var not in sample:
                # Check alternative field names
                alternatives = {
                    'instruction': ['prompt', 'question', 'input', 'text'],
                    'response': ['answer', 'output', 'completion', 'label']
                }

                found = False
                for alt in alternatives.get(var, []):
                    if alt in sample:
                        found = True
                        break

                if not found:
                    missing.append(var)

        if missing:
            errors.append(f"Missing required fields: {', '.join(missing)}")

        return len(errors) == 0, errors

    def validate_grpo_format(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate GRPO format compliance.

        Args:
            text: Generated text to validate

        Returns:
            Tuple of (is_valid, validation_info)
        """
        info = {
            'has_reasoning': False,
            'has_solution': False,
            'reasoning_text': None,
            'solution_text': None,
            'format_valid': False,
            'errors': []
        }

        # Check for reasoning markers
        reasoning_pattern = re.escape(self.config.reasoning_start_marker) + r"(.*?)" + re.escape(self.config.reasoning_end_marker)
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)

        if reasoning_match:
            info['has_reasoning'] = True
            info['reasoning_text'] = reasoning_match.group(1).strip()
        else:
            info['errors'].append("Missing reasoning section")

        # Check for solution markers
        solution_pattern = re.escape(self.config.solution_start_marker) + r"(.*?)" + re.escape(self.config.solution_end_marker)
        solution_match = re.search(solution_pattern, text, re.DOTALL)

        if solution_match:
            info['has_solution'] = True
            info['solution_text'] = solution_match.group(1).strip()
        else:
            info['errors'].append("Missing solution section")

        # Check order (reasoning should come before solution)
        if info['has_reasoning'] and info['has_solution']:
            reasoning_pos = text.find(self.config.reasoning_end_marker)
            solution_pos = text.find(self.config.solution_start_marker)
            if reasoning_pos > solution_pos:
                info['errors'].append("Solution appears before reasoning end")

        info['format_valid'] = info['has_reasoning'] and info['has_solution'] and len(info['errors']) == 0

        return info['format_valid'], info


class TemplateLibrary:
    """Library for managing GRPO prompt templates."""

    def __init__(self, library_dir: Optional[str] = None, load_defaults: bool = True):
        """Initialize template library with GRPO as default.

        Args:
            library_dir: Directory for template storage
            load_defaults: Whether to create the default GRPO template
        """
        self.library_dir = Path(library_dir) if library_dir else Path("presets/templates")
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        self._load_library()

        # Always ensure GRPO template is available
        if load_defaults and "grpo" not in self.templates:
            self.create_default_templates()

    def _load_library(self):
        """Load all templates from library directory."""
        for template_file in self.library_dir.glob("*.json"):
            try:
                template = PromptTemplate.load(template_file)
                self.templates[template.config.name] = template
                logger.info(f"Loaded template: {template.config.name}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def add(self, template: PromptTemplate, save: bool = True):
        """Add template to library.

        Args:
            template: Template to add
            save: Whether to save to disk
        """
        self.templates[template.config.name] = template

        if save:
            template_path = self.library_dir / f"{template.config.name}.json"
            template.save(template_path)

    def get(self, name: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get template by name (defaults to 'grpo').

        Args:
            name: Template name (defaults to 'grpo')

        Returns:
            Template or None
        """
        if name is None:
            name = "grpo"  # Default to GRPO
        return self.templates.get(name)

    def list(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())

    def remove(self, name: str, delete_file: bool = True):
        """Remove template from library.

        Args:
            name: Template name
            delete_file: Whether to delete file
        """
        if name in self.templates:
            del self.templates[name]

            if delete_file:
                template_path = self.library_dir / f"{name}.json"
                if template_path.exists():
                    template_path.unlink()

    def create_default_templates(self):
        """Create default GRPO template only."""
        # THE default GRPO template (Qwen-based, Unsloth-compatible)
        grpo_config = TemplateConfig(
            name="grpo",
            description="Default GRPO template (Qwen-based, Unsloth-compatible)",
            system_prompt="You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>",
            chat_template=(
                "{% if messages[0]['role'] == 'system' %}"
                "{{ messages[0]['content'] + eos_token }}"
                "{% set loop_messages = messages[1:] %}"
                "{% else %}"
                "{{ system_prompt + eos_token }}"
                "{% set loop_messages = messages %}"
                "{% endif %}"
                "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ reasoning_start }}"
                "{% endif %}"
            ),
            reasoning_start_marker="<start_working_out>",
            reasoning_end_marker="<end_working_out>",
            solution_start_marker="<SOLUTION>",
            solution_end_marker="</SOLUTION>",
            prepend_reasoning_start=True,
            model_type="grpo"
        )
        self.add(PromptTemplate(grpo_config))

        logger.info("Created default GRPO template")

    def create_additional_templates(self):
        """Create additional template variations (optional - not loaded by default)."""
        # DeepSeek-style template
        deepseek_config = TemplateConfig(
            name="deepseek_reasoning",
            description="DeepSeek-style reasoning template",
            system_prompt="You are a helpful AI assistant. Think step by step.",
            instruction_template="{{ instruction }}",
            response_template="{{ response }}",
            reasoning_start_marker="<think>",
            reasoning_end_marker="</think>",
            solution_start_marker="<answer>",
            solution_end_marker="</answer>",
            model_type="deepseek"
        )
        self.add(PromptTemplate(deepseek_config))

        # Classic reasoning template
        reasoning_config = TemplateConfig(
            name="reasoning",
            description="Template for chain-of-thought reasoning",
            system_prompt="You are an AI assistant that explains your reasoning step by step.",
            instruction_template="{{ instruction }}",
            response_template="{{ response }}",
            reasoning_start_marker="<thinking>",
            reasoning_end_marker="</thinking>",
            solution_start_marker="<answer>",
            solution_end_marker="</answer>"
        )
        self.add(PromptTemplate(reasoning_config))

        # Math reasoning template
        math_config = TemplateConfig(
            name="math_reasoning",
            description="Template for mathematical reasoning (OpenR1-style)",
            system_prompt="You are solving a math problem. Show your complete working and provide the final answer.",
            instruction_template="{{ instruction }}",
            response_template="{{ response }}",
            reasoning_start_marker="[WORKING]",
            reasoning_end_marker="[/WORKING]",
            solution_start_marker="#### ",
            solution_end_marker="",
            model_type="math"
        )
        self.add(PromptTemplate(math_config))

        # Code generation template with reasoning
        code_config = TemplateConfig(
            name="code_generation",
            description="Template for code generation with planning",
            system_prompt="You are a helpful coding assistant. First explain your approach, then provide the implementation.",
            instruction_template="{{ instruction }}",
            response_template="{{ response }}",
            reasoning_start_marker="# Planning:\n",
            reasoning_end_marker="\n# Implementation:",
            solution_start_marker="```python\n",
            solution_end_marker="\n```"
        )
        self.add(PromptTemplate(code_config))

        # Qwen-specific GRPO template
        qwen_grpo_config = TemplateConfig(
            name="qwen_grpo",
            description="Qwen model GRPO template (Unsloth-optimized)",
            chat_template=self.MODEL_DEFAULTS['qwen']['chat_template'].replace(
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                "{% if add_generation_prompt %}<|im_start|>assistant\n<start_working_out>{% endif %}"
            ),
            reasoning_start_marker="<start_working_out>",
            reasoning_end_marker="<end_working_out>",
            solution_start_marker="<SOLUTION>",
            solution_end_marker="</SOLUTION>",
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            prepend_reasoning_start=False,  # Already in template
            model_type="qwen"
        )
        self.add(PromptTemplate(qwen_grpo_config))

        # Chat template
        chat_config = TemplateConfig(
            name="chat",
            description="Template for conversational tasks",
            system_prompt="You are a friendly and helpful assistant.",
            instruction_template="{{ instruction }}",
            response_template="{{ response }}",
            chat_template=self.MODEL_DEFAULTS['chatml']['chat_template'],
            model_type="chatml"
        )
        self.add(PromptTemplate(chat_config))

        logger.info("Created additional templates")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing GRPO-Default Template System")
    print("=" * 60)

    # Test 1: Default template is GRPO
    print("\n1. Testing default GRPO template:")
    default_template = PromptTemplate()  # No config = GRPO default
    print(f"Default template name: {default_template.config.name}")
    print(f"Is GRPO: {default_template.config.model_type == 'grpo'}")
    print(f"Reasoning markers: {default_template.config.reasoning_start_marker} / {default_template.config.reasoning_end_marker}")

    # Test 2: Create GRPO template for specific model (Qwen)
    print("\n2. Creating GRPO template for Qwen (default grpo_mode=True):")
    qwen_template = PromptTemplate.from_model_type('qwen')  # GRPO by default
    print(f"Template name: {qwen_template.config.name}")
    print(f"Reasoning markers: {qwen_template.config.reasoning_start_marker} / {qwen_template.config.reasoning_end_marker}")

    # Test 3: Apply template to messages (Unsloth-style)
    print("\n3. Testing message-based GRPO format:")
    messages = [
        {"role": "user", "content": "What is the square root of 144?"},
    ]

    # Without generation prompt
    result = qwen_template.apply_chat_template(messages, add_generation_prompt=False)
    print("Without generation prompt:")
    print(result[:200] + "..." if len(result) > 200 else result)

    # With generation prompt (for inference)
    result_with_prompt = qwen_template.apply_chat_template(messages, add_generation_prompt=True)
    print("\nWith generation prompt (includes reasoning start):")
    print(result_with_prompt[:200] + "..." if len(result_with_prompt) > 200 else result_with_prompt)

    # Test 4: Validate GRPO format
    print("\n4. Testing GRPO format validation:")
    test_response = (
        "<start_working_out>\n"
        "To find the square root of 144, I need to find a number that when multiplied by itself equals 144.\n"
        "12 × 12 = 144\n"
        "<end_working_out>\n"
        "<SOLUTION>12</SOLUTION>"
    )

    is_valid, info = qwen_template.validate_grpo_format(test_response)
    print(f"Response is valid: {is_valid}")
    print(f"Has reasoning: {info['has_reasoning']}")
    print(f"Has solution: {info['has_solution']}")
    if info['solution_text']:
        print(f"Extracted solution: {info['solution_text']}")

    # Test 5: UnslothTemplateAdapter
    print("\n5. Testing UnslothTemplateAdapter for GRPO:")
    adapter = UnslothTemplateAdapter()

    grpo_template = adapter.create_grpo_template(
        system_prompt="You are a math tutor. Show your work step by step.",
        reasoning_start="<think>",
        reasoning_end="</think>",
        solution_start="<answer>",
        solution_end="</answer>",
        eos_token="<|im_end|>"
    )

    print("Generated GRPO template (first 300 chars):")
    print(grpo_template[:300] + "...")

    # Test 6: Template Library defaults to GRPO
    print("\n6. Template library with GRPO as default:")
    library = TemplateLibrary()  # Automatically creates GRPO

    templates = library.list()
    print(f"Available templates: {', '.join(templates)}")

    # Get default template (should be GRPO)
    default_from_lib = library.get()  # No name = default to GRPO
    if default_from_lib:
        print(f"\nDefault template loaded: {default_from_lib.config.name}")
        print(f"System prompt: {default_from_lib.config.system_prompt[:80]}...")

    # Test that we can still get it by name
    grpo_by_name = library.get("grpo")
    print(f"\nGRPO by name available: {grpo_by_name is not None}")
    print(f"Default and named are same: {default_from_lib is grpo_by_name}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
