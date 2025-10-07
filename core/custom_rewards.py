"""Custom reward framework for GRPO training."""

import re
import ast
import math
import json
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
import numpy as np
import logging

from utils.validators import Validators
from utils.logging_config import get_logger


logger = get_logger(__name__)


class RewardType(Enum):
    """Types of reward functions."""
    BINARY = "binary"  # 0 or 1
    CONTINUOUS = "continuous"  # 0.0 to 1.0
    SPARSE = "sparse"  # Mostly 0 with occasional rewards
    DENSE = "dense"  # Continuous feedback
    CUSTOM = "custom"  # User-defined


@dataclass
class RewardConfig:
    """Configuration for a reward function."""
    name: str
    description: str
    type: RewardType
    weight: float = 1.0

    # Format matching
    use_regex: bool = False
    regex_pattern: Optional[str] = None

    # Numerical comparison
    extract_number: bool = False
    number_tolerance: float = 1e-6
    relative_tolerance: bool = False

    # Length penalties
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    optimal_length: Optional[int] = None

    # Template validation
    section_tags: Optional[List[str]] = None
    required_sections: Optional[List[str]] = None
    order_matters: bool = False

    # Multi-choice validation
    valid_choices: Optional[List[str]] = None
    case_sensitive: bool = False
    exact_match: bool = False

    # Section content validation
    section_tag: Optional[str] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    required_keywords: Optional[List[str]] = None

    # Sequential pattern validation
    patterns: Optional[List[str]] = None
    strict_order: bool = False

    # Signal accuracy validation
    signal_direction_match_score: float = 0.70  # Score for correct direction, wrong strength

    # Custom function
    custom_code: Optional[str] = None
    safe_mode: bool = True


class RewardFunction:
    """Base class for reward functions."""

    def __init__(self, config: RewardConfig):
        """Initialize reward function.

        Args:
            config: Reward configuration
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate reward configuration."""
        if self.config.use_regex and self.config.regex_pattern:
            try:
                re.compile(self.config.regex_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        if self.config.custom_code:
            valid, msg = Validators.validate_python_code(
                self.config.custom_code,
                safe_mode=self.config.safe_mode
            )
            if not valid:
                raise ValueError(f"Invalid custom code: {msg}")

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute reward for generated text.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response (optional)
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Reward value
        """
        raise NotImplementedError


class BinaryReward(RewardFunction):
    """Binary reward (0 or 1) based on exact or pattern matching."""

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute binary reward."""
        if reference is None:
            return 0.0

        if self.config.use_regex and self.config.regex_pattern:
            # Use regex matching
            pattern = re.compile(self.config.regex_pattern)
            gen_match = bool(pattern.search(generated))
            ref_match = bool(pattern.search(reference))
            return 1.0 if gen_match == ref_match else 0.0
        else:
            # Exact matching
            gen_normalized = generated.strip().lower()
            ref_normalized = reference.strip().lower()
            return 1.0 if gen_normalized == ref_normalized else 0.0


class NumericalReward(RewardFunction):
    """Reward based on numerical answer extraction and comparison."""

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text."""
        # Pattern to match numbers (including decimals and scientific notation)
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                num = float(match)
                numbers.append(num)
            except ValueError:
                continue

        return numbers

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute numerical reward."""
        if reference is None:
            return 0.0

        gen_numbers = self._extract_numbers(generated)
        ref_numbers = self._extract_numbers(reference)

        if not ref_numbers:
            # No numbers in reference
            return 1.0 if not gen_numbers else 0.0

        if not gen_numbers:
            # No numbers extracted from generation
            return 0.0

        # Compare first number (or all numbers)
        ref_num = ref_numbers[0]
        gen_num = gen_numbers[0]

        if self.config.relative_tolerance:
            # Relative tolerance
            if ref_num == 0:
                is_close = abs(gen_num) < self.config.number_tolerance
            else:
                rel_error = abs(gen_num - ref_num) / abs(ref_num)
                is_close = rel_error < self.config.number_tolerance
        else:
            # Absolute tolerance
            is_close = abs(gen_num - ref_num) < self.config.number_tolerance

        return 1.0 if is_close else 0.0


class LengthReward(RewardFunction):
    """Reward based on response length (token count if tokenizer provided, else word count).

    Best Practice for max_length:
        Set max_length to (max_new_tokens - margin) to detect clipped/truncated responses.

        Example:
            max_new_tokens = 512  # Your generation limit
            max_length = 492      # Detects responses that hit the limit (20 token margin)

        This penalizes responses that were likely cut off mid-thought, encouraging
        the model to learn more concise responses that fit within the token budget.
    """

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute length-based reward.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response (optional, unused)
            tokenizer: Optional tokenizer for token-based length calculation

        Returns:
            Reward value based on length constraints
        """
        # Use token count if tokenizer is provided, otherwise fall back to word count
        if tokenizer is not None:
            try:
                gen_length = len(tokenizer.encode(generated, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"Failed to tokenize for length reward: {e}, falling back to word count")
                gen_length = len(generated.split())
        else:
            gen_length = len(generated.split())

        reward = 1.0

        # Check minimum length (only if explicitly set)
        if self.config.min_length is not None and gen_length < self.config.min_length:
            penalty = (self.config.min_length - gen_length) / self.config.min_length
            reward *= (1 - penalty)

        # Check maximum length (only if explicitly set)
        if self.config.max_length is not None and gen_length > self.config.max_length:
            penalty = (gen_length - self.config.max_length) / self.config.max_length
            reward *= (1 - min(penalty, 0.5))  # Cap penalty at 50%

        # Check optimal length (only if explicitly set)
        if self.config.optimal_length is not None:
            distance = abs(gen_length - self.config.optimal_length)
            penalty = distance / self.config.optimal_length
            reward *= math.exp(-penalty)  # Gaussian-like penalty

        return max(0.0, min(1.0, reward))


class FormatReward(RewardFunction):
    """Reward based on format compliance."""

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute format-based reward."""
        if not self.config.regex_pattern:
            return 1.0

        pattern = re.compile(self.config.regex_pattern, re.MULTILINE | re.DOTALL)
        match = pattern.search(generated)

        return 1.0 if match else 0.0


class CustomReward(RewardFunction):
    """Custom user-defined reward function."""

    def __init__(self, config: RewardConfig):
        """Initialize custom reward."""
        super().__init__(config)
        self._compile_function()

    def _compile_function(self):
        """Compile custom reward function."""
        if not self.config.custom_code:
            raise ValueError("Custom code is required for custom reward")

        # Create a restricted namespace
        namespace = {
            're': re,
            'math': math,
            'np': np,
            'len': len,
            'str': str,
            'float': float,
            'int': int,
            'abs': abs,
            'min': min,
            'max': max,
        }

        # Compile and execute the code
        try:
            exec(self.config.custom_code, namespace)

            # Look for compute_reward function
            if 'compute_reward' not in namespace:
                raise ValueError("Custom code must define compute_reward(instruction, generated, reference) function")

            self._custom_func = namespace['compute_reward']

        except Exception as e:
            raise ValueError(f"Failed to compile custom code: {e}")

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute custom reward."""
        try:
            reward = self._custom_func(instruction, generated, reference)
            return float(max(0.0, min(1.0, reward)))
        except Exception as e:
            logger.error(f"Custom reward computation failed: {e}")
            return 0.0


class TemplateValidationReward(RewardFunction):
    """Reward for validating structured template output."""

    def __init__(self, config: RewardConfig):
        """Initialize template validation reward."""
        super().__init__(config)
        self.section_tags = config.section_tags or []
        self.required_sections = config.required_sections or []
        self.order_matters = config.order_matters or False

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute template validation reward with partial credit."""
        score = 1.0
        found_sections = []
        required_found = 0
        required_total = len(self.required_sections)

        # Check for required sections
        for tag in self.section_tags:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"

            if open_tag in generated and close_tag in generated:
                # Extract content between tags
                start = generated.find(open_tag) + len(open_tag)
                end = generated.find(close_tag)

                if end > start:
                    content = generated[start:end].strip()
                    if content:  # Has content
                        found_sections.append(tag)
                        if tag in self.required_sections:
                            required_found += 1
                    elif tag in self.required_sections:
                        score *= 0.8  # Smaller penalty for empty required section
                elif tag in self.required_sections:
                    # Empty tag - partial credit
                    score *= 0.6
            # Missing required section - don't zero out the score

        # Give partial credit based on how many required sections were found
        if required_total > 0:
            section_completeness = required_found / required_total
            # Minimum 20% score, maximum 100% for all sections
            score *= (0.2 + 0.8 * section_completeness)

        # Check order if required
        if self.order_matters and len(found_sections) > 1:
            expected_order = [tag for tag in self.section_tags if tag in found_sections]
            if found_sections != expected_order:
                score *= 0.85  # Smaller penalty for wrong order

        return max(0.0, min(1.0, score))


class MultiChoiceValidationReward(RewardFunction):
    """Reward for validating output contains one valid choice."""

    def __init__(self, config: RewardConfig):
        """Initialize multi-choice validation reward."""
        super().__init__(config)
        self.valid_choices = config.valid_choices or []
        self.case_sensitive = config.case_sensitive or False
        self.exact_match = config.exact_match or False

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute multi-choice validation reward."""
        if not self.valid_choices:
            return 1.0

        text = generated if self.case_sensitive else generated.upper()
        choices = self.valid_choices if self.case_sensitive else [c.upper() for c in self.valid_choices]

        found_choices = []
        for choice in choices:
            if self.exact_match:
                # Look for exact word boundaries
                pattern = r'\b' + re.escape(choice) + r'\b'
                if re.search(pattern, text):
                    found_choices.append(choice)
            else:
                if choice in text:
                    found_choices.append(choice)

        # Reward if at least one valid choice is found
        if len(found_choices) == 1:
            return 1.0  # Perfect score for exactly one choice
        elif len(found_choices) > 1:
            return 0.5  # Partial credit for multiple choices (better than none)
        else:
            return 0.0  # No valid choice found


class SectionContentReward(RewardFunction):
    """Reward for validating content within specific sections."""

    def __init__(self, config: RewardConfig):
        """Initialize section content reward."""
        super().__init__(config)
        self.section_tag = config.section_tag or ""
        self.min_words = config.min_words or 0
        self.max_words = config.max_words or float('inf')
        self.required_keywords = config.required_keywords or []

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute section content reward."""
        score = 1.0

        # Extract section content
        open_tag = f"<{self.section_tag}>"
        close_tag = f"</{self.section_tag}>"

        if open_tag in generated and close_tag in generated:
            start = generated.find(open_tag) + len(open_tag)
            end = generated.find(close_tag)

            if end > start:
                content = generated[start:end].strip()
                words = content.split()
                word_count = len(words)

                # Check word count
                if word_count < self.min_words:
                    score *= 0.5
                elif word_count > self.max_words:
                    score *= 0.7

                # Check for required keywords - use percentage-based scoring
                content_lower = content.lower()
                if self.required_keywords:
                    keywords_found = 0
                    for keyword in self.required_keywords:
                        if keyword.lower() in content_lower:
                            keywords_found += 1

                    # Give partial credit based on percentage of keywords found
                    # At least 30% score even if no keywords, up to 100% for all keywords
                    keyword_ratio = keywords_found / len(self.required_keywords)
                    keyword_score = 0.3 + (0.7 * keyword_ratio)
                    score *= keyword_score
            else:
                return 0.0  # Empty section
        else:
            return 0.0  # Section not found

        return max(0.0, min(1.0, score))


class SequentialPatternReward(RewardFunction):
    """Reward for patterns appearing in specific order."""

    def __init__(self, config: RewardConfig):
        """Initialize sequential pattern reward."""
        super().__init__(config)
        self.patterns = config.patterns or []
        self.strict_order = config.strict_order or False

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute sequential pattern reward."""
        if not self.patterns:
            return 1.0

        positions = []
        for pattern in self.patterns:
            try:
                regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
                match = regex.search(generated)
                if match:
                    positions.append(match.start())
                else:
                    positions.append(-1)  # Pattern not found
            except re.error:
                positions.append(-1)

        # Check if all patterns were found
        if -1 in positions:
            missing_count = positions.count(-1)
            return max(0.0, 1.0 - (missing_count * 0.3))

        # Check order if all patterns found
        if self.strict_order:
            sorted_positions = sorted(positions)
            if positions == sorted_positions:
                return 1.0
            else:
                return 0.5  # Patterns found but wrong order
        else:
            return 1.0  # All patterns found, order doesn't matter


class SignalAccuracyReward(RewardFunction):
    """Reward for trading signal accuracy with partial credit for direction.

    This reward function provides partial credit when the model gets the trading
    direction correct (BUY vs SELL) but the strength wrong (STRONG vs WEAK).

    Scoring:
        - Perfect match (e.g., STRONG_BUY == STRONG_BUY): 1.00
        - Right direction, wrong strength (e.g., WEAK_BUY when STRONG_BUY): 0.70
        - HOLD when should be WEAK signal: 0.50
        - HOLD when should be STRONG signal: 0.40
        - Opposite direction (e.g., BUY when should be SELL): 0.00
    """

    # Signal hierarchy for comparison
    SIGNAL_HIERARCHY = {
        'STRONG_BUY': 2,
        'WEAK_BUY': 1,
        'HOLD': 0,
        'WEAK_SELL': -1,
        'STRONG_SELL': -2
    }

    def __init__(self, config: RewardConfig):
        """Initialize signal accuracy reward.

        Args:
            config: Reward configuration with valid_choices containing signal names
        """
        super().__init__(config)
        self.valid_signals = config.valid_choices or []
        self.direction_match_score = config.signal_direction_match_score

    def _extract_signal(self, text: str) -> Optional[str]:
        """Extract signal from <signal> tags.

        Args:
            text: Text containing signal tags

        Returns:
            Extracted signal name (normalized) or None
        """
        # Extract from <signal>...</signal> tags
        pattern = r'<signal>\s*([^<]+?)\s*</signal>'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            signal = match.group(1).strip().upper()
            # Normalize: replace spaces with underscores
            signal = signal.replace(' ', '_')

            # Validate against known signals
            if signal in self.SIGNAL_HIERARCHY:
                return signal

        return None

    def _get_signal_direction(self, signal: str) -> str:
        """Get signal direction: BUY, SELL, or HOLD.

        Args:
            signal: Signal name (e.g., 'STRONG_BUY')

        Returns:
            Direction: 'BUY', 'SELL', or 'HOLD'
        """
        if 'BUY' in signal:
            return 'BUY'
        elif 'SELL' in signal:
            return 'SELL'
        else:
            return 'HOLD'

    def _is_strong_signal(self, signal: str) -> bool:
        """Check if signal is strong (not weak or hold).

        Args:
            signal: Signal name

        Returns:
            True if STRONG_BUY or STRONG_SELL
        """
        return 'STRONG' in signal

    def compute(self,
                instruction: str,
                generated: str,
                reference: Optional[str] = None,
                tokenizer: Optional[Any] = None) -> float:
        """Compute signal accuracy reward with partial credit.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response with correct signal
            tokenizer: Optional tokenizer (unused for this reward)

        Returns:
            Reward value (0.0 to 1.0)
        """
        # Extract generated signal
        gen_signal = self._extract_signal(generated)

        if gen_signal is None:
            # No valid signal found in generation
            return 0.0

        # If no reference, just reward valid format
        if reference is None:
            return 1.0

        # Extract reference signal
        ref_signal = self._extract_signal(reference)

        if ref_signal is None:
            # Can't compare without reference signal, give benefit of doubt
            return 1.0

        # Perfect match
        if gen_signal == ref_signal:
            return 1.0

        # Get directions and strengths
        gen_dir = self._get_signal_direction(gen_signal)
        ref_dir = self._get_signal_direction(ref_signal)

        # Opposite direction = no credit (most important check)
        if (gen_dir == 'BUY' and ref_dir == 'SELL') or \
           (gen_dir == 'SELL' and ref_dir == 'BUY'):
            return 0.0

        # Same direction, different strength = configurable credit
        # This is the key partial credit case!
        if gen_dir == ref_dir and gen_dir != 'HOLD':
            return self.direction_match_score

        # Fallback (shouldn't reach here)
        return 0.0


class RewardPreset:
    """A preset reward configuration with metadata."""

    def __init__(self,
                 name: str,
                 category: str,
                 description: str,
                 example_input: str,
                 example_output: str,
                 builder_func: Callable[[], 'CustomRewardBuilder'],
                 difficulty: str = "intermediate",
                 tags: List[str] = None,
                 expected_fields: Optional[Dict[str, str]] = None,
                 field_examples: Optional[Dict[str, str]] = None,
                 optional_fields: Optional[List[str]] = None):
        """Initialize reward preset.

        Args:
            name: Display name of the preset
            category: Category (e.g., 'Math', 'Code', 'Language')
            description: Detailed description of what this rewards
            example_input: Example input that works with this reward
            example_output: Example output that would score high
            builder_func: Function that creates the configured builder
            difficulty: Difficulty level ('beginner', 'intermediate', 'advanced')
            tags: Additional searchable tags
            expected_fields: Expected dataset fields and their descriptions
            field_examples: Example values for each expected field
            optional_fields: List of optional field names
        """
        self.name = name
        self.category = category
        self.description = description
        self.example_input = example_input
        self.example_output = example_output
        self.builder_func = builder_func
        self.difficulty = difficulty
        self.tags = tags or []
        self.expected_fields = expected_fields or {
            "instruction": "The input prompt or question",
            "response": "The expected output or answer"
        }
        self.field_examples = field_examples or {
            "instruction": example_input,
            "response": example_output
        }
        self.optional_fields = optional_fields or ["reference"]

    def create_builder(self) -> 'CustomRewardBuilder':
        """Create a reward builder from this preset."""
        return self.builder_func()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'example_input': self.example_input,
            'example_output': self.example_output,
            'difficulty': self.difficulty,
            'tags': self.tags,
            'expected_fields': self.expected_fields,
            'field_examples': self.field_examples,
            'optional_fields': self.optional_fields
        }


class RewardPresetLibrary:
    """Library of preset reward configurations."""

    def __init__(self):
        """Initialize the preset library."""
        self.presets: Dict[str, RewardPreset] = {}
        self.categories: Dict[str, List[str]] = {}
        self._load_default_presets()
        self._load_custom_presets()

    def _load_default_presets(self):
        """Load all default preset configurations."""
        # Math & Science presets
        self.add_preset(RewardPreset(
            name="Mathematical Problem Solving",
            category="Math & Science",
            description="Rewards accurate numerical answers with proper formatting (e.g., \\boxed{} notation)",
            example_input="Solve: 2x + 5 = 13",
            example_output="To solve 2x + 5 = 13:\n2x = 8\nx = 4\n\\boxed{4}",
            builder_func=create_math_reward,
            difficulty="intermediate",
            tags=["math", "algebra", "numerical"]
        ))

        self.add_preset(RewardPreset(
            name="Scientific Calculation",
            category="Math & Science",
            description="Rewards precise scientific calculations with proper units and significant figures",
            example_input="Calculate the force: mass=5kg, acceleration=2m/s²",
            example_output="F = ma = 5kg × 2m/s² = 10N",
            builder_func=lambda: self._create_scientific_reward(),
            difficulty="intermediate",
            tags=["physics", "chemistry", "units"]
        ))

        # Code Generation presets
        self.add_preset(RewardPreset(
            name="Code Generation",
            category="Programming",
            description="Rewards well-formatted code with proper syntax, comments, and structure",
            example_input="Write a function to reverse a string",
            example_output="```python\ndef reverse_string(s):\n    # Return reversed string\n    return s[::-1]\n```",
            builder_func=create_code_reward,
            difficulty="intermediate",
            tags=["code", "programming", "syntax"]
        ))

        self.add_preset(RewardPreset(
            name="Algorithm Implementation",
            category="Programming",
            description="Rewards correct algorithm implementation with efficiency considerations",
            example_input="Implement binary search",
            example_output="```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```",
            builder_func=lambda: self._create_algorithm_reward(),
            difficulty="advanced",
            tags=["algorithms", "data structures", "optimization"]
        ))

        # Language & Writing presets

        self.add_preset(RewardPreset(
            name="Technical Documentation",
            category="Language & Writing",
            description="Rewards clear, structured technical documentation",
            example_input="Document this API endpoint",
            example_output="## GET /api/users\n\n**Description:** Retrieves user information\n\n**Parameters:**\n- `id` (required): User ID\n\n**Response:** JSON object with user data",
            builder_func=lambda: self._create_documentation_reward(),
            difficulty="intermediate",
            tags=["documentation", "technical", "api"]
        ))

        # Q&A presets
        self.add_preset(RewardPreset(
            name="Explanatory Q&A",
            category="Question Answering",
            description="Rewards detailed explanations with examples",
            example_input="How does photosynthesis work?",
            example_output="Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in two stages: 1) Light reactions in the thylakoids, and 2) Calvin cycle in the stroma...",
            builder_func=lambda: self._create_explanatory_qa_reward(),
            difficulty="intermediate",
            tags=["explanation", "educational", "detailed"]
        ))

        self.add_preset(RewardPreset(
            name="Step-by-Step Instructions",
            category="Question Answering",
            description="Rewards clear, numbered step-by-step instructions",
            example_input="How do I make coffee?",
            example_output="1. Boil water to 200°F\n2. Add 2 tbsp ground coffee per cup\n3. Pour water over grounds\n4. Steep for 4 minutes\n5. Press/filter and serve",
            builder_func=lambda: self._create_instruction_reward(),
            difficulty="beginner",
            tags=["instructions", "tutorial", "how-to"]
        ))


        # Data & Analysis presets
        self.add_preset(RewardPreset(
            name="Data Analysis",
            category="Data & Analysis",
            description="Rewards data-driven insights with supporting evidence",
            example_input="Analyze this sales trend",
            example_output="The data shows a 15% increase in Q3, primarily driven by seasonal demand. Key metrics: Revenue +$2.3M, Units +5,000.",
            builder_func=lambda: self._create_data_analysis_reward(),
            difficulty="advanced",
            tags=["data", "analytics", "insights"]
        ))

        self.add_preset(RewardPreset(
            name="Technical Analysis Signal",
            category="Data & Analysis",
            description="Rewards structured technical analysis with proper signal classification",
            example_input="Analyze indicators: RSI=75, MACD bullish crossover, Price above SMA200",
            example_output="<analysis>\nRSI at 75 indicates overbought conditions. MACD bullish crossover suggests upward momentum. Price above SMA200 confirms uptrend.\n</analysis>\n<signal>\nWEAK_BUY\n</signal>",
            builder_func=lambda: self._create_technical_analysis_reward(),
            difficulty="advanced",
            tags=["trading", "technical analysis", "signals", "finance"]
        ))

        # Reasoning presets
        self.add_preset(RewardPreset(
            name="Logical Reasoning",
            category="Reasoning",
            description="Rewards clear logical reasoning with premise-conclusion structure",
            example_input="If all birds can fly and penguins are birds, can penguins fly?",
            example_output="This is a logical fallacy. The premise 'all birds can fly' is false. Penguins are birds but cannot fly, demonstrating the invalid premise.",
            builder_func=lambda: self._create_logical_reasoning_reward(),
            difficulty="advanced",
            tags=["logic", "reasoning", "critical thinking"]
        ))

        self.add_preset(RewardPreset(
            name="Chain of Thought",
            category="Reasoning",
            description="Rewards step-by-step reasoning process",
            example_input="How many windows are in a typical house?",
            example_output="Let me think step by step:\n1. Typical house has 3-4 bedrooms (2 windows each) = 6-8\n2. Living room (2-3 windows) = 2-3\n3. Kitchen (1-2 windows) = 1-2\nTotal: approximately 10-15 windows",
            builder_func=lambda: self._create_chain_of_thought_reward(),
            difficulty="intermediate",
            tags=["reasoning", "step-by-step", "analytical"]
        ))

        # Conversation presets removed - too gameable via keyword stuffing

        # Format compliance presets
        self.add_preset(RewardPreset(
            name="Markdown Formatting",
            category="Formatting",
            description="Rewards proper markdown formatting with headers, lists, and code blocks",
            example_input="Format this as markdown",
            example_output="# Title\n\n## Section 1\n\n- Point 1\n- Point 2\n\n```code\nexample\n```",
            builder_func=lambda: self._create_markdown_reward(),
            difficulty="beginner",
            tags=["markdown", "formatting", "structure"]
        ))

        self.add_preset(RewardPreset(
            name="Citation Format",
            category="Formatting",
            description="Rewards proper citation formatting (APA/MLA style)",
            example_input="Cite this source",
            example_output="Smith, J. (2023). *Title of Work*. Publisher. https://example.com",
            builder_func=lambda: self._create_citation_reward(),
            difficulty="intermediate",
            tags=["citation", "academic", "references"]
        ))

    def add_preset(self, preset: RewardPreset):
        """Add a preset to the library."""
        self.presets[preset.name] = preset

        if preset.category not in self.categories:
            self.categories[preset.category] = []
        self.categories[preset.category].append(preset.name)

    def get_preset(self, name: str) -> Optional[RewardPreset]:
        """Get a preset by name."""
        return self.presets.get(name)

    def get_presets_by_category(self, category: str) -> List[RewardPreset]:
        """Get all presets in a category."""
        preset_names = self.categories.get(category, [])
        return [self.presets[name] for name in preset_names]

    def get_all_categories(self) -> List[str]:
        """Get all category names."""
        return list(self.categories.keys())

    def search_presets(self, query: str) -> List[RewardPreset]:
        """Search presets by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for preset in self.presets.values():
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower() or
                any(query_lower in tag.lower() for tag in preset.tags)):
                results.append(preset)

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert library to dictionary for JSON serialization."""
        return {
            'categories': self.categories,
            'presets': {name: preset.to_dict() for name, preset in self.presets.items()}
        }

    # Custom preset management methods
    def _get_custom_presets_file(self) -> str:
        """Get path to custom presets JSON file."""
        from pathlib import Path
        presets_dir = Path(__file__).parent.parent / 'presets' / 'rewards'
        presets_dir.mkdir(parents=True, exist_ok=True)
        return str(presets_dir / 'custom_presets.json')

    def _load_custom_presets(self):
        """Load user-created custom presets from file."""
        try:
            custom_file = self._get_custom_presets_file()
            if not Path(custom_file).exists():
                return

            with open(custom_file, 'r') as f:
                custom_data = json.load(f)

            for preset_data in custom_data.get('presets', []):
                # Create preset from stored component data
                self.add_custom_preset_from_data(preset_data)

            logger.info(f"Loaded {len(custom_data.get('presets', []))} custom presets")
        except Exception as e:
            logger.error(f"Failed to load custom presets: {e}")

    def _save_custom_presets(self):
        """Save all custom presets to file."""
        try:
            custom_file = self._get_custom_presets_file()
            custom_presets = []

            # Find all custom presets (those in "Custom" category)
            for name, preset in self.presets.items():
                if preset.category == "Custom":
                    # Get the builder to extract component details
                    builder = preset.create_builder()
                    components = builder.get_component_details()

                    custom_presets.append({
                        'name': preset.name,
                        'description': preset.description,
                        'example_input': preset.example_input,
                        'example_output': preset.example_output,
                        'difficulty': preset.difficulty,
                        'tags': preset.tags,
                        'components': components,
                        'created_at': datetime.now().isoformat()
                    })

            with open(custom_file, 'w') as f:
                json.dump({'presets': custom_presets}, f, indent=2)

            logger.info(f"Saved {len(custom_presets)} custom presets")
        except Exception as e:
            logger.error(f"Failed to save custom presets: {e}")

    def add_custom_preset_from_data(self, preset_data: Dict[str, Any]):
        """Add a custom preset from component data.

        Args:
            preset_data: Dictionary containing preset info and components
        """
        def builder_func() -> 'CustomRewardBuilder':
            builder = CustomRewardBuilder()
            for comp in preset_data.get('components', []):
                self._add_component_to_builder(builder, comp)
            return builder

        preset = RewardPreset(
            name=preset_data['name'],
            category="Custom",
            description=preset_data.get('description', 'Custom reward configuration'),
            example_input=preset_data.get('example_input', ''),
            example_output=preset_data.get('example_output', ''),
            builder_func=builder_func,
            difficulty=preset_data.get('difficulty', 'intermediate'),
            tags=preset_data.get('tags', ['custom'])
        )

        self.add_preset(preset)

    def _add_component_to_builder(self, builder: 'CustomRewardBuilder', comp: Dict[str, Any]):
        """Add a component to builder from component data dict."""
        name = comp.get('name', 'component')
        comp_type = comp.get('type', 'continuous')
        weight = comp.get('weight', 1.0)
        params = comp.get('parameters', {})

        # Map component type to builder method
        if comp_type == 'binary' or name == 'binary':
            regex_pattern = params.get('regex_pattern') or params.get('pattern')
            builder.add_binary_reward(name, regex_pattern=regex_pattern, weight=weight)

        elif comp_type == 'numerical' or name == 'numerical':
            tolerance = params.get('tolerance', 1e-6)
            relative = params.get('relative', False)
            builder.add_numerical_reward(name, tolerance=tolerance, relative=relative, weight=weight)

        elif comp_type == 'length' or 'length' in name:
            min_len = params.get('min_length')
            max_len = params.get('max_length')
            opt_len = params.get('optimal_length')
            builder.add_length_reward(name, min_length=min_len, max_length=max_len,
                                     optimal_length=opt_len, weight=weight)

        elif comp_type == 'format' or name == 'format':
            pattern = params.get('pattern') or params.get('regex_pattern', '')
            if pattern:
                builder.add_format_reward(name, pattern=pattern, weight=weight)

        elif comp_type == 'template':
            section_tags = params.get('section_tags', [])
            required = params.get('required_sections', section_tags)
            order = params.get('order_matters', False)
            builder.add_template_validation(name, section_tags=section_tags,
                                           required_sections=required,
                                           order_matters=order, weight=weight)

        elif comp_type == 'multi_choice' or comp_type == 'choice':
            valid_choices = params.get('valid_choices', [])
            case_sens = params.get('case_sensitive', False)
            exact = params.get('exact_match', True)
            builder.add_multi_choice_validation(name, valid_choices=valid_choices,
                                               case_sensitive=case_sens,
                                               exact_match=exact, weight=weight)

        elif comp_type == 'section_content' or comp_type == 'content':
            section_tag = params.get('section_tag', '')
            min_words = params.get('min_words')
            max_words = params.get('max_words')
            keywords = params.get('required_keywords', [])
            builder.add_section_content(name, section_tag=section_tag,
                                       min_words=min_words, max_words=max_words,
                                       required_keywords=keywords, weight=weight)

        elif comp_type == 'sequential' or comp_type == 'pattern':
            patterns = params.get('patterns', [])
            strict = params.get('strict_order', True)
            builder.add_sequential_pattern(name, patterns=patterns,
                                          strict_order=strict, weight=weight)

    def add_custom_preset(self, name: str, description: str, components: List[Dict[str, Any]],
                         example_input: str = '', example_output: str = '',
                         difficulty: str = 'intermediate', tags: List[str] = None) -> bool:
        """Add a new custom preset.

        Args:
            name: Preset name
            description: Preset description
            components: List of component configurations
            example_input: Example input
            example_output: Example output
            difficulty: Difficulty level
            tags: Tags for searchability

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if name already exists
            if name in self.presets:
                logger.warning(f"Preset '{name}' already exists")
                return False

            preset_data = {
                'name': name,
                'description': description,
                'example_input': example_input,
                'example_output': example_output,
                'difficulty': difficulty,
                'tags': tags or ['custom'],
                'components': components
            }

            self.add_custom_preset_from_data(preset_data)
            self._save_custom_presets()
            return True
        except Exception as e:
            logger.error(f"Failed to add custom preset: {e}")
            return False

    def delete_custom_preset(self, name: str) -> bool:
        """Delete a custom preset.

        Args:
            name: Preset name to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.presets:
                return False

            preset = self.presets[name]

            # Only allow deletion of custom presets
            if preset.category != "Custom":
                logger.warning(f"Cannot delete non-custom preset: {name}")
                return False

            # Remove from presets
            del self.presets[name]

            # Remove from category
            if name in self.categories.get("Custom", []):
                self.categories["Custom"].remove(name)

            # Save changes
            self._save_custom_presets()
            return True
        except Exception as e:
            logger.error(f"Failed to delete custom preset: {e}")
            return False

    def update_custom_preset(self, name: str, description: str = None,
                            components: List[Dict[str, Any]] = None) -> bool:
        """Update an existing custom preset.

        Args:
            name: Preset name
            description: New description (optional)
            components: New components (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.presets:
                return False

            preset = self.presets[name]
            if preset.category != "Custom":
                logger.warning(f"Cannot update non-custom preset: {name}")
                return False

            # Delete and re-add with new data
            old_preset = self.presets[name]
            self.delete_custom_preset(name)

            return self.add_custom_preset(
                name=name,
                description=description or old_preset.description,
                components=components if components is not None else [],
                example_input=old_preset.example_input,
                example_output=old_preset.example_output,
                difficulty=old_preset.difficulty,
                tags=old_preset.tags
            )
        except Exception as e:
            logger.error(f"Failed to update custom preset: {e}")
            return False

    def get_custom_presets(self) -> Dict[str, RewardPreset]:
        """Get all custom presets.

        Returns:
            Dictionary of custom preset name to RewardPreset
        """
        return {name: preset for name, preset in self.presets.items()
                if preset.category == "Custom"}

    # Helper methods to create specific reward configurations
    def _create_scientific_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        builder.add_numerical_reward("numerical_accuracy", tolerance=0.01, relative=True, weight=0.6)
        builder.add_format_reward("units", pattern=r"\d+\s*[a-zA-Z]+/?[a-zA-Z]*", weight=0.2)
        builder.add_length_reward("explanation", min_length=20, max_length=200, weight=0.2)
        return builder

    def _create_algorithm_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        builder.add_format_reward("code_block", pattern=r"```[\w]*\n.*?\n```", weight=0.3)
        builder.add_format_reward("function_def", pattern=r"def\s+\w+\s*\(|function\s+\w+\s*\(", weight=0.3)
        builder.add_format_reward("efficiency_comment", pattern=r"#.*O\(|//.*O\(", weight=0.2)
        builder.add_length_reward("code_length", min_length=30, max_length=500, weight=0.2)
        return builder


    def _create_documentation_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Match headers anywhere in text (multiline mode handled by FormatReward)
        builder.add_format_reward("headers", pattern=r"#{1,6}\s+.+", weight=0.3)
        # Match list items anywhere in text
        builder.add_format_reward("lists", pattern=r"(?:^|\n)[\-\*]\s+.+", weight=0.2)
        # Match code blocks (non-greedy for multiple blocks)
        builder.add_format_reward("code_examples", pattern=r"```.*?```", weight=0.3)
        builder.add_length_reward("detail", min_length=50, max_length=400, weight=0.2)
        return builder


    def _create_explanatory_qa_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        builder.add_length_reward("detailed", min_length=50, max_length=300, optimal_length=150, weight=0.3)
        # Case-insensitive matching for common explanatory phrases
        builder.add_format_reward("examples", pattern=r"(?i)(for example|such as|like|e\.g\.|i\.e\.)", weight=0.3)
        # Match structured elements (works with multiline)
        builder.add_format_reward("structured", pattern=r"(?:First|Second|Finally|\d+[\.\)]\s)", weight=0.4)
        return builder

    def _create_instruction_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Match numbered steps anywhere (beginning of line or after newline)
        builder.add_format_reward("numbered_steps", pattern=r"(?:^|\n)\d+[\.\)]\s+", weight=0.5)
        builder.add_length_reward("step_length", min_length=20, max_length=200, weight=0.2)
        # Match action verbs after numbered steps
        builder.add_format_reward("action_verbs", pattern=r"(?:^|\n)\d+[\.\)]\s+[A-Z][a-z]+", weight=0.3)
        return builder


    def _create_data_analysis_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # More comprehensive number matching including decimals and commas
        builder.add_format_reward("numbers_present", pattern=r"\d+(?:[,\.]\d+)*(?:%|[KMB])?", weight=0.4)
        # Match various metric formats
        builder.add_format_reward("metrics", pattern=r"(?:\d+(?:\.\d+)?%|\$\d+(?:,\d{3})*(?:\.\d+)?|[+\-]\d+(?:\.\d+)?)", weight=0.3)
        builder.add_length_reward("insight_length", min_length=30, max_length=200, weight=0.3)
        return builder


    def _create_technical_analysis_reward(self) -> 'CustomRewardBuilder':
        """Create reward for technical analysis signal classification.

        This reward prioritizes:
        1. Signal accuracy (85%) - with partial credit for direction (THE ONLY THING THAT MATTERS)
        2. Format compliance (10%) - proper <signal> tags present
        3. Length penalty (5%) - discourage max-length rambling

        The signal accuracy component gives partial credit when the direction
        is correct but strength is wrong (e.g., WEAK_BUY instead of STRONG_BUY).

        Analysis content is NOT rewarded - only the signal output matters.
        """
        builder = CustomRewardBuilder()

        # Signal accuracy - with partial credit for correct direction
        # This is the ONLY metric that matters for trading signals
        builder.add_signal_accuracy(
            "signal_accuracy",
            valid_signals=["STRONG_BUY", "WEAK_BUY", "HOLD", "WEAK_SELL", "STRONG_SELL"],
            direction_match_score=0.70,  # 70% credit for correct direction, wrong strength
            weight=0.85  # Massively increased - this is what we care about
        )

        # Template validation - just ensure <signal> tags are present
        # We don't care about <analysis> tags anymore
        builder.add_template_validation(
            "template_structure",
            section_tags=["signal"],
            required_sections=["signal"],
            order_matters=False,
            weight=0.10  # Reduced - just basic format checking
        )

        # Length reward - discourage max-length outputs that hit token limit
        # Keep responses concise
        builder.add_length_reward(
            "response_length",
            min_length=10,
            max_length=300,
            optimal_length=50,
            weight=0.05  # Minimal weight - just prevent extreme rambling
        )

        return builder

    def _create_logical_reasoning_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Case-insensitive matching for logical connectives
        builder.add_format_reward("reasoning_words", pattern=r"(?i)(therefore|thus|because|since|if\s+\w+.*\s+then)", weight=0.4)
        builder.add_length_reward("reasoning_length", min_length=30, max_length=200, weight=0.3)
        # Match conclusion indicators
        builder.add_format_reward("conclusion", pattern=r"(?i)(conclude[sd]?|conclusion|therefore|thus|hence)", weight=0.3)
        return builder

    def _create_chain_of_thought_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Match step markers and sequence words
        builder.add_format_reward("step_markers", pattern=r"(?i)(Step \d+|(?:^|\n)\d+[\.\)]\s|First|Next|Then|Finally)", weight=0.5)
        builder.add_length_reward("thought_process", min_length=50, max_length=300, weight=0.2)
        # Match reasoning process indicators
        builder.add_format_reward("reasoning", pattern=r"(?i)(Let me|Let's|I need to|This means|So\s+|Therefore)", weight=0.3)
        return builder


    def _create_markdown_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Match markdown headers anywhere in text
        builder.add_format_reward("headers", pattern=r"(?:^|\n)#{1,6}\s+.+", weight=0.3)
        # Match list markers
        builder.add_format_reward("lists", pattern=r"(?:^|\n)[\-\*\+]\s+", weight=0.2)
        # Match code blocks (non-greedy)
        builder.add_format_reward("code_blocks", pattern=r"```[\s\S]*?```", weight=0.3)
        # Match emphasis (bold, italic)
        builder.add_format_reward("emphasis", pattern=r"(?:\*\*[^*]+\*\*|__[^_]+__|(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_))", weight=0.2)
        return builder

    def _create_citation_reward(self) -> 'CustomRewardBuilder':
        builder = CustomRewardBuilder()
        # Match author name patterns (various formats)
        builder.add_format_reward("author", pattern=r"[A-Z][a-z]+(?:,\s+[A-Z]\.?|,\s+[A-Z][a-z]+|(?:\s+[A-Z]\.)+)", weight=0.3)
        # Match publication year in parentheses
        builder.add_format_reward("year", pattern=r"\((?:19|20)\d{2}\)", weight=0.2)
        # Match italicized or emphasized titles
        builder.add_format_reward("title", pattern=r"(?:\*[^*]+\*|_[^_]+_)", weight=0.2)
        # Match URLs or DOIs
        builder.add_format_reward("url", pattern=r"(?:https?://|doi:|DOI:)", weight=0.3)
        return builder


class CustomRewardBuilder:
    """Builder for creating and combining custom reward functions."""

    def __init__(self):
        """Initialize reward builder."""
        self.rewards: List[RewardFunction] = []
        self.weights: List[float] = []

    def add_reward(self, reward: RewardFunction, weight: float = 1.0):
        """Add a reward function.

        Args:
            reward: Reward function
            weight: Weight for this reward
        """
        self.rewards.append(reward)
        self.weights.append(weight)

    def add_binary_reward(self, name: str, regex_pattern: Optional[str] = None, weight: float = 1.0):
        """Add binary reward function."""
        config = RewardConfig(
            name=name,
            description=f"Binary reward: {name}",
            type=RewardType.BINARY,
            use_regex=bool(regex_pattern),
            regex_pattern=regex_pattern,
            weight=weight
        )
        self.add_reward(BinaryReward(config), weight)

    def add_numerical_reward(self, name: str, tolerance: float = 1e-6, relative: bool = False, weight: float = 1.0):
        """Add numerical reward function."""
        config = RewardConfig(
            name=name,
            description=f"Numerical reward: {name}",
            type=RewardType.CONTINUOUS,
            extract_number=True,
            number_tolerance=tolerance,
            relative_tolerance=relative,
            weight=weight
        )
        self.add_reward(NumericalReward(config), weight)

    def add_length_reward(self, name: str,
                         min_length: Optional[int] = None,
                         max_length: Optional[int] = None,
                         optimal_length: Optional[int] = None,
                         weight: float = 1.0):
        """Add length-based reward function."""
        config = RewardConfig(
            name=name,
            description=f"Length reward: {name}",
            type=RewardType.CONTINUOUS,
            min_length=min_length,
            max_length=max_length,
            optimal_length=optimal_length,
            weight=weight
        )
        self.add_reward(LengthReward(config), weight)

    def add_format_reward(self, name: str, pattern: str, weight: float = 1.0):
        """Add format compliance reward."""
        config = RewardConfig(
            name=name,
            description=f"Format reward: {name}",
            type=RewardType.BINARY,
            use_regex=True,
            regex_pattern=pattern,
            weight=weight
        )
        self.add_reward(FormatReward(config), weight)

    def add_custom_reward(self, name: str, code: str, weight: float = 1.0, safe_mode: bool = True):
        """Add custom reward function."""
        config = RewardConfig(
            name=name,
            description=f"Custom reward: {name}",
            type=RewardType.CUSTOM,
            custom_code=code,
            safe_mode=safe_mode,
            weight=weight
        )
        self.add_reward(CustomReward(config), weight)

    def add_template_validation(self, name: str,
                               section_tags: List[str],
                               required_sections: Optional[List[str]] = None,
                               order_matters: bool = False,
                               weight: float = 1.0):
        """Add template validation reward."""
        config = RewardConfig(
            name=name,
            description=f"Template validation: {name}",
            type=RewardType.BINARY,
            section_tags=section_tags,
            required_sections=required_sections or section_tags,
            order_matters=order_matters,
            weight=weight
        )
        self.add_reward(TemplateValidationReward(config), weight)

    def add_multi_choice_validation(self, name: str,
                                   valid_choices: List[str],
                                   case_sensitive: bool = False,
                                   exact_match: bool = True,
                                   weight: float = 1.0):
        """Add multi-choice validation reward."""
        config = RewardConfig(
            name=name,
            description=f"Multi-choice validation: {name}",
            type=RewardType.BINARY,
            valid_choices=valid_choices,
            case_sensitive=case_sensitive,
            exact_match=exact_match,
            weight=weight
        )
        self.add_reward(MultiChoiceValidationReward(config), weight)

    def add_section_content(self, name: str,
                           section_tag: str,
                           min_words: Optional[int] = None,
                           max_words: Optional[int] = None,
                           required_keywords: Optional[List[str]] = None,
                           weight: float = 1.0):
        """Add section content validation reward."""
        config = RewardConfig(
            name=name,
            description=f"Section content: {name}",
            type=RewardType.CONTINUOUS,
            section_tag=section_tag,
            min_words=min_words,
            max_words=max_words,
            required_keywords=required_keywords,
            weight=weight
        )
        self.add_reward(SectionContentReward(config), weight)

    def add_sequential_pattern(self, name: str,
                              patterns: List[str],
                              strict_order: bool = True,
                              weight: float = 1.0):
        """Add sequential pattern reward."""
        config = RewardConfig(
            name=name,
            description=f"Sequential pattern: {name}",
            type=RewardType.BINARY,
            patterns=patterns,
            strict_order=strict_order,
            weight=weight
        )
        self.add_reward(SequentialPatternReward(config), weight)

    def add_signal_accuracy(self, name: str,
                           valid_signals: List[str],
                           direction_match_score: float = 0.70,
                           weight: float = 1.0):
        """Add signal accuracy reward with partial credit for direction.

        This reward function is designed for trading signal classification tasks
        where getting the direction right (BUY vs SELL) is more important than
        getting the exact strength (STRONG vs WEAK).

        Partial credit scoring:
            - Perfect match: 1.00
            - Right direction, wrong strength: direction_match_score (default 0.70)
            - Opposite direction: 0.00

        Args:
            name: Reward component name
            valid_signals: List of valid signal names (e.g., ['STRONG_BUY', 'WEAK_BUY', ...])
            direction_match_score: Score for correct direction but wrong strength (0.0-1.0)
            weight: Weight for this reward component
        """
        config = RewardConfig(
            name=name,
            description=f"Signal accuracy with partial credit: {name}",
            type=RewardType.CONTINUOUS,
            valid_choices=valid_signals,
            signal_direction_match_score=direction_match_score,
            weight=weight
        )
        self.add_reward(SignalAccuracyReward(config), weight)

    def compute_total_reward(self,
                            instruction: str,
                            generated: str,
                            reference: Optional[str] = None,
                            tokenizer: Optional[Any] = None) -> Tuple[float, Dict[str, float]]:
        """Compute total reward from all components with brevity bonus.

        Uses a two-pass approach:
        1. First pass: compute all component rewards independently
        2. Apply brevity bonus to length component if base quality is high
        3. Compute final weighted sum

        This encourages concise responses when quality is high, preventing the model
        from gaming rewards by being short-but-wrong.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Tuple of (total_reward, component_rewards)
        """
        if not self.rewards:
            return 0.0, {}

        # First pass: compute all component rewards
        component_rewards = {}
        for reward_func in self.rewards:
            reward_value = reward_func.compute(instruction, generated, reference, tokenizer)
            component_rewards[reward_func.config.name] = reward_value

        # Calculate base quality from non-length components
        # This prevents short-but-wrong responses from getting brevity bonus
        quality_components = {k: v for k, v in component_rewards.items()
                            if 'length' not in k.lower()}

        if quality_components:
            base_quality = sum(quality_components.values()) / len(quality_components)
        else:
            base_quality = 0.0

        # Apply brevity bonus to length component if quality is high
        quality_threshold = 0.7
        length_component_name = None

        # Find the length component (typically 'response_length' or similar)
        for name in component_rewards:
            if 'length' in name.lower():
                length_component_name = name
                break

        if length_component_name and base_quality > quality_threshold:
            # Use token count if tokenizer provided, otherwise word count
            if tokenizer is not None:
                try:
                    count = len(tokenizer.encode(generated, add_special_tokens=False))
                except Exception:
                    count = len(generated.split())
            else:
                count = len(generated.split())

            # Brevity bonus: shorter is better within 30-150 unit range (tokens or words)
            # Up to +15% bonus for responses closer to 30 units
            min_brevity_units = 30
            max_brevity_units = 150

            if min_brevity_units <= count <= max_brevity_units:
                # Linear interpolation: 150 units = 1.0x, 30 units = 1.15x
                brevity_multiplier = 1.0 + (max_brevity_units - count) / (max_brevity_units - min_brevity_units) * 0.15
                component_rewards[length_component_name] *= brevity_multiplier

        # Compute final weighted sum
        total_weight = sum(self.weights)
        weighted_sum = sum(component_rewards[reward_func.config.name] * weight
                          for reward_func, weight in zip(self.rewards, self.weights))

        total_reward = weighted_sum / total_weight if total_weight > 0 else 0.0

        return total_reward, component_rewards

    def test_reward(self,
                   test_cases: List[Tuple[str, str, str, float]]) -> Dict[str, Any]:
        """Test reward function with examples.

        Args:
            test_cases: List of (instruction, generated, reference, expected_reward)

        Returns:
            Test results
        """
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for instruction, generated, reference, expected in test_cases:
            total, components = self.compute_total_reward(instruction, generated, reference)

            # Check if reward is close to expected
            is_close = abs(total - expected) < 0.01

            result = {
                'instruction': instruction[:50] + '...' if len(instruction) > 50 else instruction,
                'generated': generated[:50] + '...' if len(generated) > 50 else generated,
                'expected': expected,
                'actual': total,
                'components': components,
                'passed': is_close
            }

            results['details'].append(result)

            if is_close:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def save_config(self, path: str):
        """Save reward configuration to file."""
        config = {
            'rewards': [
                {
                    'config': asdict(r.config),
                    'weight': w,
                    'class': r.__class__.__name__
                }
                for r, w in zip(self.rewards, self.weights)
            ]
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def get_component_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all reward components.

        Returns:
            List of dictionaries containing component details
        """
        if not self.rewards:
            return []

        total_weight = sum(self.weights) if self.weights else 1.0
        components = []

        for reward_func, weight in zip(self.rewards, self.weights):
            config = reward_func.config

            # Determine component type
            component_type = config.type.value if hasattr(config.type, 'value') else str(config.type)

            # Build component details
            details = {
                'name': config.name,
                'description': config.description,
                'type': component_type,
                'class': reward_func.__class__.__name__,
                'weight': weight,
                'weight_percentage': (weight / total_weight * 100) if total_weight > 0 else 0,
                'parameters': {}
            }

            # Add type-specific parameters
            if isinstance(reward_func, NumericalReward):
                details['parameters'] = {
                    'tolerance': config.number_tolerance,
                    'relative': config.relative_tolerance
                }
            elif isinstance(reward_func, LengthReward):
                # Include all parameters, showing "Excluded" for None values
                details['parameters'] = {
                    'min_length': config.min_length if config.min_length is not None else "Excluded",
                    'max_length': config.max_length if config.max_length is not None else "Excluded",
                    'optimal_length': config.optimal_length if config.optimal_length is not None else "Excluded"
                }
            elif isinstance(reward_func, FormatReward):
                details['parameters'] = {
                    'pattern': config.regex_pattern
                }
            elif isinstance(reward_func, BinaryReward):
                details['parameters'] = {
                    'uses_regex': config.use_regex,
                    'pattern': config.regex_pattern
                }
            elif isinstance(reward_func, TemplateValidationReward):
                details['parameters'] = {
                    'section_tags': config.section_tags,
                    'required_sections': config.required_sections,
                    'order_matters': config.order_matters
                }
            elif isinstance(reward_func, MultiChoiceValidationReward):
                details['parameters'] = {
                    'valid_choices': config.valid_choices,
                    'case_sensitive': config.case_sensitive,
                    'exact_match': config.exact_match
                }
            elif isinstance(reward_func, SectionContentReward):
                # Show all parameters, with "Excluded" for optional ones
                details['parameters'] = {
                    'section_tag': config.section_tag,
                    'min_words': config.min_words if (config.min_words is not None and config.min_words > 0) else "Excluded",
                    'max_words': config.max_words if (config.max_words is not None and config.max_words != float('inf')) else "Excluded",
                    'required_keywords': config.required_keywords[:5] if config.required_keywords else "Excluded"
                }
            elif isinstance(reward_func, SequentialPatternReward):
                details['parameters'] = {
                    'patterns': config.patterns[:3] if config.patterns else [],  # Show first 3
                    'strict_order': config.strict_order
                }
            elif isinstance(reward_func, SignalAccuracyReward):
                details['parameters'] = {
                    'valid_signals': config.valid_choices,
                    'direction_match_score': config.signal_direction_match_score
                }

            components.append(details)

        return components

    @classmethod
    def load_config(cls, path: str) -> 'CustomRewardBuilder':
        """Load reward configuration from file."""
        with open(path, 'r') as f:
            config = json.load(f)

        builder = cls()

        for reward_info in config['rewards']:
            reward_config = RewardConfig(**reward_info['config'])
            weight = reward_info['weight']
            class_name = reward_info['class']

            # Create appropriate reward instance
            if class_name == 'BinaryReward':
                reward = BinaryReward(reward_config)
            elif class_name == 'NumericalReward':
                reward = NumericalReward(reward_config)
            elif class_name == 'LengthReward':
                reward = LengthReward(reward_config)
            elif class_name == 'FormatReward':
                reward = FormatReward(reward_config)
            elif class_name == 'CustomReward':
                reward = CustomReward(reward_config)
            else:
                logger.warning(f"Unknown reward class: {class_name}")
                continue

            builder.add_reward(reward, weight)

        return builder


class RewardTester:
    """Test reward functions with sample inputs/outputs."""

    def __init__(self, reward_builder: CustomRewardBuilder):
        """Initialize tester with a reward builder.

        Args:
            reward_builder: The reward builder to test
        """
        self.reward_builder = reward_builder

    def test_single(self,
                   instruction: str,
                   generated: str,
                   reference: Optional[str] = None,
                   tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Test reward on a single example.

        Args:
            instruction: Input instruction
            generated: Generated response
            reference: Reference response (optional)
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Dictionary with total reward and component breakdown
        """
        total_reward, component_rewards = self.reward_builder.compute_total_reward(
            instruction, generated, reference, tokenizer
        )

        return {
            'total_reward': total_reward,
            'components': component_rewards,
            'instruction': instruction,
            'generated': generated,
            'reference': reference
        }

    def test_batch(self, test_cases: List[Dict[str, str]], tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Test reward on multiple examples.

        Args:
            test_cases: List of dicts with 'instruction', 'generated', 'reference' keys
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Dictionary with statistics and individual results
        """
        results = []
        rewards = []

        for case in test_cases:
            result = self.test_single(
                case.get('instruction', ''),
                case.get('generated', ''),
                case.get('reference'),
                tokenizer
            )
            results.append(result)
            rewards.append(result['total_reward'])

        # Calculate statistics
        rewards_array = np.array(rewards)

        return {
            'results': results,
            'statistics': {
                'mean': float(np.mean(rewards_array)),
                'std': float(np.std(rewards_array)),
                'min': float(np.min(rewards_array)),
                'max': float(np.max(rewards_array)),
                'median': float(np.median(rewards_array))
            },
            'distribution': self._calculate_distribution(rewards_array)
        }

    def _calculate_distribution(self, rewards: np.ndarray) -> Dict[str, int]:
        """Calculate reward distribution in buckets."""
        buckets = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }

        for reward in rewards:
            if reward <= 0.2:
                buckets['0.0-0.2'] += 1
            elif reward <= 0.4:
                buckets['0.2-0.4'] += 1
            elif reward <= 0.6:
                buckets['0.4-0.6'] += 1
            elif reward <= 0.8:
                buckets['0.6-0.8'] += 1
            else:
                buckets['0.8-1.0'] += 1

        return buckets

    def compare_responses(self,
                         instruction: str,
                         responses: List[str],
                         reference: Optional[str] = None,
                         tokenizer: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Compare multiple responses for the same instruction.

        Args:
            instruction: Input instruction
            responses: List of different responses to compare
            reference: Reference response (optional)
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            List of results sorted by total reward (highest first)
        """
        results = []

        for i, response in enumerate(responses):
            result = self.test_single(instruction, response, reference, tokenizer)
            result['response_index'] = i
            results.append(result)

        # Sort by total reward (highest first)
        results.sort(key=lambda x: x['total_reward'], reverse=True)

        return results

    def test_with_model(self,
                       instruction: str,
                       reference: Optional[str],
                       model_tester,
                       model_type: str,
                       model_key: str,
                       generation_config: Optional[Dict[str, Any]] = None,
                       session_info = None,
                       system_prompt: Optional[str] = None,
                       tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Test reward function by generating a response with a model and scoring it.

        Args:
            instruction: Input instruction/prompt
            reference: Reference response for comparison (optional)
            model_tester: ModelTester instance for generation
            model_type: "trained" or "base"
            model_key: Session ID for trained models or model name for base models
            generation_config: Generation parameters (temperature, max_tokens, etc.)
            session_info: SessionInfo object with training configuration
            system_prompt: Custom system prompt to use (overrides session config)
            tokenizer: Optional tokenizer for token-based metrics

        Returns:
            Dictionary with:
                - instruction: The input instruction
                - generated: Model-generated response
                - reference: Reference response (if provided)
                - total_reward: Overall reward score
                - components: Breakdown of component scores
                - generation_metadata: Info about the generation (tokens, time, etc.)
        """
        try:
            # Import TestConfig if generation config provided
            from core.model_tester import TestConfig

            # Convert generation config to TestConfig if provided
            test_config = None
            if generation_config:
                test_config = TestConfig(
                    temperature=generation_config.get('temperature', 0.7),
                    max_new_tokens=generation_config.get('max_new_tokens', 512),
                    top_p=generation_config.get('top_p', 0.95),
                    top_k=generation_config.get('top_k', 50),
                    repetition_penalty=generation_config.get('repetition_penalty', 1.0),
                    do_sample=generation_config.get('do_sample', True),
                    num_beams=generation_config.get('num_beams', 1)
                )

            # Generate response using the model
            generation_result = model_tester.generate_response(
                prompt=instruction,
                model_type=model_type,
                model_key=model_key,
                config=test_config,
                use_chat_template=True,
                session_info=session_info,
                override_system_prompt=system_prompt if system_prompt else None
            )

            if not generation_result.get('success', False):
                return {
                    'success': False,
                    'error': generation_result.get('error', 'Unknown generation error')
                }

            generated_text = generation_result.get('response', '')

            # Score the generated response using the reward function
            total_reward, component_rewards = self.reward_builder.compute_total_reward(
                instruction=instruction,
                generated=generated_text,
                reference=reference,
                tokenizer=tokenizer
            )

            return {
                'success': True,
                'instruction': instruction,
                'generated': generated_text,
                'reference': reference,
                'total_reward': total_reward,
                'components': component_rewards,
                'generation_metadata': {
                    'model_type': model_type,
                    'model_key': model_key,
                    'tokens_generated': generation_result.get('tokens_generated', 0),
                    'generation_time': generation_result.get('generation_time', 0)
                }
            }

        except Exception as e:
            logger.error(f"Failed to test with model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def visualize_components(self, test_result: Dict[str, Any]) -> str:
        """Create a text visualization of component rewards.

        Args:
            test_result: Result from test_single

        Returns:
            ASCII bar chart of component rewards
        """
        components = test_result['components']
        total = test_result['total_reward']

        viz = []
        viz.append(f"Total Reward: {total:.3f}")
        viz.append("-" * 40)

        for name, value in components.items():
            bar_length = int(value * 20)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            viz.append(f"{name:20s} {bar} {value:.3f}")

        return "\n".join(viz)


class RewardTemplate:
    """Task-specific template that configures entire training setup."""

    def __init__(self,
                 name: str,
                 description: str,
                 reward_preset: str,
                 dataset_format: str,
                 example_dataset: List[Dict[str, str]],
                 recommended_settings: Dict[str, Any],
                 tips: List[str]):
        """Initialize reward template.

        Args:
            name: Template name
            description: What this template is for
            reward_preset: Name of reward preset to use
            dataset_format: Expected dataset format description
            example_dataset: Small example dataset
            recommended_settings: Recommended training hyperparameters
            tips: Usage tips
        """
        self.name = name
        self.description = description
        self.reward_preset = reward_preset
        self.dataset_format = dataset_format
        self.example_dataset = example_dataset
        self.recommended_settings = recommended_settings
        self.tips = tips

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'reward_preset': self.reward_preset,
            'dataset_format': self.dataset_format,
            'example_dataset': self.example_dataset,
            'recommended_settings': self.recommended_settings,
            'tips': self.tips
        }


class RewardTemplateLibrary:
    """Library of task-specific templates."""

    def __init__(self):
        """Initialize template library."""
        self.templates: Dict[str, RewardTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default templates."""
        # Math problem solving template
        self.templates['math_problem_solving'] = RewardTemplate(
            name="Math Problem Solving",
            description="Train models to solve mathematical problems with step-by-step solutions",
            reward_preset="Mathematical Problem Solving",
            dataset_format="Each sample should have 'instruction' (the problem) and 'response' (solution with \\boxed{} answer)",
            example_dataset=[
                {
                    "instruction": "Solve for x: 2x + 5 = 13",
                    "response": "To solve 2x + 5 = 13:\n1) Subtract 5 from both sides: 2x = 8\n2) Divide by 2: x = 4\n\\boxed{4}"
                },
                {
                    "instruction": "What is 15% of 80?",
                    "response": "To find 15% of 80:\n15% = 0.15\n0.15 × 80 = 12\n\\boxed{12}"
                }
            ],
            recommended_settings={
                'learning_rate': 5e-5,
                'num_epochs': 3,
                'batch_size': 4,
                'num_generations': 2
            },
            tips=[
                "Ensure answers are in \\boxed{} format",
                "Include step-by-step reasoning",
                "Use consistent mathematical notation"
            ]
        )

        # Code generation template
        self.templates['code_generation'] = RewardTemplate(
            name="Code Generation",
            description="Train models to generate clean, well-documented code",
            reward_preset="Code Generation",
            dataset_format="Each sample should have 'instruction' (coding task) and 'response' (code with explanation)",
            example_dataset=[
                {
                    "instruction": "Write a Python function to check if a number is prime",
                    "response": "```python\ndef is_prime(n):\n    \"\"\"Check if a number is prime.\"\"\"\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```\n\nThis function checks divisibility up to the square root of n for efficiency."
                }
            ],
            recommended_settings={
                'learning_rate': 3e-5,
                'num_epochs': 2,
                'batch_size': 2,
                'max_length': 512
            },
            tips=[
                "Include code in markdown code blocks",
                "Add comments and docstrings",
                "Provide brief explanations after code"
            ]
        )

        # Q&A template
        self.templates['question_answering'] = RewardTemplate(
            name="Question Answering",
            description="Train models to provide accurate, concise answers",
            reward_preset="Factual Q&A",
            dataset_format="Each sample should have 'instruction' (question) and 'response' (direct answer)",
            example_dataset=[
                {
                    "instruction": "What is the capital of France?",
                    "response": "The capital of France is Paris."
                },
                {
                    "instruction": "Who wrote Romeo and Juliet?",
                    "response": "Romeo and Juliet was written by William Shakespeare."
                }
            ],
            recommended_settings={
                'learning_rate': 5e-5,
                'num_epochs': 2,
                'batch_size': 8,
                'max_length': 128
            },
            tips=[
                "Keep answers concise and direct",
                "Use complete sentences",
                "Avoid unnecessary elaboration"
            ]
        )

        # Add more templates as needed

    def get_template(self, name: str) -> Optional[RewardTemplate]:
        """Get template by name."""
        return self.templates.get(name)

    def get_all_templates(self) -> Dict[str, RewardTemplate]:
        """Get all templates."""
        return self.templates


# Preset reward functions
def create_math_reward() -> CustomRewardBuilder:
    """Create reward for mathematical problems."""
    builder = CustomRewardBuilder()

    # Numerical accuracy
    builder.add_numerical_reward("numerical_accuracy", tolerance=1e-6, weight=0.7)

    # Format compliance (answer in a box)
    builder.add_format_reward(
        "answer_format",
        pattern=r"\\boxed\{[^}]+\}|Final Answer:.*|Answer:.*",
        weight=0.2
    )

    # Length penalty (prefer concise solutions)
    builder.add_length_reward(
        "length",
        min_length=10,
        max_length=500,
        optimal_length=100,
        weight=0.1
    )

    return builder


def create_code_reward() -> CustomRewardBuilder:
    """Create reward for code generation."""
    builder = CustomRewardBuilder()

    # Code block detection
    builder.add_format_reward(
        "code_block",
        pattern=r"```[\w]*\n.*?\n```",
        weight=0.3
    )

    # Function definition detection
    builder.add_format_reward(
        "function_def",
        pattern=r"def\s+\w+\s*\(|function\s+\w+\s*\(|const\s+\w+\s*=",
        weight=0.3
    )

    # Comments detection
    builder.add_format_reward(
        "comments",
        pattern=r"#.*|//.*|/\*.*?\*/",
        weight=0.2
    )

    # Length reward (prefer reasonable length)
    builder.add_length_reward(
        "length",
        min_length=20,
        max_length=1000,
        optimal_length=200,
        weight=0.2
    )

    return builder


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING SIGNAL ACCURACY REWARD WITH PARTIAL CREDIT")
    print("=" * 80)

    # Create a signal accuracy reward instance
    from core.custom_rewards import SignalAccuracyReward, RewardConfig, RewardType

    config = RewardConfig(
        name="test_signal_accuracy",
        description="Test signal accuracy with partial credit",
        type=RewardType.CONTINUOUS,
        valid_choices=["STRONG_BUY", "WEAK_BUY", "HOLD", "WEAK_SELL", "STRONG_SELL"]
    )

    signal_reward = SignalAccuracyReward(config)

    # Test cases: (instruction, generated, reference, expected_score)
    test_cases = [
        # Perfect matches
        ("", "<analysis>Good</analysis><signal>STRONG_BUY</signal>",
         "<analysis>Good</analysis><signal>STRONG_BUY</signal>", 1.00, "Perfect match"),

        # Right direction, wrong strength (70% credit)
        ("", "<analysis>Good</analysis><signal>WEAK_BUY</signal>",
         "<analysis>Good</analysis><signal>STRONG_BUY</signal>", 0.70, "Right direction (BUY), wrong strength"),

        ("", "<analysis>Good</analysis><signal>STRONG_SELL</signal>",
         "<analysis>Good</analysis><signal>WEAK_SELL</signal>", 0.70, "Right direction (SELL), wrong strength"),

        # HOLD scenarios
        ("", "<analysis>Good</analysis><signal>HOLD</signal>",
         "<analysis>Good</analysis><signal>WEAK_BUY</signal>", 0.50, "HOLD when should be WEAK"),

        ("", "<analysis>Good</analysis><signal>HOLD</signal>",
         "<analysis>Good</analysis><signal>STRONG_BUY</signal>", 0.40, "HOLD when should be STRONG"),

        # Opposite direction (0% credit)
        ("", "<analysis>Good</analysis><signal>STRONG_BUY</signal>",
         "<analysis>Good</analysis><signal>WEAK_SELL</signal>", 0.00, "Opposite direction BUY vs SELL"),

        ("", "<analysis>Good</analysis><signal>WEAK_SELL</signal>",
         "<analysis>Good</analysis><signal>STRONG_BUY</signal>", 0.00, "Opposite direction SELL vs BUY"),

        # Format variations (with spaces)
        ("", "<analysis>Good</analysis><signal>STRONG BUY</signal>",
         "<analysis>Good</analysis><signal>STRONG_BUY</signal>", 1.00, "Space in signal name (normalized)"),
    ]

    print("\nRunning test cases...\n")

    passed = 0
    failed = 0

    for instruction, generated, reference, expected, description in test_cases:
        actual = signal_reward.compute(instruction, generated, reference)
        is_correct = abs(actual - expected) < 0.01

        status = "✅ PASS" if is_correct else "❌ FAIL"
        passed += 1 if is_correct else 0
        failed += 0 if is_correct else 1

        print(f"{status} | {description}")
        print(f"     Expected: {expected:.2f}, Actual: {actual:.2f}")
        if not is_correct:
            print(f"     ⚠️  MISMATCH!")
        print()

    print("=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed, {failed} failed")
    print("=" * 80)

    # Also test the math reward (original test)
    print("\n" + "=" * 80)
    print("TESTING MATH REWARD (ORIGINAL)")
    print("=" * 80)

    builder = create_math_reward()

    math_test_cases = [
        ("What is 2+2?", "The answer is 4", "4", 0.7),
        ("Solve x^2 = 16", "x = 4 or x = -4\n\\boxed{4, -4}", "x = 4, x = -4", 0.9),
        ("Calculate 10/3", "3.333333", "3.333333", 0.7),
    ]

    results = builder.test_reward(math_test_cases)
    print(f"\nTest results: {results['passed']}/{len(math_test_cases)} passed")

    for detail in results['details']:
        print(f"\nInstruction: {detail['instruction']}")
        print(f"Expected: {detail['expected']:.2f}, Actual: {detail['actual']:.2f}")
        print(f"Components: {detail['components']}")
